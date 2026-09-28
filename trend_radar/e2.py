"""E2 기사 관련성 실험 (v2 설계 3장, 기획안 5장·9장).

임의 판정 최소화 원칙
- 학습 라벨: 약한 지도학습. 서로 다른 근거의 규칙(라벨 함수)을 두고, 라벨 모델이 각 규칙의 정확도를
  규칙 간 일치·충돌에서 추정한다 (Dawid-Skene 방식 EM). 사람·LLM의 기사별 판정을 쓰지 않는다.
- 평가: 학습에 쓰지 않은 외부·형식 신호(앵커)로 방향을 확인한다.
    콘텐츠 앵커: 제목 따옴표 표현 중 3건 이상, 한 기업 집중도 0.8 이상 (작품·서비스명)
    시장 앵커: 네이버 경제 섹션(sid=101) + 시세형 제목, 또는 특징주·공시 태그
- 판정 구간: Otsu 임계값(두 집단 분산 최소)으로 자르고, 모델 간 의견이 갈리면 REVIEW.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ABSTAIN, NEG, POS = 0, -1, 1

QUOTE = re.compile(r"[‘'\"“《〈『「<]([^’'\"”》〉』」>]{2,25})[’'\"”》〉』」>]")
PRICE_TITLE = re.compile(r"%\s*[↑↓]|[↑↓]\s*\d|상한가|하한가|목표가|목표주가|급등|급락|강세|약세|52주|신고가|신저가")
MARKET_TAGS = {"특징주", "주요공시", "오늘의 주요공시", "공시", "공시속보"}
# 다른 업종 신호 (콘텐츠와 무관한 업종 어휘)
OTHER_INDUSTRY = set("""
은행 보험 카드 증권사 대출 금리 채권 부동산 건설 아파트 분양 반도체 메모리 파운드리 배터리 이차전지 2차전지 양극재
바이오 제약 신약 임상 의료 병원 자동차 전기차 조선 방산 항공 여행 관광 화장품 식품 유통 편의점 백화점 통신 요금
석유 가스 전력 원전 에너지 철강 화학 코인 비트코인 가상자산 스테이블코인 정치 선거 대선 국회 검찰 재판
""".split())


# ---------------------------------------------------------------- 앵커 (평가 전용)

def quoted(title: str) -> list[str]:
    out = []
    for q in QUOTE.findall(title or ""):
        q = q.strip()
        if q.count(" ") >= 3 or re.search(r"(다|요|죠|까|네|나|자)$", q) or re.search(r"[?!…~]", q) \
                or re.fullmatch(r"[\d.,%]+", q):
            continue
        out.append(q)
    return out


def anchors(arts: pd.DataFrame, sid: pd.Series, min_articles: int = 3, min_conc: float = 0.8) -> tuple[pd.DataFrame, dict]:
    q = arts["title"].map(quoted)
    cnt = collections.Counter(x for l in q for x in set(l))
    cand = {x for x, n in cnt.items() if n >= min_articles}
    co = collections.defaultdict(collections.Counter)
    for l, cs in zip(q, arts["companies"]):
        for x in set(l) & cand:
            co[x].update(cs)
    conc = {x: c.most_common(1)[0][1] / cnt[x] for x, c in co.items()}
    works = {x for x in cand if conc.get(x, 0) >= min_conc}
    pos = q.map(lambda l: any(x in works for x in l))
    s = arts["gid"].map(sid)
    neg = ((s == "101") & arts["title"].str.contains(PRICE_TITLE)) | arts["tag"].isin(MARKET_TAGS)
    out = pd.DataFrame({"gid": arts["gid"], "anchor_content": pos.to_numpy(), "anchor_market": neg.to_numpy(),
                        "sid": s.to_numpy()})
    info = {"work_names": len(works), "anchor_content": int(pos.sum()), "anchor_market": int(neg.sum()),
            "anchor_both": int((pos & neg).sum()),
            "rule": {"min_articles": min_articles, "min_company_concentration": min_conc},
            "top_work_names": sorted(works, key=lambda x: -cnt[x])[:50]}
    return out, info


# ---------------------------------------------------------------- 라벨 함수 (학습 전용, 앵커 규칙과 겹치지 않음)

def labeling_functions(arts: pd.DataFrame, kw: list[list[str]], lab: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lc = cfg["layers"]
    cseed, mseed = set(lc["content_seed"]), set(lc["market_seed"])
    sets = [set(w) for w in kw]
    c = np.array([len(s & cseed) for s in sets])
    m = np.array([len(s & mseed) for s in sets])
    o = np.array([len(s & OTHER_INDUSTRY) for s in sets])
    title_c = np.array([any(w in t for w in (s & cseed)) for s, t in zip(sets, arts["title"])])
    strong = arts["strong"].to_numpy()
    related = (arts["tier"] == "related").to_numpy()
    share = lab["content_share"].to_numpy()
    cs, ms = lab["content_score"].to_numpy(), lab["market_score"].to_numpy()
    L = {
        "LF_content_seed_2plus": np.where(c >= 2, POS, ABSTAIN),
        "LF_content_seed_in_title": np.where(title_c, POS, ABSTAIN),
        "LF_content_leaning_score": np.where((cs >= 3) & (share >= 0.7), POS, ABSTAIN),
        "LF_market_seed_3plus_no_content": np.where((m >= 3) & (c == 0), NEG, ABSTAIN),
        "LF_market_leaning_score": np.where((ms >= 3) & (share <= 0.3), NEG, ABSTAIN),
        "LF_passing_mention_no_content": np.where(~strong & (c < 2), NEG, ABSTAIN),
        "LF_other_industry": np.where((o >= 2) & (c == 0), NEG, ABSTAIN),
        "LF_related_tier_no_content": np.where(related & (c < 2), NEG, ABSTAIN),
    }
    return pd.DataFrame(L)


def label_model(L: pd.DataFrame, iters: int = 100, smooth: float = 1.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Dawid-Skene(혼동행렬) EM. 규칙 j마다 P(투표 k | 관련), P(투표 k | 무관)을 따로 추정한다 (k = +, -, 기권).
    한쪽으로만 투표하는 규칙은 '자기 방향 클래스에서 더 자주 투표한다'는 제약으로 뒤집힌 해를 막는다.
    초기값은 다수결."""
    X = L.to_numpy()
    n, J = X.shape
    cats = (POS, NEG, ABSTAIN)
    onehot = np.stack([(X == k) for k in cats], axis=2).astype(float)  # n × J × 3
    polarity = np.array([int(np.sign(X[:, j][X[:, j] != 0].sum())) if (X[:, j] != 0).any() else 0 for j in range(J)])
    votes = (X == POS).sum(1) - (X == NEG).sum(1)
    p = np.where(votes > 0, 0.9, np.where(votes < 0, 0.1, 0.5))
    for _ in range(iters):
        # M: 클래스별 투표 분포 (라플라스 평활)
        th1 = (onehot * p[:, None, None]).sum(0) + smooth          # J × 3, 관련
        th0 = (onehot * (1 - p)[:, None, None]).sum(0) + smooth    # J × 3, 무관
        th1 /= th1.sum(1, keepdims=True)
        th0 /= th0.sum(1, keepdims=True)
        # 식별 제약: 규칙은 자기 방향 클래스에서 그 방향 투표 확률이 더 높다
        for j in range(J):
            k = 0 if polarity[j] > 0 else 1
            if polarity[j] > 0 and th1[j, k] < th0[j, k]:
                th1[j, k], th0[j, k] = th0[j, k], th1[j, k]
            if polarity[j] < 0 and th0[j, k] < th1[j, k]:
                th1[j, k], th0[j, k] = th0[j, k], th1[j, k]
            th1[j] /= th1[j].sum()
            th0[j] /= th0[j].sum()
        prior = float(np.clip(p.mean(), 0.02, 0.98))
        # E
        lp = np.log(prior) + (onehot * np.log(th1)[None]).sum((1, 2))
        ln = np.log(1 - prior) + (onehot * np.log(th0)[None]).sum((1, 2))
        p_new = 1 / (1 + np.exp(np.clip(ln - lp, -50, 50)))
        if np.max(np.abs(p_new - p)) < 1e-5:
            p = p_new
            break
        p = p_new
    k_own = np.where(polarity > 0, 0, 1)
    own_rel = th1[np.arange(J), k_own]
    own_irr = th0[np.arange(J), k_own]
    # 규칙이 투표했을 때 그 방향이 맞을 확률 (사후 정밀도)
    prec = np.where(polarity > 0, own_rel * prior / (own_rel * prior + own_irr * (1 - prior)),
                    own_irr * (1 - prior) / (own_irr * (1 - prior) + own_rel * prior))
    stats = pd.DataFrame({"lf": L.columns, "polarity": polarity, "coverage": (X != ABSTAIN).mean(0),
                          "P(vote|relevant)": own_rel, "P(vote|irrelevant)": own_irr, "estimated_precision": prec})
    stats.attrs["prior"] = prior
    return p, stats


def otsu(p: np.ndarray, bins: int = 100) -> float:
    """확률 분포를 두 집단으로 나누는 임계값 (집단 내 분산 최소)."""
    h, edges = np.histogram(p, bins=bins, range=(0, 1))
    mid = (edges[:-1] + edges[1:]) / 2
    w = h / h.sum()
    best, t = -1, 0.5
    for k in range(1, bins):
        w0, w1 = w[:k].sum(), w[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (w[:k] * mid[:k]).sum() / w0, (w[k:] * mid[k:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, t = between, edges[k]
    return float(t)


def train_ws(X, p: np.ndarray, confident: float = 0.2, seed: int = 0, C: float = 1.0):
    """라벨 모델 확률 중 확실한 기사(|p-0.5| ≥ confident)로 로지스틱 회귀를 학습한다 (확신도 가중)."""
    from sklearn.linear_model import LogisticRegression

    sel = np.abs(p - 0.5) >= confident
    y = p[sel] >= 0.5
    w = np.abs(p[sel] - 0.5) * 2
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
    m.fit(X[sel], y, sample_weight=w)
    return m, int(sel.sum())


# ---------------------------------------------------------------- 평가

def evaluate(pred: dict[str, np.ndarray], anc: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """모델별 앵커 기반 대리 지표."""
    pos, neg = anc["anchor_content"].to_numpy(), anc["anchor_market"].to_numpy()
    only_pos, only_neg, both = pos & ~neg, neg & ~pos, pos & neg
    rows = []
    for name, r in pred.items():
        rows.append({"model": name, "relevant_rate": r.mean(),
                     "content_anchor_recall": r[only_pos].mean(),     # 콘텐츠 앵커를 관련으로 잡은 비율
                     "market_anchor_exclusion": 1 - r[only_neg].mean(),  # 시장 앵커를 무관으로 거른 비율
                     "boundary_relevant_rate": r[both].mean() if both.any() else np.nan})
    df = pd.DataFrame(rows)
    df["anchor_balanced_accuracy"] = (df["content_anchor_recall"] + df["market_anchor_exclusion"]) / 2
    return df


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def run_e2(cfg: dict, raw: pd.DataFrame, out_root: Path, llm_dir: Path | None = None, bootstrap: int = 5) -> dict:
    from .config import Lexicon
    from .layers import classify
    from .load import build_corpus
    from .relevance import features
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .text import extract_keywords, prepped_texts, space_joined_names, tokenize_corpus

    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e2_relevance", cfg, out_root, snap["data_snapshot_id"])

    with run.stage("prepare", rows_in=len(raw)) as st:
        corpus = build_corpus(raw, cfg)
        arts = corpus.articles
        names = space_joined_names(sorted(corpus.article_company["company"].unique()),
                                   cfg["cleaning"].get("company_aliases") or {})
        texts = prepped_texts(arts, names)
        tokens = tokenize_corpus(texts, names, out_root / "cache")
        kw = extract_keywords(texts, tokens, Lexicon.load(), cfg)
        lab, _ = classify(arts, kw, cfg)
        st["rows_out"] = len(arts)

    with run.stage("anchors") as st:
        links = raw.drop_duplicates(cfg["input"]["columns"]["article_id"]).set_index(cfg["input"]["columns"]["article_id"])
        sid = links["네이버링크"].astype(str).str.extract(r"[?&]sid1?=(\d{3})")[0] if "네이버링크" in links else pd.Series(dtype=str)
        anc, anc_info = anchors(arts, sid)
        st["rows_out"] = int(anc_info["anchor_content"] + anc_info["anchor_market"])

    with run.stage("weak_supervision") as st:
        L = labeling_functions(arts, kw, lab, cfg)
        p_lm, lf_stats = label_model(L)
        X, fnames = features(arts.assign(content_score=lab["content_score"].to_numpy(),
                                         market_score=lab["market_score"].to_numpy(),
                                         content_share=lab["content_share"].to_numpy()), kw)
        m, n_train = train_ws(X, p_lm, seed=cfg.get("reproducibility", {}).get("random_seed", 0) or 0)
        p_ws = m.predict_proba(X)[:, 1]
        t = otsu(p_ws)
        st["rows_out"] = n_train

    with run.stage("bootstrap_stability") as st:
        rng = np.random.default_rng(0)
        decisions = []
        for b in range(bootstrap):
            idx = rng.choice(len(arts), len(arts), replace=True)
            mb, _ = train_ws(X[idx], p_lm[idx], seed=b)
            decisions.append(mb.predict_proba(X)[:, 1] >= t)
        agree = np.mean([np.mean(d == (p_ws >= t)) for d in decisions])
        flip = np.mean(np.std(np.stack(decisions).astype(float), axis=0) > 0)
        st["rows_out"] = bootstrap

    with run.stage("evaluate") as st:
        pred = {
            "m_rel_seed (v1 층 분류, 기준선)": lab["content"].to_numpy(),
            "m_rel_labelmodel (라벨 모델 단독)": p_lm >= 0.5,
            f"m_rel_ws_lr (약한 지도 + 로지스틱, 임계 {t:.2f})": p_ws >= t,
        }
        if llm_dir and (llm_dir / "relevance_pred.csv").exists():
            pr = pd.read_csv(llm_dir / "relevance_pred.csv").set_index("gid")["p_relevant"]
            pred["m_rel_lr_llm (LLM 라벨 학습, 도전자)"] = arts["gid"].map(pr).fillna(0).to_numpy() >= 0.5
        ev = evaluate(pred, anc, arts["sectors"])
        names_ = list(pred)
        agreement = pd.DataFrame([[kappa(pred[a].astype(int), pred[b].astype(int)) for b in names_] for a in names_],
                                 index=names_, columns=names_)
        # 분야별 관련 비율 (분야 편차)
        sec = corpus.article_sector.merge(pd.DataFrame({"gid": arts["gid"], "ws": p_ws >= t}), on="gid")
        by_sector = sec.groupby("sector")["ws"].agg(["size", "mean"]).rename(columns={"size": "articles", "mean": "relevant_rate"})
        # 최종 판정: 약한 지도 모델 기준, 다른 독립 방법(기준선·라벨 모델)과 갈리면 REVIEW
        ws, base, lm = p_ws >= t, pred[names_[0]], p_lm >= 0.5
        decision = np.where(ws & base & lm, "INCLUDE", np.where(~ws & ~base & ~lm, "EXCLUDE", "REVIEW"))
        st["rows_out"] = len(decision)

    with run.stage("write") as st:
        top = np.argsort(-np.abs(m.coef_[0]))
        coef = pd.DataFrame({"feature": np.array(fnames)[top], "weight": m.coef_[0][top]}).head(200)
        out = pd.DataFrame({"gid": arts["gid"], "date": arts["date"].dt.date, "title": arts["title"],
                            "p_labelmodel": p_lm.round(4), "p_ws_lr": p_ws.round(4), "seed_layer": base,
                            "decision": decision, "anchor_content": anc["anchor_content"],
                            "anchor_market": anc["anchor_market"]})
        L_votes = L.apply(lambda r: ",".join(f"{k}:{'+' if v > 0 else '-'}" for k, v in r.items() if v != 0), axis=1)
        out["lf_votes"] = L_votes.to_numpy()
        pq = run.dir / "02_article_relevance.parquet"
        out.to_parquet(pq, index=False)
        run.artifact(pq, "article_relevance", rows=len(out))
        rep = run.dir / "e2_relevance_report.xlsx"
        with pd.ExcelWriter(rep) as xw:
            ev.to_excel(xw, sheet_name="모델 비교 (앵커 대리지표)", index=False)
            agreement.to_excel(xw, sheet_name="모델 간 일치 (kappa)")
            lf_stats.to_excel(xw, sheet_name="라벨 함수 (추정 정확도)", index=False)
            by_sector.reset_index().to_excel(xw, sheet_name="분야별 관련 비율", index=False)
            pd.Series(decision).value_counts().rename_axis("판정").reset_index(name="기사 수").to_excel(
                xw, sheet_name="판정 구간", index=False)
            coef.to_excel(xw, sheet_name="분류기 가중치", index=False)
            pd.DataFrame({"work_name": anc_info["top_work_names"]}).to_excel(xw, sheet_name="콘텐츠 앵커 상위", index=False)
        run.artifact(rep, "e2_report")
        st["rows_out"] = len(out)

    summary = {
        "anchors": {k: v for k, v in anc_info.items() if k != "top_work_names"},
        "label_model_prior": round(lf_stats.attrs["prior"], 3),
        "ws_train_confident": n_train, "otsu_threshold": round(t, 3),
        "bootstrap": {"runs": bootstrap, "mean_agreement": round(float(agree), 4), "share_ever_flipped": round(float(flip), 4)},
        "decision_counts": pd.Series(decision).value_counts().to_dict(),
        "models": ev.round(4).to_dict("records"),
    }
    (run.dir / "e2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_relevance_v2_e2", "run_id": run.run_id, "task": "article_relevance",
        "data_snapshot_id": snap["data_snapshot_id"], "models": list(pred), "evaluation": "anchor_proxy",
        "result": summary["models"], "otsu_threshold": summary["otsu_threshold"], "bootstrap": summary["bootstrap"]})
    run.finish()
    return summary
