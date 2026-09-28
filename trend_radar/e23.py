"""E2.3 3분류 라벨 모델 (2026-09-28 사용자 결정).

E2.2 진단: 이진 라벨 모델의 잠재 변수가 '콘텐츠 대 비콘텐츠'가 아니라 '시장 대 비시장'으로 잡혀 정치·사회 기사가
'관련' 쪽에 붙는다. 그래서 잠재 클래스를 셋으로 둔다.

| 클래스 | 의미 | 최종 관련성 |
|---|---|---|
| CONTENT | 콘텐츠 산업·작품·제작·유통·정책 | 관련 |
| MARKET | 주가·증권·실적·투자 중심 | 무관 |
| OTHER | 정치·사회·일반 사건·다른 업종·방송사 출처성 언급 | 무관 |

P(relevant) = P(CONTENT). MARKET과 OTHER를 학습 단계에서 합치지 않는다 (라벨 모델·분류기 모두 3클래스).

라벨 함수 투표값: CONTENT / MARKET / OTHER / NONCONTENT(시장·기타 중 어느 쪽인지 모르는 비콘텐츠 신호) / 기권.
3클래스 Dawid-Skene: 규칙마다 클래스별 투표 분포(혼동행렬)를 추정. 방향 제약으로 클래스 순열 뒤집힘을 막는다.
"""
from __future__ import annotations

import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["CONTENT", "MARKET", "OTHER"]
C, M, O = 0, 1, 2
AB, VC, VM, VO, VN = 0, 1, 2, 3, 4          # 투표 부호: 기권, CONTENT, MARKET, OTHER, NONCONTENT
VOTE_NAME = {VC: "CONTENT", VM: "MARKET", VO: "OTHER", VN: "NONCONTENT"}


# ---------------------------------------------------------------- 라벨 함수 (3분류)

def lfs3(arts: pd.DataFrame, kw: list[list[str]], lab: pd.DataFrame, cfg: dict, anc: pd.DataFrame) -> pd.DataFrame:
    """라벨 함수 계열별 투표 클래스 재배치. lab은 검색 기업명을 뺀 콘텐츠 점수(noco)."""
    from .e2 import OTHER_INDUSTRY, PRICE_TITLE, lf_broadcaster_source_only, lf_politics_society_section

    lc, r2 = cfg["layers"], cfg.get("e2", {})
    cseed, mseed = set(lc["content_seed"]), set(lc["market_seed"])
    sets = [set(w) for w in kw]
    c = np.array([len(s & cseed) for s in sets])
    m = np.array([len(s & mseed) for s in sets])
    o = np.array([len(s & OTHER_INDUSTRY) for s in sets])
    titles = arts["title"].fillna("").to_numpy()
    texts = [f"{t} {s_}" for t, s_ in zip(arts["title"].fillna(""), arts["summary"].fillna(""))]
    title_c = np.array([any(w in t for w in (s & cseed)) for s, t in zip(sets, titles)])
    price_title = np.array([bool(PRICE_TITLE.search(t)) for t in titles])
    strong = arts["strong"].to_numpy()
    related = (arts["tier"] == "related").to_numpy()
    share = lab["content_share"].to_numpy()
    cs, ms = lab["content_score"].to_numpy(), lab["market_score"].to_numpy()
    policy_words = set(r2.get("policy_anchor_words", []))
    policy = np.array([any(w in t for w in policy_words) for t in texts])
    ac = anc["anchor_content"].to_numpy().astype(bool)
    market_dom = (m >= 3) & (m > c)
    bro = {k: list(v) for k, v in (r2.get("broadcasters") or {}).items()}
    topic = set(r2.get("broadcast_topic_words", [])) | cseed
    neg_bro = lf_broadcaster_source_only(arts, bro, topic, ac)
    neg_sec = lf_politics_society_section(anc["sid"].to_numpy(), ac, texts, policy_words, set(r2.get("excluded_sections", ["100", "102"])))
    L = {
        # CONTENT: 작품명·장르·제작·출시·흥행 (시장 표현이 우세하면 기권)
        "LF_content_seed_2plus": np.where((c >= 2) & ~market_dom, VC, AB),
        "LF_content_seed_in_title": np.where(title_c & ~price_title, VC, AB),
        "LF_content_leaning_score": np.where((cs >= 3) & (share >= 0.7), VC, AB),
        # CONTENT: 콘텐츠 정책·지원사업 (정치·사회 섹션이어도 정책 대상이 콘텐츠면 유지)
        "LF_content_policy": np.where(policy & (c >= 1), VC, AB),
        # MARKET: 주가·목표가·특징주·증권 (콘텐츠 사건이 원인이어도 시장 중심이면 MARKET)
        "LF_market_seed_3plus_no_content": np.where((m >= 3) & (c == 0), VM, AB),
        "LF_market_leaning_score": np.where((ms >= 3) & (share <= 0.3), VM, AB),
        "LF_market_dominant": np.where(market_dom & (c >= 1), VM, AB),
        # NONCONTENT: 비콘텐츠지만 시장·기타 중 어느 쪽인지 모름
        "LF_passing_mention_no_content": np.where(~strong & (c < 2), VN, AB),
        "LF_related_tier_no_content": np.where(related & (c < 2), VN, AB),
        # OTHER: 다른 업종·정치·사회 섹션·방송사 출처성 (콘텐츠 근거가 있으면 기권)
        "LF_other_industry": np.where((o >= 2) & (c == 0), VO, AB),
        "LF_politics_society_section": np.where(neg_sec != 0, VO, AB),
        "LF_broadcaster_source_only": np.where(neg_bro != 0, VO, AB),
    }
    return pd.DataFrame(L)


def lf_targets(L: pd.DataFrame) -> dict[str, int]:
    """규칙마다 유일한 투표 부호 (방향 제약용)."""
    out = {}
    for j in L.columns:
        v = L[j].to_numpy()
        vals = set(np.unique(v[v != AB]).tolist())
        assert len(vals) <= 1, f"{j}는 한 방향으로만 투표해야 한다"
        out[j] = vals.pop() if vals else AB
    return out


# ---------------------------------------------------------------- 3클래스 Dawid-Skene

def label_model3(L: pd.DataFrame, iters: int = 200, smooth: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """반환: 사후확률 q (n × 3), θ (J × 3 × 5, P(투표|클래스)), 사전확률 π, 규칙별 투표 부호.
    제약: CONTENT·MARKET·OTHER 규칙은 자기 클래스에서 그 투표 확률이 가장 크다. NONCONTENT 규칙은 CONTENT에서 가장 작다.
    초기값: 규칙 계열별 방향 (CONTENT→C, MARKET→M, OTHER→O, NONCONTENT→M·O 반반)."""
    V = L.to_numpy()
    n, J = V.shape
    tgt = lf_targets(L)
    tv = np.array([tgt[j] for j in L.columns])
    onehot = np.stack([(V == a) for a in range(5)], axis=2).astype(float)       # n × J × 5
    init = np.zeros((n, 3)) + 0.2
    init[:, C] += (V == VC).sum(1)
    init[:, M] += (V == VM).sum(1) + 0.5 * (V == VN).sum(1)
    init[:, O] += (V == VO).sum(1) + 0.5 * (V == VN).sum(1)
    q = init / init.sum(1, keepdims=True)
    cls_of = {VC: C, VM: M, VO: O}
    for _ in range(iters):
        th = np.einsum("nk,nja->jka", q, onehot) + smooth
        th /= th.sum(2, keepdims=True)
        for j in range(J):
            v = tv[j]
            if v in cls_of:
                t = cls_of[v]
                k = int(np.argmax(th[j, :, v]))
                if k != t:
                    th[j, t, v], th[j, k, v] = th[j, k, v], th[j, t, v]
            elif v == VN:
                k = int(np.argmin(th[j, :, v]))
                if k != C:
                    th[j, C, v], th[j, k, v] = th[j, k, v], th[j, C, v]
            th[j] /= th[j].sum(1, keepdims=True)
        pi = np.clip(q.mean(0), 0.01, None)
        pi /= pi.sum()
        lq = np.log(pi)[None] + np.einsum("nja,jka->nk", onehot, np.log(th))
        lq -= lq.max(1, keepdims=True)
        q_new = np.exp(lq)
        q_new /= q_new.sum(1, keepdims=True)
        if np.max(np.abs(q_new - q)) < 1e-6:
            q = q_new
            break
        q = q_new
    return q, th, pi, tgt


def lf_stats3(L: pd.DataFrame, th: np.ndarray, pi: np.ndarray, tgt: dict[str, int]) -> pd.DataFrame:
    V = L.to_numpy()
    X = np.stack([(V[:, j] != AB).astype(float) for j in range(V.shape[1])], 1)
    corr = np.corrcoef(X.T) if X.shape[1] > 1 else np.eye(1)
    rows = []
    for j, name in enumerate(L.columns):
        v = tgt[name]
        post = pi * th[j, :, v]
        post = post / post.sum()
        prec = 1 - post[C] if v == VN else post[{VC: C, VM: M, VO: O}[v]]
        same = [k for k, nm in enumerate(L.columns) if k != j and tgt[nm] == v]
        max_corr = float(np.nanmax(corr[j, same])) if same else 0.0
        rows.append({"lf": name, "vote": VOTE_NAME.get(v, "-"), "coverage": round(float((V[:, j] != AB).mean()), 4),
                     **{f"P(vote|{c})": round(float(th[j, k, v]), 5) for k, c in enumerate(CLASSES)},
                     "estimated_precision": round(float(prec), 4), "max_corr_same_direction": round(max_corr, 3),
                     "independence_violation_suspected": bool(prec >= 0.99 and max_corr >= 0.3)})
    return pd.DataFrame(rows)


def lf_agreement(L: pd.DataFrame) -> pd.DataFrame:
    """두 규칙이 함께 투표한 기사 중 양립하는 비율 (같은 클래스, 또는 NONCONTENT와 MARKET·OTHER)."""
    V = L.to_numpy()
    ok = {(a, b) for a in (VC, VM, VO, VN) for b in (VC, VM, VO, VN)
          if a == b or {a, b} in ({VN, VM}, {VN, VO})}
    cols = list(L.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, j in itertools.combinations(range(len(cols)), 2):
        both = (V[:, i] != AB) & (V[:, j] != AB)
        if both.sum() >= 20:
            r = np.mean([(a, b) in ok for a, b in zip(V[both, i], V[both, j])])
            out.iat[i, j] = out.iat[j, i] = round(float(r), 3)
    return out


def ablation3(L: pd.DataFrame, q_full: np.ndarray, probe: np.ndarray) -> pd.DataFrame:
    rows = []
    _, _, pi_full, _ = label_model3(L)
    for j in L.columns:
        q, _, pi, _ = label_model3(L.drop(columns=[j]))
        d = q[:, C] - q_full[:, C]
        rows.append({"removed_lf": j, "mean_abs_change_P(CONTENT)": round(float(np.abs(d).mean()), 4),
                     "probe_mean_change_P(CONTENT)": round(float(d[probe].mean()), 4) if probe.any() else None,
                     "content_decision_flip_rate": round(float(((q[:, C] >= 0.5) != (q_full[:, C] >= 0.5)).mean()), 4),
                     **{f"prior_{c}": round(float(pi[k]), 4) for k, c in enumerate(CLASSES)},
                     **{f"prior_change_{c}": round(float(pi[k] - pi_full[k]), 4) for k, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows).sort_values("mean_abs_change_P(CONTENT)", ascending=False)


# ---------------------------------------------------------------- 3클래스 분류기

def soft_rows3(q: np.ndarray, covered: np.ndarray, confidence: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """기사 하나를 클래스별로 복제: y=k, 가중 = P(k) (× 정규화 엔트로피 신뢰도 1 - H(q)/log 3)."""
    idx = np.where(covered)[0]
    qq = np.clip(q[idx], 1e-9, 1)
    conf = 1 - (-(qq * np.log(qq)).sum(1) / np.log(3)) if confidence else np.ones(len(idx))
    rows = np.concatenate([idx] * 3)
    y = np.concatenate([np.full(len(idx), k) for k in range(3)])
    w = np.concatenate([qq[:, k] * conf for k in range(3)])
    keep = w > 1e-9
    return rows[keep], y[keep], w[keep]


def train3(X, q: np.ndarray, covered: np.ndarray, confidence: bool, seed: int = 0, C_: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    rows, y, w = soft_rows3(q, covered, confidence)
    m = LogisticRegression(C=C_, max_iter=2000, random_state=seed)
    m.fit(X[rows], y, sample_weight=w)
    return m


# ---------------------------------------------------------------- 평가·선정

def gates3(df: pd.DataFrame, base_flip: float, rules: dict) -> pd.DataFrame:
    g = rules.get("e23_gates", {})
    d = df.copy()
    d["gate_probe"] = d["probe_in_rate_drop_pp"] >= g.get("probe_drop_pp", 5.0)
    d["gate_anchor"] = d["anchor_in_rate_drop_pp"] <= g.get("anchor_drop_pp", 2.0)
    d["gate_policy"] = d["policy_to_exclude_rate"] < g.get("policy_exclude_max", 0.01)
    d["gate_broadcast"] = d["broadcast_excess_drop_pp"] <= g.get("broadcast_excess_pp", 2.0)
    d["gate_flip"] = d["direction_flip_rate"] < base_flip
    gc = [c for c in d.columns if c.startswith("gate_")]
    d["gates_passed"] = d[gc].sum(axis=1)
    d["pass_all"] = d[gc].all(axis=1)
    d["include_change_rel"] = d["include_change_rel"].astype(float)
    d["auto_confirm_blocked"] = d["include_change_rel"].abs() >= g.get("include_change_max", 0.30)
    return d


def set_candidate(cfg_path: Path, model_id: str | None, run_id: str | None, status: str) -> None:
    """trend_radar.yaml의 e2 후보 포인터만 바꾼다 (selected_*는 E4까지 통과한 뒤 사람이 결정)."""
    s = cfg_path.read_text(encoding="utf-8")
    s = re.sub(r"(?m)^(  candidate_model_id: ).*$", rf"\g<1>{model_id or 'null'}", s)
    s = re.sub(r"(?m)^(  candidate_run_id: ).*$", rf"\g<1>{run_id or 'null'}", s)
    s = re.sub(r"(?m)^(  selection_status: ).*$", rf"\g<1>{status}", s)
    cfg_path.write_text(s, encoding="utf-8")


def run_e23(cfg: dict, raw: pd.DataFrame, out_root: Path, bootstrap: int = 3) -> dict:
    from .config import Lexicon
    from .e2 import anchors, change_report, evaluate, otsu
    from .e22 import drop_search_companies, separate_source_tokens
    from .layers import classify
    from .load import build_corpus
    from .relevance import features
    from .runlog import REGISTRY, ROOT, Run, append_jsonl, snapshot
    from .selection import mark_failed, selected_e2
    from .text import extract_keywords, prepped_texts, space_joined_names, tokenize_corpus

    r2 = cfg.get("e2", {})
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e23_relevance_3class", cfg, out_root, snap["data_snapshot_id"])
    base_run, base_path, _ = selected_e2(cfg, REGISTRY, out_root, use_candidate=False)   # E2.1 운영 (기준선)
    prev = pd.read_parquet(base_path)
    seed0 = cfg.get("reproducibility", {}).get("random_seed", 0) or 0

    with run.stage("prepare", rows_in=len(raw)) as st:
        corpus = build_corpus(raw, cfg)
        arts = corpus.articles
        aliases = cfg["cleaning"].get("company_aliases") or {}
        names = space_joined_names(sorted(corpus.article_company["company"].unique()), aliases)
        texts = prepped_texts(arts, names)
        tokens = tokenize_corpus(texts, names, out_root / "cache")
        kw = extract_keywords(texts, tokens, Lexicon.load(), cfg)
        links = raw.drop_duplicates(cfg["input"]["columns"]["article_id"]).set_index(cfg["input"]["columns"]["article_id"])
        sid = links["네이버링크"].astype(str).str.extract(r"[?&]sid1?=(\d{3})")[0] if "네이버링크" in links else pd.Series(dtype=str)
        anc, anc_info = anchors(arts, sid)
        raw_texts = [f"{t} {s}" for t, s in zip(arts["title"], arts["summary"])]
        probe = np.array([any(w in t for w in r2.get("politics_probe_words", [])) for t in raw_texts]) | \
            np.isin(anc["sid"].to_numpy(), r2.get("excluded_sections", ["100", "102"]))
        kw_noco, dropped = drop_search_companies(kw, arts, aliases, set(r2.get("industry_action_words", [])))
        kw_src = separate_source_tokens(kw, arts, {k: list(v) for k, v in (r2.get("broadcasters") or {}).items()})
        lab, _ = classify(arts, kw_noco, cfg)
        st["rows_out"] = len(arts)

    with run.stage("label_model") as st:
        L = lfs3(arts, kw, lab, cfg, anc)
        q, th, pi, tgt = label_model3(L)
        stats = lf_stats3(L, th, pi, tgt)
        agree = lf_agreement(L)
        abl = ablation3(L, q, probe)
        covered = (L.to_numpy() != AB).any(1)
        L_wo = L.drop(columns=["LF_content_leaning_score"])
        q_wo, _, pi_wo, _ = label_model3(L_wo)
        st["rows_out"] = int(covered.sum())

    with run.stage("classifiers") as st:
        Xf, _ = features(arts.assign(content_score=lab["content_score"].to_numpy(), market_score=lab["market_score"].to_numpy(),
                                     content_share=lab["content_share"].to_numpy()), kw_src)
        clfs = {"LM3+softA": train3(Xf, q, covered, False, seed0), "LM3+confidence": train3(Xf, q, covered, True, seed0)}
        probs = {k: m.predict_proba(Xf) for k, m in clfs.items()}
        st["rows_out"] = len(clfs)

    # 비교: E2.1 이진(기준선) / LM3 단독 / LM3+softA / LM3+신뢰도 가중 (+ 진단: LM3에서 콘텐츠 점수 규칙 제거)
    seed_l = lab["content"].to_numpy()
    lm_c = q.argmax(1) == C
    base_cov = (prev["lf_votes"].fillna("") != "").to_numpy() if "lf_votes" in prev else covered
    pt = otsu(prev["p_ws_lr"].to_numpy())
    base_flip = float(((prev["p_ws_lr"].to_numpy() >= pt) != (prev["p_labelmodel"].to_numpy() >= 0.5))[base_cov].mean())
    models = {"E2.1_binary(기준선)": None, "LM3_only": q, "LM3+softA": probs["LM3+softA"], "LM3+confidence": probs["LM3+confidence"],
              "diag:LM3_without_content_leaning": q_wo}
    rows, outs = [], {}
    for name, P in models.items():
        if P is None:
            out = prev.copy()
            score, t = prev["p_ws_lr"].to_numpy(), pt
            flip = base_flip
            argmax = None
        else:
            score = P[:, C]
            t = otsu(score)
            lm_vote = (q_wo.argmax(1) == C) if name.startswith("diag") else lm_c
            ws = score >= t
            decision = np.where(ws & seed_l & lm_vote, "INCLUDE", np.where(~ws & ~seed_l & ~lm_vote, "EXCLUDE", "REVIEW"))
            out = pd.DataFrame({"gid": arts["gid"], "date": arts["date"].dt.date, "title": arts["title"],
                                "p_labelmodel": q[:, C].round(4), "p_ws_lr": score.round(4), "seed_layer": seed_l, "decision": decision,
                                "anchor_content": anc["anchor_content"], "anchor_market": anc["anchor_market"],
                                **{f"p_{c.lower()}": P[:, k].round(4) for k, c in enumerate(CLASSES)}})
            flip = float(((score >= t) != (q[:, C] >= 0.5))[covered].mean())
            argmax = P.argmax(1)
        ch, _ = change_report(prev, out, arts, corpus.article_sector, anc, raw_texts, r2)
        sec = ch["sector_change_rate"]
        others = [v for k, v in sec.items() if k != "방송 및 영상"]
        dec = out["decision"].to_numpy()
        ev = evaluate({"m": out["p_ws_lr"].to_numpy() >= t}, anc, arts["sectors"]).iloc[0]
        row = {"model": name, "otsu": round(float(t), 3),
               "include": int((dec == "INCLUDE").sum()), "review": int((dec == "REVIEW").sum()), "exclude": int((dec == "EXCLUDE").sum()),
               "include_change_rel": round((dec == "INCLUDE").sum() / max((prev["decision"] == "INCLUDE").sum(), 1) - 1, 4),
               "probe_in_rate": ch["politics_probe"]["in_rate_new"],
               "probe_in_rate_drop_pp": round(100 * (ch["politics_probe"]["in_rate_prev"] - ch["politics_probe"]["in_rate_new"]), 2),
               "anchor_in_rate": ch["content_anchor"]["in_rate_new"],
               "anchor_in_rate_drop_pp": round(100 * (ch["content_anchor"]["in_rate_prev"] - ch["content_anchor"]["in_rate_new"]), 2),
               "policy_to_exclude_rate": ch["policy_articles"]["rate"],
               "broadcast_change_rate": sec.get("방송 및 영상"),
               "broadcast_excess_drop_pp": round(100 * (float(np.mean(others)) - sec.get("방송 및 영상", 0)), 2),
               "direction_flip_rate": round(flip, 4),
               "probe_direction_flip_rate": round(float(((score >= t) != ((prev["p_labelmodel"].to_numpy() if P is None else q[:, C]) >= 0.5))
                                                        [(base_cov if P is None else covered) & probe].mean()), 4),
               "content_anchor_recall": round(float(ev["content_anchor_recall"]), 4),
               "market_anchor_exclusion": round(float(ev["market_anchor_exclusion"]), 4),
               "anchor_balanced_accuracy": round(float(ev["anchor_balanced_accuracy"]), 4)}
        if argmax is not None:
            ac = anc["anchor_content"].to_numpy().astype(bool)
            am = anc["anchor_market"].to_numpy().astype(bool)
            row.update({"content_anchor_pred_noncontent": round(float((argmax[ac] != C).mean()), 4),
                        "market_anchor_pred_content": round(float((argmax[am] == C).mean()), 4),
                        "probe_pred_content": round(float((argmax[probe] == C).mean()), 4),
                        **{f"share_{c}": round(float((argmax == k).mean()), 4) for k, c in enumerate(CLASSES)}})
        rows.append(row)
        outs[name] = (out, t, P)
    cmp_ = gates3(pd.DataFrame(rows), base_flip, cfg)
    cmp_.loc[cmp_["model"].str.startswith(("E2.1", "diag")), [c for c in cmp_.columns if c.startswith("gate_")] + ["pass_all"]] = False
    cands = cmp_[cmp_["pass_all"] & ~cmp_["model"].str.startswith(("E2.1", "diag"))] \
        .sort_values(["probe_in_rate_drop_pp", "direction_flip_rate"], ascending=[False, True])

    with run.stage("write") as st:
        best = cands.iloc[0]["model"] if len(cands) else None
        show = best or cmp_[~cmp_["model"].str.startswith(("E2.1", "diag"))].sort_values(["gates_passed", "probe_in_rate_drop_pp"], ascending=False).iloc[0]["model"]
        out, t, P = outs[show]
        out = out.copy()
        out["lf_votes"] = L.apply(lambda r: ",".join(f"{k}:{VOTE_NAME[v]}" for k, v in r.items() if v != AB), axis=1).to_numpy()
        out["e23_model"] = show
        # 클래스별 대표·경계 사례 30건씩
        cases = []
        srt = np.sort(P, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        am_ = P.argmax(1)
        for k, c in enumerate(CLASSES):
            idx = np.where(am_ == k)[0]
            for kind, sel in (("representative", idx[np.argsort(-P[idx, k])[:30]]), ("boundary", idx[np.argsort(margin[idx])[:30]])):
                for i in sel:
                    cases.append({"class": c, "kind": kind, "gid": arts["gid"].iat[i], "title": arts["title"].iat[i],
                                  **{f"p_{cc.lower()}": round(float(P[i, kk]), 4) for kk, cc in enumerate(CLASSES)},
                                  "lf_votes": out["lf_votes"].iat[i], "decision": out["decision"].iat[i]})
        cases = pd.DataFrame(cases)
        conf3 = pd.crosstab(pd.Series(np.array(CLASSES)[q.argmax(1)], name="label_model"),
                            pd.Series(np.array(CLASSES)[P.argmax(1)], name=show))
        with pd.ExcelWriter(run.dir / "e23_report.xlsx") as xw:
            cmp_.to_excel(xw, sheet_name="models_gates", index=False)
            stats.to_excel(xw, sheet_name="lf_stats_3class", index=False)
            agree.to_excel(xw, sheet_name="lf_agreement")
            abl.to_excel(xw, sheet_name="lf_ablation", index=False)
            conf3.to_excel(xw, sheet_name="lm_vs_classifier")
            cases.to_excel(xw, sheet_name="class_cases", index=False)
        run.artifact(run.dir / "e23_report.xlsx", "e23_report")
        with open(run.dir / "e23_comparison_pack.jsonl", "w", encoding="utf-8") as f:
            for r in cmp_.to_dict("records"):
                f.write(json.dumps({"type": "model", **r}, ensure_ascii=False, default=str) + "\n")
            for r in cases.to_dict("records"):
                f.write(json.dumps({"type": "case", **r}, ensure_ascii=False, default=str) + "\n")
        if best:
            pq = run.dir / "02_article_relevance.parquet"
            out.to_parquet(pq, index=False)
            run.artifact(pq, "article_relevance_candidate", rows=len(out))
        st["rows_out"] = len(cases)

    prior = dict(zip(CLASSES, np.round(pi, 4).tolist()))
    summary = {
        "baseline_run_id": base_run, "baseline_flip_rate": round(base_flip, 4),
        "candidate_model": best, "otsu_threshold": round(float(t), 3),
        "auto_confirm_blocked": bool(cmp_.set_index("model").loc[best, "auto_confirm_blocked"]) if best else None,
        "class_prior": prior, "class_prior_without_content_leaning": dict(zip(CLASSES, np.round(pi_wo, 4).tolist())),
        "covered": round(float(covered.mean()), 4), "search_company_dropped_articles": int(dropped.sum()),
        "models": cmp_.to_dict("records"),
        "lf_stats": stats.to_dict("records"),
        "independence_violation_suspected": stats.loc[stats["independence_violation_suspected"], "lf"].tolist(),
        "decision_counts": out["decision"].value_counts().to_dict(),
    }
    (run.dir / "e2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if best and not summary["auto_confirm_blocked"]:
        set_candidate(ROOT / "trend_radar.yaml", f"e2_3_{best.replace('+', '_').lower()}", run.run_id, "candidate")
        status = "candidate"
    elif best:
        status = "pending_review (INCLUDE 변화 30% 이상)"
    else:
        mark_failed(REGISTRY, run.run_id, "e2_3", list(models), "E2 내부 필수 기준을 통과한 모델 없음")
        status = "failed"
    summary["status"] = status
    (run.dir / "e2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_relevance_v2_e23", "run_id": run.run_id, "task": "article_relevance_3class",
        "data_snapshot_id": snap["data_snapshot_id"], "baseline_run_id": base_run, "models": list(models),
        "candidate_model": best, "status": status, "otsu_threshold": summary["otsu_threshold"],
        "result": cmp_[["model", "probe_in_rate_drop_pp", "anchor_in_rate_drop_pp", "policy_to_exclude_rate",
                        "broadcast_excess_drop_pp", "direction_flip_rate", "pass_all"]].to_dict("records")})
    run.finish()
    return summary
