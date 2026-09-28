"""E2.2 관련성 학습 방식 비교 (2026-09-28 사용자 결정). 선정 모델 하나만 E3 → E5a → E4.2에 연결한다.

비교 축
1. 최종 분류기 학습
   - base: 라벨 모델 확신 기사(p ≤ 0.2, p ≥ 0.8)만 (기존)
   - softA: 라벨 함수가 하나 이상 투표한 기사 전체를 확률 라벨로 (기사를 y=1·w=p, y=0·w=1-p 두 행으로 복제)
   - softB: softA × 확신도 가중 (1 - binary_entropy(p)) → p≈0.5 대량 기사가 학습을 지배하지 않음
2. 검색 기업명 처리
   - orig: 기존
   - noco: 콘텐츠 점수(층 분류 → 라벨 함수·수치 특징)에서 검색 기업명 제외. 단, 제목·요약에 산업 행위어
     (제작·편성·유통·공연·실적·규제 …)가 함께 있으면 기업이 기사 주제 → 유지
   - noco_src: noco + 분류기 입력에서 출처형 방송사 표기·제목 말머리 단어를 제거하고 SRC_BROADCASTER 토큰으로 분리

진단: 라벨 함수 투표 상관행렬, 규칙 하나씩 뺀 라벨 모델의 기사별 확률 변화, 라벨 모델과 분류기 확률의
평균 절대차·방향 불일치율(전체·탐침).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .e2 import (ABSTAIN, anchors, change_report, evaluate, label_model, labeling_functions, otsu,
                 strip_source_mentions)


# ---------------------------------------------------------------- 학습

def soft_rows(p: np.ndarray, covered: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """확률 라벨 → (행 번호, y, 가중치). softA: w=p / 1-p, softB: 여기에 1 - H(p) (H: 이진 엔트로피, 밑 2)."""
    idx = np.where(covered)[0]
    pp = np.clip(p[idx], 1e-6, 1 - 1e-6)
    conf = 1.0
    if mode == "softB":
        h = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp))
        conf = 1 - h
    rows = np.r_[idx, idx]
    y = np.r_[np.ones(len(idx), int), np.zeros(len(idx), int)]
    w = np.r_[pp * conf, (1 - pp) * conf]
    keep = w > 1e-9
    return rows[keep], y[keep], w[keep]


def train_variant(X, p_lm: np.ndarray, covered: np.ndarray, mode: str, seed: int = 0, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    if mode == "base":
        from .e2 import train_ws
        return train_ws(X, p_lm, seed=seed, C=C)
    rows, y, w = soft_rows(p_lm, covered, mode)
    m = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    m.fit(X[rows], y, sample_weight=w)
    return m, int(covered.sum())


# ---------------------------------------------------------------- 검색 기업명 처리

def company_tokens(companies: list[str], aliases: dict[str, list[str]]) -> set[str]:
    from .text import token_form

    out = set()
    for c in companies:
        out.add(token_form(c))
        for a in aliases.get(c, []) or []:
            out.add(token_form(a))
    return out


def drop_search_companies(kw: list[list[str]], arts: pd.DataFrame, aliases: dict, action_words: set[str]) -> tuple[list[list[str]], np.ndarray]:
    """콘텐츠 점수용 키워드에서 검색 기업명 제거. 제목·요약에 산업 행위어가 있으면 기업이 기사 주제 → 유지.
    반환: 키워드, 제거 여부."""
    out, dropped = [], np.zeros(len(kw), bool)
    for i, (ws, t, s, cos) in enumerate(zip(kw, arts["title"], arts["summary"], arts["companies"])):
        text = f"{t} {s}"
        if any(a in text for a in action_words):
            out.append(ws)
            continue
        # 기업이 실제 문장 주체 ('SBS가 …', 'YTN은 …', '하이브, …') → 기사 내용 근거로 유지
        names = [c for c in cos] + [a for c in cos for a in (aliases.get(c) or [])]
        if any(re.search(rf"(?:^|[\s\"'‘“(]){re.escape(n)}(?:은|는|이|가|,)(?=\s)", text) for n in names):
            out.append(ws)
            continue
        ct = company_tokens(cos, aliases)
        new = [w for w in ws if w not in ct]
        dropped[i] = len(new) < len(ws)
        out.append(new)
    return out, dropped


def separate_source_tokens(kw: list[list[str]], arts: pd.DataFrame, broadcasters: dict[str, list[str]]) -> list[list[str]]:
    """분류기 입력: 제목 말머리([…]) 안에서만 나온 단어 제거, 출처 자리에만 나온 방송사 표기 → SRC_BROADCASTER."""
    from .text import token_form

    out = []
    for ws, t, s, cos in zip(kw, arts["title"], arts["summary"], arts["companies"]):
        br = " ".join(re.findall(r"\[([^\]]{0,30})\]", t or ""))
        outside = re.sub(r"\[[^\]]{0,30}\]", " ", t or "") + " " + (s or "")
        new = [w for w in ws if not (w in br and w not in outside)]
        hit = [c for c in cos if c in broadcasters]
        if hit:
            names = sorted({x for c in hit for x in broadcasters[c]}, key=len, reverse=True)
            if not any(n in strip_source_mentions(f"{t} {s}", names) for n in names):
                toks = {token_form(n) for n in names} | set(names)
                new = [w for w in new if w not in toks] + ["SRC_BROADCASTER"]
        out.append(new)
    return out


# ---------------------------------------------------------------- 진단

def lf_correlation(L: pd.DataFrame) -> pd.DataFrame:
    X = L.to_numpy().astype(float)
    sd = X.std(0)
    c = np.corrcoef(X[:, sd > 0].T) if (sd > 0).sum() > 1 else np.eye(1)
    cols = L.columns[sd > 0]
    return pd.DataFrame(c, index=cols, columns=cols).round(3)


def lf_ablation(L: pd.DataFrame, p_full: np.ndarray, probe: np.ndarray) -> pd.DataFrame:
    rows = []
    for j in L.columns:
        p_wo, st = label_model(L.drop(columns=[j]))
        d = p_wo - p_full
        rows.append({"removed_lf": j, "mean_abs_change": round(float(np.abs(d).mean()), 4),
                     "mean_change": round(float(d.mean()), 4),
                     "probe_mean_change": round(float(d[probe].mean()), 4) if probe.any() else None,
                     "decision_flip_rate": round(float(((p_wo >= 0.5) != (p_full >= 0.5)).mean()), 4),
                     "prior_without": round(st.attrs["prior"], 4)})
    return pd.DataFrame(rows).sort_values("mean_abs_change", ascending=False)


def agreement_stats(p_clf: np.ndarray, t: float, p_lm: np.ndarray, covered: np.ndarray, probe: np.ndarray) -> dict:
    flip = (p_clf >= t) != (p_lm >= 0.5)
    return {"mean_abs_clf_minus_lm": round(float(np.abs(p_clf - p_lm)[covered].mean()), 4),
            "direction_flip_rate": round(float(flip[covered].mean()), 4),
            "probe_direction_flip_rate": round(float(flip[covered & probe].mean()), 4) if (covered & probe).any() else None,
            "lm_irrelevant_but_clf_relevant": round(float(((p_lm < 0.5) & (p_clf >= t))[covered].mean()), 4)}


def select(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """사용자 선정 기준 (E2.1 대비). 통과 기준을 모두 넘은 모델 중 탐침 감소가 크고 방향 불일치가 작은 모델."""
    g = rules.get("e22_gates", {})
    d = df.copy()
    d["gate_probe"] = d["probe_in_rate_drop_pp"] >= g.get("probe_drop_pp", 5.0)
    d["gate_anchor"] = d["anchor_in_rate_drop_pp"] <= g.get("anchor_drop_pp", 2.0)
    d["gate_broadcast"] = d["broadcast_excess_drop_pp"] <= g.get("broadcast_excess_pp", 3.0)
    d["gate_policy"] = d["policy_to_exclude_rate"] < g.get("policy_exclude_max", 0.01)
    d["gate_flip"] = d["direction_flip_rate"] < d["base_flip_rate"]
    d["relevant_rate_shift_flag"] = d["relevant_rate_change_rel"].abs() > g.get("relevant_rate_shift", 0.10)
    gc = [c for c in d.columns if c.startswith("gate_")]
    d["gates_passed"] = d[gc].sum(axis=1)
    d["pass_all"] = d[gc].all(axis=1)
    return d.sort_values(["pass_all", "gates_passed", "probe_in_rate_drop_pp", "direction_flip_rate"],
                         ascending=[False, False, False, True])


# ---------------------------------------------------------------- 실행

def run_e22(cfg: dict, raw: pd.DataFrame, out_root: Path, bootstrap: int = 5) -> dict:
    from .config import Lexicon
    from .layers import classify
    from .load import build_corpus
    from .relevance import features
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .text import extract_keywords, prepped_texts, space_joined_names, tokenize_corpus

    r2 = cfg.get("e2", {})
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e22_relevance_variants", cfg, out_root, snap["data_snapshot_id"])
    recs = [json.loads(l) for l in (REGISTRY / "experiment_registry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    prev_id = [r for r in recs if r.get("task") == "article_relevance"][-1]["run_id"]
    prev = pd.read_parquet(out_root / "runs" / prev_id / "02_article_relevance.parquet")

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
        probe_w = r2.get("politics_probe_words", [])
        probe = np.array([any(w in t for w in probe_w) for t in raw_texts]) | np.isin(anc["sid"].to_numpy(), r2.get("excluded_sections", ["100", "102"]))
        st["rows_out"] = len(arts)

    kw_noco, dropped = drop_search_companies(kw, arts, aliases, set(r2.get("industry_action_words", [])))
    bro = {k: list(v) for k, v in (r2.get("broadcasters") or {}).items()}
    kw_src = separate_source_tokens(kw, arts, bro)
    variants = {"orig": (kw, kw), "noco": (kw_noco, kw), "noco_src": (kw_noco, kw_src)}

    rows, keep, diag = [], {}, {}
    seed0 = cfg.get("reproducibility", {}).get("random_seed", 0) or 0
    base_flip = None
    with run.stage("variants") as st:
        for vname, (kw_score, kw_feat) in variants.items():
            lab, _ = classify(arts, kw_score, cfg)
            L = labeling_functions(arts, kw, lab, cfg, anc)
            p_lm, lf_stats = label_model(L)
            covered = (L.to_numpy() != ABSTAIN).any(1)
            Xf, fnames = features(arts.assign(content_score=lab["content_score"].to_numpy(), market_score=lab["market_score"].to_numpy(),
                                              content_share=lab["content_share"].to_numpy()), kw_feat)
            diag[vname] = {"lf_stats": lf_stats, "L": L, "p_lm": p_lm}
            for mode in ("base", "softA", "softB"):
                m, n_train = train_variant(Xf, p_lm, covered, mode, seed=seed0)
                p = m.predict_proba(Xf)[:, 1]
                t = otsu(p)
                ws, base_l, lm = p >= t, lab["content"].to_numpy(), p_lm >= 0.5
                decision = np.where(ws & base_l & lm, "INCLUDE", np.where(~ws & ~base_l & ~lm, "EXCLUDE", "REVIEW"))
                out = pd.DataFrame({"gid": arts["gid"], "date": arts["date"].dt.date, "title": arts["title"],
                                    "p_labelmodel": p_lm.round(4), "p_ws_lr": p.round(4), "seed_layer": base_l, "decision": decision,
                                    "anchor_content": anc["anchor_content"], "anchor_market": anc["anchor_market"]})
                ch, _ = change_report(prev, out, arts, corpus.article_sector, anc, raw_texts, r2)
                ag = agreement_stats(p, t, p_lm, covered, probe)
                ev = evaluate({"m": ws}, anc, arts["sectors"]).iloc[0].to_dict()
                sec = ch["sector_change_rate"]
                others = [v for k, v in sec.items() if k not in ("방송 및 영상",)]
                name = f"{vname}|{mode}"
                if name == "orig|base":
                    base_flip = ag["direction_flip_rate"]
                rows.append({
                    "variant": name, "company_handling": vname, "training": mode, "train_rows": n_train, "otsu": round(t, 3),
                    "relevant_rate": round(float(ws.mean()), 4),
                    "include": int((decision == "INCLUDE").sum()), "review": int((decision == "REVIEW").sum()),
                    "exclude": int((decision == "EXCLUDE").sum()),
                    "probe_in_rate": ch["politics_probe"]["in_rate_new"],
                    "probe_in_rate_drop_pp": round(100 * (ch["politics_probe"]["in_rate_prev"] - ch["politics_probe"]["in_rate_new"]), 2),
                    "probe_mcnemar_p": ch["politics_probe"]["mcnemar_p"],
                    "anchor_in_rate": ch["content_anchor"]["in_rate_new"],
                    "anchor_in_rate_drop_pp": round(100 * (ch["content_anchor"]["in_rate_prev"] - ch["content_anchor"]["in_rate_new"]), 2),
                    "anchor_include_rate": ch["content_anchor"]["include_rate_new"],
                    "broadcast_change_rate": sec.get("방송 및 영상"),
                    "broadcast_excess_drop_pp": round(100 * (float(np.mean(others)) - sec.get("방송 및 영상", 0)), 2),
                    "policy_to_exclude_rate": ch["policy_articles"]["rate"],
                    "relevant_rate_change_rel": round(ch["all_in_new"] / max(ch["all_in_prev"], 1) - 1, 4),
                    **ag, "anchor_balanced_accuracy": round(ev["anchor_balanced_accuracy"], 4),
                    "content_anchor_recall": round(ev["content_anchor_recall"], 4),
                    "market_anchor_exclusion": round(ev["market_anchor_exclusion"], 4)})
                keep[name] = (out, L, p_lm, lf_stats, m, Xf, t, fnames, lab)
        st["rows_out"] = len(rows)

    with run.stage("diagnostics") as st:
        vt = pd.DataFrame(rows)
        vt["base_flip_rate"] = base_flip
        sel = select(vt, cfg)
        best = sel.iloc[0]["variant"]
        corr = lf_correlation(diag["orig"]["L"])
        abl = lf_ablation(diag["orig"]["L"], diag["orig"]["p_lm"], probe)
        abl_noco = lf_ablation(diag["noco"]["L"], diag["noco"]["p_lm"], probe)
        st["rows_out"] = len(abl)

    # 선정 모델: 부트스트랩 안정성 + E3가 읽는 형식으로 저장
    with run.stage("write") as st:
        out, L, p_lm, lf_stats, m, Xf, t, fnames, lab = keep[best]
        mode = best.split("|")[1]
        covered = (L.to_numpy() != ABSTAIN).any(1)
        rng = np.random.default_rng(0)
        decisions = []
        for b in range(bootstrap):
            idx = rng.choice(len(arts), len(arts), replace=True)
            mb, _ = train_variant(Xf[idx], p_lm[idx], covered[idx], mode, seed=b)
            decisions.append(mb.predict_proba(Xf)[:, 1] >= t)
        ws = out["p_ws_lr"].to_numpy() >= t
        agree = float(np.mean([np.mean(d == ws) for d in decisions]))
        flip = float(np.mean(np.std(np.stack(decisions).astype(float), axis=0) > 0))
        out["lf_votes"] = L.apply(lambda r: ",".join(f"{k}:{'+' if v > 0 else '-'}" for k, v in r.items() if v != 0), axis=1).to_numpy()
        out["e22_variant"] = best
        pq = run.dir / "02_article_relevance.parquet"
        out.to_parquet(pq, index=False)
        run.artifact(pq, "article_relevance", rows=len(out))
        with pd.ExcelWriter(run.dir / "e22_variant_report.xlsx") as xw:
            sel.to_excel(xw, sheet_name="variants_gates", index=False)
            corr.to_excel(xw, sheet_name="lf_vote_correlation")
            abl.to_excel(xw, sheet_name="lf_ablation_orig", index=False)
            abl_noco.to_excel(xw, sheet_name="lf_ablation_noco", index=False)
            for v in ("orig", "noco"):
                diag[v]["lf_stats"].to_excel(xw, sheet_name=f"lf_stats_{v}", index=False)
        run.artifact(run.dir / "e22_variant_report.xlsx", "e22_variant_report")
        st["rows_out"] = len(out)

    decision = out["decision"].to_numpy()
    summary = {
        "selected_variant": best, "selected_pass_all": bool(sel.iloc[0]["pass_all"]), "prev_run_id": prev_id,
        "otsu_threshold": round(t, 3),
        "bootstrap": {"runs": bootstrap, "mean_agreement": round(agree, 4), "share_ever_flipped": round(flip, 4)},
        "decision_counts": pd.Series(decision).value_counts().to_dict(),
        "variants": sel.drop(columns=["base_flip_rate"]).to_dict("records"),
        "lf_content_leaning_ablation": abl[abl["removed_lf"] == "LF_content_leaning_score"].to_dict("records"),
        "search_company_dropped_articles": int(dropped.sum()),
        "anchors": {k: v for k, v in anc_info.items() if k != "top_work_names"},
        "models": [evaluate({f"m_rel_ws_lr ({best})": ws}, anc, arts["sectors"]).round(4).to_dict("records")[0]],
    }
    (run.dir / "e2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_relevance_v2_e22", "run_id": run.run_id, "task": "article_relevance",
        "data_snapshot_id": snap["data_snapshot_id"], "models": list(sel["variant"]), "selected": best,
        "evaluation": "anchor_proxy + probe + change_vs_prev", "otsu_threshold": summary["otsu_threshold"],
        "bootstrap": summary["bootstrap"], "result": summary["models"]})
    run.finish()
    return summary
