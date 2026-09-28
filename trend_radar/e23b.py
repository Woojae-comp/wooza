"""E2.3b 라벨 모델 보정 진단 (2026-09-28 사용자 결정, 3분류 라벨 모델 미세 조정의 마지막 실험).

- 규칙을 근거 계열별 투표 하나로 통합 (상관된 같은 방향 규칙의 이중 계산 제거). 세부 강도는 진단값으로만 보존.
  | 계열 | 내부 근거 | 출력 |
  |---|---|---|
  | F_content | 제목 씨앗어·본문 씨앗어 2개+·콘텐츠 점수 | CONTENT (시장 우세와 충돌하면 기권·충돌 기록) |
  | F_market | 시장 씨앗어 3개+·시장 점수·시장 우세 | MARKET |
  | F_other_industry | 다른 업종 어휘 2개+, 콘텐츠 씨앗어 없음 | OTHER |
  | F_politics_society | 네이버 정치·사회 섹션 (정책어·콘텐츠 앵커면 기권·충돌 기록) | OTHER |
  | F_broadcaster_source | 방송사 출처성 표기만 | OTHER |
  | F_content_policy | 콘텐츠 대상 정책·지원사업 | CONTENT |
  탐침 단어(정치 사건어)와 특징주·공시 태그는 평가(탐침·시장 앵커)에 쓰므로 규칙에 넣지 않는다.
- 구조: G3 (CONTENT/MARKET/OTHER 동시 추정) vs H2 (1단계 MARKET/NON-MARKET → 2단계 NON-MARKET 안에서 CONTENT/OTHER)
- 혼동행렬 사전분포: 중심 정확도 0.7, 가상 관측치 2 / 5 / 10 (Prior-L/M/H). 클래스 사전확률은 약한 대칭 디리클레(α=1).
- 라벨 모델 단계 기준을 통과한 구조에만 분류기(Soft A·신뢰도 가중)를 붙인다. 둘 다 실패하면 E2.1 유지.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .e23 import AB, C, CLASSES, M, O, VC, VM, VO, VOTE_NAME

FAMILIES = ["F_content", "F_market", "F_other_industry", "F_politics_society", "F_broadcaster_source", "F_content_policy"]
PRIORS = {"L": 2.0, "M": 5.0, "H": 10.0}


# ---------------------------------------------------------------- 근거 계열 통합

def families(arts: pd.DataFrame, kw: list[list[str]], lab: pd.DataFrame, cfg: dict, anc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """반환: 계열별 투표(부호), 진단용 세부 강도, 계열별 충돌 건수."""
    from .e2 import OTHER_INDUSTRY, PRICE_TITLE, lf_broadcaster_source_only, lf_politics_society_section

    lc, r2 = cfg["layers"], cfg.get("e2", {})
    cseed, mseed = set(lc["content_seed"]), set(lc["market_seed"])
    sets = [set(w) for w in kw]
    c = np.array([len(s & cseed) for s in sets])
    m = np.array([len(s & mseed) for s in sets])
    o = np.array([len(s & OTHER_INDUSTRY) for s in sets])
    titles = arts["title"].fillna("").to_numpy()
    texts = [f"{t} {s_}" for t, s_ in zip(arts["title"].fillna(""), arts["summary"].fillna(""))]
    title_c = np.array([any(w in t for w in (s & cseed)) for s, t in zip(sets, titles)]) & \
        ~np.array([bool(PRICE_TITLE.search(t)) for t in titles])
    share = lab["content_share"].to_numpy()
    cs, ms = lab["content_score"].to_numpy(), lab["market_score"].to_numpy()
    market_dom = (m >= 3) & (m > c)
    ac = anc["anchor_content"].to_numpy().astype(bool)
    policy_words = set(r2.get("policy_anchor_words", []))
    policy = np.array([any(w in t for w in policy_words) for t in texts])

    content_ev = np.c_[(c >= 2), title_c, (cs >= 3) & (share >= 0.7)].astype(int)
    market_ev = np.c_[(m >= 3) & (c == 0), (ms >= 3) & (share <= 0.3), market_dom & (c >= 1)].astype(int)
    content_any, market_any = content_ev.any(1), market_ev.any(1)
    content_conflict = content_any & market_dom            # 계열 안 충돌: 콘텐츠 근거 + 시장 우세 → 기권
    sec = np.isin(anc["sid"].to_numpy(), list(r2.get("excluded_sections", ["100", "102"])))
    pol_raw = lf_politics_society_section(anc["sid"].to_numpy(), ac, texts, policy_words, set(r2.get("excluded_sections", ["100", "102"])))
    politics_conflict = sec & (pol_raw == 0)                # 섹션은 정치·사회인데 정책어·콘텐츠 앵커 → 기권
    bro = lf_broadcaster_source_only(arts, {k: list(v) for k, v in (r2.get("broadcasters") or {}).items()},
                                     set(r2.get("broadcast_topic_words", [])) | cseed, ac)
    F = pd.DataFrame({
        "F_content": np.where(content_any & ~content_conflict, VC, AB),
        "F_market": np.where(market_any, VM, AB),
        "F_other_industry": np.where((o >= 2) & (c == 0), VO, AB),
        "F_politics_society": np.where(pol_raw != 0, VO, AB),
        "F_broadcaster_source": np.where(bro != 0, VO, AB),
        "F_content_policy": np.where(policy & (c >= 1), VC, AB),
    })
    strength = pd.DataFrame({"content_strength": content_ev.sum(1), "market_strength": market_ev.sum(1)})
    conflicts = {"F_content": int(content_conflict.sum()), "F_politics_society": int(politics_conflict.sum())}
    return F, strength, conflicts


# ---------------------------------------------------------------- 사전분포 있는 Dawid-Skene (K 클래스, 계열마다 한 방향)

def ds_fit(V: np.ndarray, K: int, code_class: dict[int, int], acc: float = 0.7, strength: float = 0.0, class_alpha: float = 1.0,
           w: np.ndarray | None = None, iters: int = 300, smooth: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V: n × J 투표 부호 (0 = 기권). code_class: 투표 부호 → 대상 클래스. 규칙마다 한 부호만 쓴다.
    사전분포: 대상 클래스의 투표 확률 평균 = K·acc·적용률, 다른 클래스 = K·(1-acc)/(K-1)·적용률 (균등 클래스일 때 정밀도 acc).
    가상 관측치 strength만큼 혼동행렬을 이 평균으로 수축. 클래스 사전확률은 대칭 디리클레(class_alpha).
    반환: 사후확률 q (n × K), θ (J × K × A), π."""
    n, J = V.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    A = int(max(code_class) + 1)
    onehot = np.stack([(V == a) for a in range(A)], axis=2).astype(float)
    codes = np.array([np.unique(V[:, j][V[:, j] != AB])[0] if (V[:, j] != AB).any() else AB for j in range(J)])
    cov = np.array([(w * (V[:, j] != AB)).sum() / w.sum() for j in range(J)])
    th0 = np.zeros((J, K, A))
    for j in range(J):
        v = codes[j]
        if v == AB:
            th0[j, :, AB] = 1
            continue
        t = code_class[v]
        for k in range(K):
            p = K * acc * cov[j] if k == t else K * (1 - acc) / (K - 1) * cov[j]
            th0[j, k, v] = min(max(p, 1e-4), 0.99)
            th0[j, k, AB] = 1 - th0[j, k, v]
    init = np.full((n, K), 0.2)
    for j in range(J):
        if codes[j] != AB:
            init[:, code_class[codes[j]]] += (V[:, j] == codes[j])
    q = init / init.sum(1, keepdims=True)
    for _ in range(iters):
        cnt = np.einsum("n,nk,nja->jka", w, q, onehot)
        th = (cnt + strength * th0 + smooth) / (cnt.sum(2, keepdims=True) + strength + smooth * A)
        for j in range(J):
            v = codes[j]
            if v == AB:
                continue
            t = code_class[v]
            k = int(np.argmax(th[j, :, v]))
            if k != t:
                th[j, t, v], th[j, k, v] = th[j, k, v], th[j, t, v]
            th[j] /= th[j].sum(1, keepdims=True)
        pi = ((w[:, None] * q).sum(0) + class_alpha) / (w.sum() + K * class_alpha)
        lq = np.log(pi)[None] + np.einsum("nja,jka->nk", onehot, np.log(th))
        lq -= lq.max(1, keepdims=True)
        q_new = np.exp(lq)
        q_new /= q_new.sum(1, keepdims=True)
        if np.max(np.abs(q_new - q)) < 1e-6:
            q = q_new
            break
        q = q_new
    return q, th, pi


def fit_g3(F: pd.DataFrame, strength: float) -> tuple[np.ndarray, dict]:
    q, th, pi = ds_fit(F.to_numpy(), 3, {VC: C, VM: M, VO: O}, strength=strength)
    return q, {"theta": th, "pi": pi}


def fit_h2(F: pd.DataFrame, strength: float) -> tuple[np.ndarray, dict]:
    """1단계 MARKET(0)/NON-MARKET(1): 시장 계열 → MARKET, 나머지 계열 → NON-MARKET.
    2단계 CONTENT(0)/OTHER(1): NON-MARKET 확률을 기사 가중치로, 콘텐츠·정책 계열 → CONTENT, 기타 계열 → OTHER (시장 계열 제외)."""
    V = F.to_numpy()
    V1 = np.where(V == VM, 1, np.where(V != AB, 2, AB))
    q1, th1, pi1 = ds_fit(V1, 2, {1: 0, 2: 1}, strength=strength)
    cols2 = [j for j, c in enumerate(F.columns) if c != "F_market"]
    V2 = V[:, cols2]
    V2 = np.where(V2 == VC, 1, np.where(V2 == VO, 2, AB))
    q2, th2, pi2 = ds_fit(V2, 2, {1: 0, 2: 1}, strength=strength, w=q1[:, 1])
    q = np.c_[q1[:, 1] * q2[:, 0], q1[:, 0], q1[:, 1] * q2[:, 1]]      # CONTENT, MARKET, OTHER
    pi = np.array([pi1[1] * pi2[0], pi1[0], pi1[1] * pi2[1]])
    return q, {"theta1": th1, "pi1": pi1, "theta2": th2, "pi2": pi2, "pi": pi}


FIT = {"G3": fit_g3, "H2": fit_h2}


# ---------------------------------------------------------------- 진단

def family_table(F: pd.DataFrame, info: dict, structure: str, conflicts: dict) -> pd.DataFrame:
    rows = []
    for j, f in enumerate(F.columns):
        v = F[f].to_numpy()
        row = {"family": f, "vote": VOTE_NAME.get(int(np.unique(v[v != AB])[0]), "-") if (v != AB).any() else "-",
               "coverage": round(float((v != AB).mean()), 4), "conflicts_abstained": conflicts.get(f, 0)}
        if structure == "G3":
            code = {"CONTENT": VC, "MARKET": VM, "OTHER": VO}.get(row["vote"])
            if code:
                for k, c in enumerate(CLASSES):
                    row[f"P(vote|{c})"] = round(float(info["theta"][j, k, code]), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def decide(q: np.ndarray, seed_layer: np.ndarray) -> tuple[np.ndarray, float]:
    from .e2 import otsu

    s = q[:, C]
    t = otsu(s)
    ws, lm = s >= t, q.argmax(1) == C
    return np.where(ws & seed_layer & lm, "INCLUDE", np.where(~ws & ~seed_layer & ~lm, "EXCLUDE", "REVIEW")), t


def lm_gates(r: dict, g: dict) -> dict:
    out = {"gate_probe": r["probe_in_rate_drop_pp"] >= g.get("probe_drop_pp", 5.0),
           "gate_anchor": r["anchor_in_rate_drop_pp"] <= g.get("anchor_drop_pp", 2.0),
           "gate_policy": r["policy_to_exclude_rate"] < g.get("policy_exclude_max", 0.01),
           "gate_broadcast": r["broadcast_excess_drop_pp"] <= g.get("broadcast_excess_pp", 2.0),
           "gate_content_not_collapsed": r["content_anchor_pred_content"] >= g.get("anchor_content_min", 0.5),
           "gate_family_sensitivity": r["max_family_removal_content_change_pp"] <= g.get("family_removal_pp", 10.0),
           "gate_prior_sensitivity": r.get("prior_LM_agreement", 1.0) >= g.get("prior_agreement_min", 0.95)}
    out["pass_all"] = all(out.values())
    return out


def prepare(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    """E2.3b 입력 준비 (근거 계열 투표까지). 검토 표본(review.py)이 후보 판정을 재현할 때도 쓴다."""
    from .config import Lexicon
    from .e2 import anchors
    from .e22 import drop_search_companies
    from .layers import classify
    from .load import build_corpus
    from .text import extract_keywords, prepped_texts, space_joined_names, tokenize_corpus

    r2 = cfg.get("e2", {})
    corpus = build_corpus(raw, cfg)
    arts = corpus.articles
    aliases = cfg["cleaning"].get("company_aliases") or {}
    names = space_joined_names(sorted(corpus.article_company["company"].unique()), aliases)
    texts = prepped_texts(arts, names)
    tokens = tokenize_corpus(texts, names, out_root / "cache")
    kw = extract_keywords(texts, tokens, Lexicon.load(), cfg)
    links = raw.drop_duplicates(cfg["input"]["columns"]["article_id"]).set_index(cfg["input"]["columns"]["article_id"])
    sid = links["네이버링크"].astype(str).str.extract(r"[?&]sid1?=(\d{3})")[0] if "네이버링크" in links else pd.Series(dtype=str)
    anc, _ = anchors(arts, sid)
    raw_texts = [f"{t} {s}" for t, s in zip(arts["title"], arts["summary"])]
    probe = np.array([any(w in t for w in r2.get("politics_probe_words", [])) for t in raw_texts]) | \
        np.isin(anc["sid"].to_numpy(), r2.get("excluded_sections", ["100", "102"]))
    kw_noco, _ = drop_search_companies(kw, arts, aliases, set(r2.get("industry_action_words", [])))
    lab, _ = classify(arts, kw_noco, cfg)
    F, strength_df, conflicts = families(arts, kw, lab, cfg, anc)
    return {"corpus": corpus, "arts": arts, "anc": anc, "raw_texts": raw_texts, "probe": probe, "kw": kw, "lab": lab,
            "seed_layer": lab["content"].to_numpy(), "F": F, "strength": strength_df, "conflicts": conflicts}


def candidate_decisions(p: dict, prior: str = "M") -> dict[str, np.ndarray]:
    """G3·H2 (사전분포 prior) 기사별 판정과 P(CONTENT). H2의 P(CONTENT) = P(NON-MARKET) × P(CONTENT | NON-MARKET)."""
    out = {}
    for s_name in ("G3", "H2"):
        q, _ = FIT[s_name](p["F"], PRIORS[prior])
        dec, _ = decide(q, p["seed_layer"])
        out[s_name] = (dec, q[:, C])
    return out


def run_e23b(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    from .config import Lexicon
    from .e2 import anchors, change_report
    from .e22 import drop_search_companies
    from .layers import classify
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .selection import mark_failed, selected_e2
    from .text import extract_keywords, prepped_texts, space_joined_names, tokenize_corpus

    r2 = cfg.get("e2", {})
    g = cfg.get("e23b_gates", {})
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e23b_label_model_diagnostic", cfg, out_root, snap["data_snapshot_id"])
    base_run, base_path, _ = selected_e2(cfg, REGISTRY, out_root, use_candidate=False)
    prev = pd.read_parquet(base_path)

    with run.stage("prepare", rows_in=len(raw)) as st:
        p = prepare(cfg, raw, out_root)
        corpus, arts, anc, raw_texts, probe = p["corpus"], p["arts"], p["anc"], p["raw_texts"], p["probe"]
        seed_l, F, strength_df, conflicts = p["seed_layer"], p["F"], p["strength"], p["conflicts"]
        st["rows_out"] = len(arts)

    ac = anc["anchor_content"].to_numpy().astype(bool)
    prev_in = prev["decision"].isin(["INCLUDE", "REVIEW"]).to_numpy()

    def evaluate_q(q: np.ndarray) -> tuple[dict, np.ndarray]:
        dec, t = decide(q, seed_l)
        out = pd.DataFrame({"gid": arts["gid"], "decision": dec})
        ch, _ = change_report(prev, out, arts, corpus.article_sector, anc, raw_texts, r2)
        sec = ch["sector_change_rate"]
        others = [v for k, v in sec.items() if k != "방송 및 영상"]
        am = q.argmax(1)
        return {"otsu": round(float(t), 3),
                "prior_CONTENT": None, "share_CONTENT": round(float((am == C).mean()), 4),
                "share_MARKET": round(float((am == M).mean()), 4), "share_OTHER": round(float((am == O).mean()), 4),
                "include": int((dec == "INCLUDE").sum()), "review": int((dec == "REVIEW").sum()), "exclude": int((dec == "EXCLUDE").sum()),
                "include_change_rel": round((dec == "INCLUDE").sum() / max((prev["decision"] == "INCLUDE").sum(), 1) - 1, 4),
                "probe_in_rate": ch["politics_probe"]["in_rate_new"],
                "probe_in_rate_drop_pp": round(100 * (ch["politics_probe"]["in_rate_prev"] - ch["politics_probe"]["in_rate_new"]), 2),
                "anchor_in_rate_drop_pp": round(100 * (ch["content_anchor"]["in_rate_prev"] - ch["content_anchor"]["in_rate_new"]), 2),
                "policy_to_exclude_rate": ch["policy_articles"]["rate"],
                "broadcast_excess_drop_pp": round(100 * (float(np.mean(others)) - sec.get("방송 및 영상", 0)), 2),
                "content_anchor_pred_content": round(float((am[ac] == C).mean()), 4),
                "content_anchor_to_MARKET": round(float((am[ac] == M).mean()), 4),
                "content_anchor_to_OTHER": round(float((am[ac] == O).mean()), 4),
                "probe_pred_content": round(float((am[probe] == C).mean()), 4)}, dec

    rows, fits, fam_tabs, sens_rows = [], {}, [], []
    with run.stage("label_models") as st:
        for s_name in ("G3", "H2"):
            decs = {}
            for p_name, strength in PRIORS.items():
                q, info = FIT[s_name](F, strength)
                r, dec = evaluate_q(q)
                r.update({"model": f"{s_name}/Prior-{p_name}", "structure": s_name, "prior": p_name,
                          "prior_CONTENT": round(float(info["pi"][C]), 4), "prior_MARKET": round(float(info["pi"][M]), 4),
                          "prior_OTHER": round(float(info["pi"][O]), 4)})
                decs[p_name] = dec
                fits[(s_name, p_name)] = (q, info, dec)
                rows.append(r)
                ft = family_table(F, info, s_name, conflicts)
                ft.insert(0, "model", f"{s_name}/Prior-{p_name}")
                fam_tabs.append(ft)
            # 사전분포 민감도: Prior-L/M 판정 일치율
            for r in rows:
                if r["structure"] == s_name:
                    r["prior_LM_agreement"] = round(float((decs["L"] == decs["M"]).mean()), 4)
                    r["prior_MH_agreement"] = round(float((decs["M"] == decs["H"]).mean()), 4)
            # 계열 제거 민감도 (Prior-M)
            q_m = fits[(s_name, "M")][0]
            base_share = (q_m.argmax(1) == C).mean()
            for f in F.columns:
                q_wo, info_wo = FIT[s_name](F.drop(columns=[f]), PRIORS["M"])
                sens_rows.append({"structure": s_name, "removed_family": f,
                                  "content_share_change_pp": round(100 * float((q_wo.argmax(1) == C).mean() - base_share), 2),
                                  "prior_CONTENT_without": round(float(info_wo["pi"][C]), 4),
                                  "mean_abs_change_P(CONTENT)": round(float(np.abs(q_wo[:, C] - q_m[:, C]).mean()), 4),
                                  "probe_mean_change_P(CONTENT)": round(float((q_wo[:, C] - q_m[:, C])[probe].mean()), 4)})
            mx = max(abs(x["content_share_change_pp"]) for x in sens_rows if x["structure"] == s_name)
            for r in rows:
                if r["structure"] == s_name:
                    r["max_family_removal_content_change_pp"] = mx
        st["rows_out"] = len(rows)

    res = pd.DataFrame(rows)
    gates = pd.DataFrame([lm_gates(r, g) for r in rows])
    res = pd.concat([res, gates], axis=1)
    sens = pd.DataFrame(sens_rows)
    fam = pd.concat(fam_tabs, ignore_index=True)
    corr = pd.DataFrame(np.corrcoef((F.to_numpy() != AB).astype(float).T), index=F.columns, columns=F.columns).round(3)

    with run.stage("diagnostics") as st:
        # 선정 후보(Prior-M 기준, 통과 구조) 또는 비교용 대표(Prior-M 중 통과 수 최대)
        mrows = res[res["prior"] == "M"].assign(n_pass=lambda d: d[[c for c in d.columns if c.startswith("gate_")]].sum(axis=1))
        passing = mrows[mrows["pass_all"]]
        show = (passing if len(passing) else mrows).sort_values(["n_pass", "probe_in_rate_drop_pp"], ascending=False).iloc[0]
        s_name = show["structure"]
        q, info, dec = fits[(s_name, "M")]
        am = q.argmax(1)
        # 탐침 중 CONTENT로 남은 이유: 계열 투표 패턴
        pat = F[probe & (am == C)].apply(lambda r: "+".join(f for f in F.columns if r[f] != AB) or "(투표 없음)", axis=1)
        probe_reason = pat.value_counts().rename_axis("families_voted").reset_index(name="articles").head(20)
        trans = pd.crosstab(pd.Series(prev["decision"].to_numpy(), name="E2.1"), pd.Series(dec, name=show["model"]))
        cases = []
        srt = np.sort(q, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        for k, c in enumerate(CLASSES):
            idx = np.where(am == k)[0]
            for kind, sel in (("representative", idx[np.argsort(-q[idx, k])[:30]]), ("boundary", idx[np.argsort(margin[idx])[:30]])):
                for i in sel:
                    cases.append({"class": c, "kind": kind, "title": arts["title"].iat[i],
                                  **{f"p_{cc.lower()}": round(float(q[i, kk]), 4) for kk, cc in enumerate(CLASSES)},
                                  "families": "+".join(f for f in F.columns if F[f].iat[i] != AB), "gid": arts["gid"].iat[i]})
        cases = pd.DataFrame(cases)
        with pd.ExcelWriter(run.dir / "e23b_report.xlsx") as xw:
            res.to_excel(xw, sheet_name="models_gates", index=False)
            fam.to_excel(xw, sheet_name="families", index=False)
            corr.to_excel(xw, sheet_name="family_correlation")
            sens.to_excel(xw, sheet_name="family_removal", index=False)
            probe_reason.to_excel(xw, sheet_name="probe_left_content", index=False)
            trans.to_excel(xw, sheet_name="transition_vs_E2.1")
            cases.to_excel(xw, sheet_name="class_cases", index=False)
            strength_df.describe().to_excel(xw, sheet_name="evidence_strength")
        run.artifact(run.dir / "e23b_report.xlsx", "e23b_report")
        st["rows_out"] = len(res)

    status = "label_model_passed" if len(passing) else "failed"
    if not len(passing):
        mark_failed(REGISTRY, run.run_id, "e2_3b", list(res["model"]), "G3·H2 모두 라벨 모델 단계 기준 미통과 → E2.1 유지, 독립 검증세트로 전환")
    summary = {"baseline_run_id": base_run, "status": status, "passing_structures": sorted(set(passing["structure"])) if len(passing) else [],
               "shown_model": show["model"], "family_coverage": {f: round(float((F[f] != AB).mean()), 4) for f in F.columns},
               "family_conflicts": conflicts, "models": res.to_dict("records"), "family_removal": sens.to_dict("records"),
               "probe_left_content_top": probe_reason.head(8).to_dict("records")}
    (run.dir / "e23b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_relevance_v2_e23b", "run_id": run.run_id, "task": "label_model_diagnostic",
        "data_snapshot_id": snap["data_snapshot_id"], "baseline_run_id": base_run, "status": status,
        "result": res[["model", "probe_in_rate_drop_pp", "anchor_in_rate_drop_pp", "policy_to_exclude_rate",
                       "content_anchor_pred_content", "pass_all"]].to_dict("records")})
    run.finish()
    return summary
