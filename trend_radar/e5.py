"""E5a 키워드 시계열 신호 (v2 설계, 기획안 8장). 결과는 trend가 아니라 keyword_signal.

최종 트렌드 유형은 E4(주제 군집·네트워크)와 결합한 E5b에서 확정한다.

월별 집계
- raw_doc_freq: 판정과 무관한 전체 기사 수
- weighted_doc_freq: 기본 INCLUDE 1 / REVIEW 0.5 / EXCLUDE 0 (민감도: 기사별 p_rel)
- share_per_1000_articles: 같은 달 가중 기사 합 대비 1,000건당
- sector_adjusted_share: 분야별 비중을 전체 기간 분야 구성비로 가중 평균 (월별 분야 구성 변화 보정)
- 누락된 달은 0으로 채운다.

지표: 전월 대비·최근 3개월 대비 증가율(평활), Z(평균·표준편차)와 Robust Z(중앙값·MAD),
Kleinberg 버스트(s·γ 3개 조합), 지속성(최근 6·12개월 등장 비율, 연속 등장), 신규성, 분야 확산, 편중도.
월간 가중 문서빈도가 min_df 미만이면 증가율·Z로 유형을 주지 않는다 (Insufficient).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BURST_PARAMS = [(2.0, 1.0), (3.0, 1.0), (2.0, 0.5)]   # (s, gamma)


# ---------------------------------------------------------------- 집계

def month_axis(dates: pd.Series) -> list[str]:
    p = dates.dt.to_period("M")
    return [str(m) for m in pd.period_range(p.min(), p.max(), freq="M")]   # 기사 없는 달도 포함


def monthly(X: sparse.csr_matrix, dates: pd.Series, w: np.ndarray, months: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(키워드 × 월) 가중 문서빈도와 월별 가중 기사 합. 기사 없는 달은 0."""
    mi = {m: i for i, m in enumerate(months)}
    col = dates.dt.to_period("M").astype(str).map(mi).to_numpy()
    P = sparse.csr_matrix((w, (np.arange(len(w)), col)), shape=(len(w), len(months)))
    return np.asarray((X.T @ P).todense()), np.asarray(P.sum(0)).ravel()


def safe_share(W: np.ndarray, T: np.ndarray, per: float = 1000.0) -> np.ndarray:
    return np.divide(W * per, T, out=np.zeros_like(W, dtype=float), where=T > 0)


# ---------------------------------------------------------------- 지표

def growth(cur_w: float, cur_t: float, base_w: float, base_t: float) -> float:
    """평활 비중 비율 ((W+1)/T) / ((W0+1)/T0). 직전이 0이어도 유한하다."""
    if cur_t <= 0 or base_t <= 0:
        return 1.0
    return ((cur_w + 1) / cur_t) / ((base_w + 1) / base_t)


def zscores(y: np.ndarray, hist: np.ndarray) -> tuple[float, float]:
    """(Z, Robust Z). 변동이 없으면 값이 같을 때 0, 다르면 ±10으로 자른다."""
    y = float(y)
    mu, sd = float(np.mean(hist)), float(np.std(hist))
    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med))) * 1.4826

    def z(center, scale):
        d = y - center
        if scale < 1e-12:
            return 0.0 if abs(d) < 1e-12 else float(np.sign(d) * 10)
        return float(np.clip(d / scale, -10, 10))
    return z(mu, sd), z(med, mad)


def kleinberg(r: np.ndarray, d: np.ndarray, s: float = 2.0, gamma: float = 1.0) -> np.ndarray:
    """Kleinberg(2002) 2상태 배치 버스트. r: 대상 문서 수(가중 가능), d: 전체 문서 수. 상태 0/1 (결정적)."""
    r = np.asarray(r, float)
    d = np.asarray(d, float)
    n = len(r)
    if n == 0 or d.sum() <= 0 or r.sum() <= 0:
        return np.zeros(n, dtype=int)
    p0 = r.sum() / d.sum()
    p1 = min(p0 * s, 0.9999)
    p = [p0, p1]

    def cost(i, t):
        if d[t] <= 0:
            return 0.0
        rt = min(r[t], d[t])
        return -(rt * np.log(p[i]) + (d[t] - rt) * np.log(1 - p[i]))

    trans = gamma * np.log(n)
    C = np.zeros((n, 2))
    B = np.zeros((n, 2), dtype=int)
    C[0] = [cost(0, 0), cost(1, 0) + trans]
    for t in range(1, n):
        for j in (0, 1):
            c0 = C[t - 1, 0] + (trans if j == 1 else 0.0)
            c1 = C[t - 1, 1]
            B[t, j] = 0 if c0 <= c1 else 1
            C[t, j] = min(c0, c1) + cost(j, t)
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmin(C[-1]))
    for t in range(n - 1, 0, -1):
        states[t - 1] = B[t, states[t]]
    return states


def run_lengths(active: np.ndarray) -> tuple[int, int]:
    """(현재 연속 길이, 최장 연속 길이)."""
    cur = best = run = 0
    for a in active:
        run = run + 1 if a else 0
        best = max(best, run)
    for a in active[::-1]:
        if not a:
            break
        cur += 1
    return cur, best


def signal_row(W: np.ndarray, T: np.ndarray, rules: dict, t: int | None = None) -> dict:
    """한 키워드의 월별 가중 빈도 W와 월별 가중 기사 합 T로 시점 t(기본: 마지막 달)의 지표와 유형.
    W[0]은 수집 첫 달이다. 첫 censor_months(기본 3)개월 안에 이미 등장했으면 left_censored (실제 최초 등장 시점 모름)
    → Emerging을 주지 않는다."""
    t = len(W) - 1 if t is None else t
    left_censored = bool((W[: rules.get("censor_months", 3)] >= 1).any())
    W, T = W[: t + 1], T[: t + 1]
    y = safe_share(W, T)
    L = len(W)
    md = rules.get("min_monthly_df", 5)
    prev = W[L - 2] if L >= 2 else 0.0
    prev_t = T[L - 2] if L >= 2 else 0.0
    g_mom = growth(W[-1], T[-1], prev, prev_t)
    b3 = slice(max(0, L - 4), L - 1)
    g_3m = growth(W[-1], T[-1], W[b3].sum() / max(L - 1 - b3.start, 1), T[b3].sum() / max(L - 1 - b3.start, 1))
    hist = y[max(0, L - 13): L - 1]
    z, rz = zscores(y[-1], hist) if len(hist) >= 3 else (0.0, 0.0)
    r6, p6 = slice(max(0, L - 6), L), slice(max(0, L - 12), max(0, L - 6))
    ratio_6 = growth(W[r6].sum(), T[r6].sum(), W[p6].sum(), T[p6].sum()) if L >= 12 else 1.0
    present = W >= 1
    pers6 = float(present[-6:].mean()) if L >= 6 else float(present.mean())
    pers12 = float(present[-12:].mean()) if L >= 12 else float(present.mean())
    run_cur, run_max = run_lengths(present)
    first = int(np.argmax(present)) if present.any() else L
    appeared_before = bool(present[: max(0, L - 12)].any())
    bursts = {f"s{s}_g{g}": kleinberg(W, T, s, g) for s, g in BURST_PARAMS}
    base = bursts[f"s{BURST_PARAMS[0][0]}_g{BURST_PARAMS[0][1]}"]
    last12 = W[-12:]
    peak_dom = float(last12.max() / last12.sum()) if last12.sum() > 0 else 0.0
    burst_recent = bool(base[-3:].any())
    burst_len = int(base[-12:].sum())
    enough = W[-1] >= md or W[-3:].mean() >= md

    if not enough:
        kind = "Insufficient"
    elif burst_recent and burst_len <= rules.get("spike_max_months", 2) and \
            (pers12 < rules.get("spike_max_persistence", 0.5) or peak_dom >= rules.get("spike_peak_dominance", 0.5)):
        kind = "Event Spike"
    elif not left_censored and not appeared_before and g_3m >= rules.get("emerging_growth", 2.0) and pers6 >= 0.5:
        kind = "Emerging"
    elif ratio_6 >= rules.get("growing_ratio", 1.3) and pers12 >= 0.5 and rz >= 0:
        kind = "Growing"
    elif ratio_6 <= rules.get("declining_ratio", 0.7) and rz < 0:
        kind = "Declining"
    elif pers12 >= rules.get("established_persistence", 0.9) and abs(ratio_6 - 1) < 0.3:
        kind = "Established"
    else:
        kind = "Stable"
    return {"growth_mom": g_mom, "growth_3m": g_3m, "ratio_6m": ratio_6, "z": z, "robust_z": rz,
            "persistence_6m": pers6, "persistence_12m": pers12, "run_current": run_cur, "run_longest": run_max,
            "first_month_idx": first, "appeared_before_12m": appeared_before, "peak_dominance_12m": peak_dom,
            "burst_recent_3m": burst_recent, "burst_months_12m": burst_len,
            **{f"burst_{k}_recent": bool(v[-3:].any()) for k, v in bursts.items()},
            "left_censored": left_censored, "weighted_df_last": float(W[-1]), "signal_type": kind}


# ---------------------------------------------------------------- 실행

def latest_run(registry: Path, task: str) -> str:
    recs = [json.loads(l) for l in (registry / "experiment_registry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in recs if r.get("task") == task][-1]["run_id"]


def run_e5a(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot

    rules = cfg.get("e5", {})
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e5a_keyword_signal", cfg, out_root, snap["data_snapshot_id"])
    from .selection import selected_e2, selected_e3
    e3_run = selected_e3(REGISTRY, selected_e2(cfg, REGISTRY, out_root)[0])
    e3_dir = out_root / "runs" / e3_run

    with run.stage("load") as st:
        X = sparse.load_npz(e3_dir / "03_doc_term.npz").tocsc()
        vocab = json.loads((e3_dir / "03_doc_term_vocab.json").read_text(encoding="utf-8"))
        rows = pd.read_csv(e3_dir / "03_doc_term_rows.csv", dtype=str)["gid"]
        dic = pd.read_csv(e3_dir / "03_keyword_dictionary.csv")
        inp = pd.read_parquet(e3_dir / "03_e3_input_articles.parquet").set_index("article_id")
        corpus = build_corpus(raw, cfg)
        arts = corpus.articles.reset_index(drop=True)
        assert arts["gid"].astype(str).tolist() == rows.tolist(), "E3 행렬과 기사 집합이 다르다 (스냅샷 확인)"
        rows = arts["gid"]
        keep = dic[dic["class_auto"].isin(rules.get("input_classes", ["CORE", "EXTENDED", "EMERGING"]))]
        idx = {t: j for j, t in enumerate(vocab)}
        keep = keep[keep["keyword"].isin(idx)]
        cols = keep["keyword"].map(idx).to_numpy()
        Xk = X[:, cols].tocsr()
        half = inp.loc[rows, "content_weight_half"].to_numpy()
        soft = inp.loc[rows, "content_weight_soft"].to_numpy()
        st["rows_out"] = len(keep)

    with run.stage("monthly") as st:
        months = month_axis(arts["date"])
        # 불완전월: 마지막 기사일이 그 달의 말일이 아니면 잠정 (확정 판정·E5b 최종 유형에서 제외)
        last_day = arts["date"].max()
        partial = bool(last_day.day < last_day.days_in_month)
        tc = len(months) - 1 - int(partial)          # 확정 판정 시점 (마지막 완전월)
        cm = months[max(0, tc - 11): tc + 1]         # 최근 12개 완전월
        Wh, Th = monthly(Xk, arts["date"], half, months)
        Ws, Ts = monthly(Xk, arts["date"], soft, months)
        Wr, Tr = monthly(Xk, arts["date"], np.ones(len(arts)), months)
        # 분야 보정 비중 (핵심 분야, 전체 기간 구성비로 가중)
        related = cfg["cleaning"].get("related_label", "연관산업")
        sec = corpus.article_sector[corpus.article_sector["sector"] != related]
        gid_row = pd.Series(np.arange(len(arts)), index=arts["gid"])
        sectors = sorted(sec["sector"].unique())
        adj = np.zeros_like(Wh)
        sector_df = np.zeros((len(keep), len(sectors)))
        wt = []
        for i, s in enumerate(sectors):
            r = sec.loc[sec["sector"] == s, "gid"].map(gid_row).dropna().astype(int).to_numpy()
            m = np.zeros(len(arts))
            m[r] = 1
            Wsec, Tsec = monthly(Xk, arts["date"], half * m, months)
            adj += Tsec.sum() * safe_share(Wsec, Tsec)
            wt.append(Tsec.sum())
            sector_df[:, i] = Wsec[:, max(0, tc - 11): tc + 1].sum(1)
        adj /= max(sum(wt), 1e-9)
        long = []
        share = safe_share(Wh, Th)
        for k, kw in enumerate(keep["keyword"]):
            for m_i, m in enumerate(months):
                if Wr[k, m_i] > 0:
                    long.append((kw, m, int(Wr[k, m_i]), round(float(Wh[k, m_i]), 2), round(float(share[k, m_i]), 3),
                                 round(float(adj[k, m_i]), 3), int(partial and m_i == len(months) - 1)))
        mon = pd.DataFrame(long, columns=["keyword", "month", "raw_doc_freq", "weighted_doc_freq",
                                          "share_per_1000_articles", "sector_adjusted_share", "is_partial_month"])
        pm = run.dir / "06a_keyword_signal_monthly.parquet"
        mon.to_parquet(pm, index=False)
        run.artifact(pm, "keyword_signal_monthly", rows=len(mon))
        st["rows_out"] = len(mon)

    with run.stage("signals") as st:
        # 편중도: 최근 12개월 해당 키워드 기사에서 상위 기업 점유율
        recent = arts["date"].dt.to_period("M").astype(str).isin(cm).to_numpy()
        comp = arts["companies"].to_numpy()
        out = []
        for k, kw in enumerate(keep["keyword"]):
            a = signal_row(Wh[k], Th, rules, tc)
            b = signal_row(Ws[k], Ts, rules, tc)
            prov = signal_row(Wh[k], Th, rules)["signal_type"] if partial else a["signal_type"]
            rr = Xk[:, k].nonzero()[0]
            rr = rr[recent[rr] & (half[rr] > 0)]
            top_co, top_share = "", 0.0
            if len(rr):
                cc = pd.Series([c for r in rr for c in comp[r]]).value_counts()
                top_co, top_share = cc.index[0], float(cc.iloc[0] / len(rr))
            sd = sector_df[k]
            active = int((sd >= rules.get("sector_min_df", 3)).sum())
            p = sd / sd.sum() if sd.sum() > 0 else sd
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(sectors))) if sd.sum() > 0 else 0.0
            row = keep.iloc[k]
            out.append({
                "keyword": kw, "keyword_id": row["keyword_id"], "class_auto": row["class_auto"], "entity_type": row["entity_type"],
                **{k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in a.items()},
                "first_month": months[a["first_month_idx"]] if a["first_month_idx"] < len(months) else None,
                "signal_asof": months[tc], "signal_type_provisional": prov,
                "provisional_month": months[-1] if partial else None,
                "signal_type_soft": b["signal_type"], "sensitivity_flag": int(a["signal_type"] != b["signal_type"]),
                "sectors_active_12m": active, "sector_entropy_12m": round(ent, 3),
                "cross_sector": bool(active >= rules.get("cross_min_sectors", 3) and ent >= rules.get("cross_min_entropy", 0.6)),
                "top_company_12m": top_co, "top_company_share_12m": round(top_share, 3),
                "raw_doc_freq_total": int(Wr[k].sum()), "weighted_doc_freq_total": round(float(Wh[k].sum()), 1),
            })
        sig = pd.DataFrame(out).drop(columns=["first_month_idx"])
        ps = run.dir / "06a_keyword_signal.csv"
        sig.to_csv(ps, index=False, encoding="utf-8-sig")
        run.artifact(ps, "keyword_signal", rows=len(sig))
        burst_cols = [c for c in sig.columns if c.startswith("burst_s")]
        burst_agree = {f"{a} vs {b}": round(float((sig[a] == sig[b]).mean()), 4)
                       for i, a in enumerate(burst_cols) for b in burst_cols[i + 1:]}
        st["rows_out"] = len(sig)

    with run.stage("backtest") as st:
        bt = backtest(e3_dir, keep, Wh[:, : tc + 1], Th[: tc + 1], months[: tc + 1], rules)
        pb = run.dir / "06a_backtest.csv"
        bt.to_csv(pb, index=False, encoding="utf-8-sig")
        run.artifact(pb, "keyword_signal_backtest", rows=len(bt))
        st["rows_out"] = len(bt)

    summary = {
        "e3_run_id": e3_run, "keywords": len(sig), "months": [months[0], months[-1]],
        "signal_asof": months[tc], "partial_month": months[-1] if partial else None,
        "last_article_date": str(last_day.date()),
        "signal_counts_provisional": sig["signal_type_provisional"].value_counts().to_dict(),
        "declining_recheck": {
            "provisional_declining": int((sig["signal_type_provisional"] == "Declining").sum()),
            "confirmed_declining": int((sig["signal_type"] == "Declining").sum()),
            "both": int(((sig["signal_type_provisional"] == "Declining") & (sig["signal_type"] == "Declining")).sum())},
        "left_censored": int(sig["left_censored"].sum()),
        "signal_counts": sig["signal_type"].value_counts().to_dict(),
        "signal_counts_soft": sig["signal_type_soft"].value_counts().to_dict(),
        "sensitivity_flag_rate": round(float(sig["sensitivity_flag"].mean()), 4),
        "cross_sector": int(sig["cross_sector"].sum()),
        "burst_param_agreement": burst_agree,
        "backtest": backtest_summary(bt),
        "top": {k: sig[sig["signal_type"] == k].sort_values("weighted_df_last", ascending=False)["keyword"].head(30).tolist()
                for k in ("Emerging", "Growing", "Event Spike", "Declining", "Established")},
    }
    (run.dir / "e5a_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_signal_v2_e5a", "run_id": run.run_id, "task": "keyword_signal",
        "data_snapshot_id": snap["data_snapshot_id"], "input_e3_run_id": e3_run,
        "models": ["growth+zscore+kleinberg (half weight)", "same (soft weight, sensitivity)"],
        "result": {k: summary[k] for k in ("signal_counts", "sensitivity_flag_rate", "burst_param_agreement", "backtest")}})
    run.finish()
    return summary


# ---------------------------------------------------------------- 백테스트 (참고 기준, 정답 아님)

def backtest(e3_dir: Path, keep: pd.DataFrame, W: np.ndarray, T: np.ndarray, months: list[str], rules: dict) -> pd.DataFrame:
    """보호 개체(작품명)의 제목 따옴표 첫 등장 달을 발표 시점의 대리값으로, 따옴표 등장 최다 달을 사건 시점의 대리값으로 쓴다.
    출처는 파생값이며 정답이 아니다 (루머·제작 발표가 먼저 나올 수 있고, 수집 시작 이전 등장은 알 수 없음)."""
    ents = pd.read_csv(e3_dir / "05_protected_entities.csv")
    pos = {t: k for k, t in enumerate(keep["keyword"])}
    mi = {m: i for i, m in enumerate(months)}
    rows = []
    for e in ents.itertuples():
        if e.entity not in pos:
            continue
        k = pos[e.entity]
        ann = mi.get(e.first_month)
        if ann is None:
            continue
        peak = int(np.argmax(W[k]))
        detect, kinds = None, []
        for t in range(len(months)):
            s = signal_row(W[k], T, rules, t)
            kinds.append(s["signal_type"])
            if detect is None and s["signal_type"] in ("Emerging", "Growing", "Event Spike"):
                detect = t
        final = kinds[-1]
        always_present = ann < rules.get("censor_months", 3)   # left_censored 구간
        rows.append({
            "event_id": f"ev_{e.token}", "keyword_id": keep.iloc[k]["keyword_id"], "keyword": e.entity,
            "event_type": "work_title_mention (proxy)", "announcement_month": months[ann],
            "release_or_event_month": months[peak], "expected_detection_window": f"{months[ann]}~{months[min(peak + 1, len(months) - 1)]}",
            "source": "derived:title_quotes (not ground truth)",
            "first_detect_month": months[detect] if detect is not None else None,
            "lag_vs_announcement": (detect - ann) if detect is not None else None,
            "lag_vs_event": (detect - peak) if detect is not None else None,
            "detected_in_window": detect is not None and ann <= detect <= peak + 1,
            "final_signal": final,
            "spike_misread_as_growing": ("Event Spike" in kinds) and ("Growing" in kinds[kinds.index("Event Spike"):]),
            "always_present_but_emerging": always_present and ("Emerging" in kinds),
        })
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}
    det = bt["first_detect_month"].notna()
    return {"events": len(bt), "detected": int(det.sum()),
            "detected_in_window_rate": round(float(bt["detected_in_window"].mean()), 3),
            "median_lag_vs_announcement": float(bt.loc[det, "lag_vs_announcement"].median()) if det.any() else None,
            "median_lag_vs_event": float(bt.loc[det, "lag_vs_event"].median()) if det.any() else None,
            "spike_misread_as_growing": int(bt["spike_misread_as_growing"].sum()),
            "always_present_but_emerging": int(bt["always_present_but_emerging"].sum()),
            "note": "대리 기준(제목 따옴표 첫 등장·최다 등장 달). 정답 아님"}
