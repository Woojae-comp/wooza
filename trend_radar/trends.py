"""트렌드 탐지와 분류 (설계 10~16장, 19장).

하나의 종합점수로 서열화하지 않는다. 키워드마다 상태(복수 가능)와 그 근거 지표를 남긴다.

상태
- Emerging     : 기존 비중이 낮았는데 최근 빠르게 증가
- Growing      : 여러 분기 연속 비중 증가
- Established  : 높은 비중을 오래 안정적으로 유지
- Spreading    : 유의미하게 등장하는 분야 수가 늘어남
- Converging   : 이전에 다른 군집에 있던 키워드와 새로 가까워짐
- Declining    : 비중과 네트워크 연결이 함께 감소
- Event-driven : 특정 월에만 급증하고 되돌아감
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .metrics import period_counts
from .network import Network, build_network, cluster_articles, match_clusters
from .text import DocTerm

STATUS_ORDER = ["Emerging", "Growing", "Established", "Spreading", "Converging", "Declining", "Event-driven"]


# ---------------------------------------------------------------- 기간 구간

@dataclass
class Windows:
    quarters: list[str]
    months: list[str]
    recent_q: list[str]
    base_q: list[str]
    recent12: list[str]
    prev12: list[str]
    first12: list[str]
    years: list[str]
    halves: list[str]

    def rows(self, arts: pd.DataFrame, unit: str, labels: list[str]) -> np.ndarray:
        return np.flatnonzero(arts[unit].isin(labels).to_numpy())


def make_windows(arts: pd.DataFrame, cfg: dict) -> Windows:
    pc = cfg["periods"]
    qs = sorted(arts["quarter"].unique())
    ms = sorted(arts["month"].unique())
    r, b, w = pc["recent_quarters"], pc["base_quarters"], pc["window_months"]
    return Windows(
        quarters=qs, months=ms,
        recent_q=qs[-r:], base_q=qs[-(r + b):-r],
        recent12=ms[-w:], prev12=ms[-2 * w:-w], first12=ms[:w],
        years=sorted(arts["year"].unique()),
        halves=sorted(arts["half"].unique()),
    )


def add_half(arts: pd.DataFrame) -> pd.DataFrame:
    a = arts
    a["half"] = a["year"] + np.where(a["date"].dt.month <= 6, "H1", "H2")
    return a


# ---------------------------------------------------------------- 상태 분류

def _spearman_rows(y: np.ndarray) -> np.ndarray:
    """행마다 시간 순서와의 순위상관."""
    n = y.shape[1]
    if n < 3:
        return np.zeros(y.shape[0])
    ry = rankdata(y, axis=1)
    rx = np.arange(1, n + 1, dtype=float)
    ry = ry - ry.mean(axis=1, keepdims=True)
    rx = rx - rx.mean()
    den = np.sqrt((ry ** 2).sum(axis=1) * (rx ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ry @ rx) / den
    return np.nan_to_num(r)


def status_frame(dt: DocTerm, arts: pd.DataFrame, rows: np.ndarray, win: Windows, cfg: dict,
                 company_X=None, company_names: list[str] | None = None) -> pd.DataFrame:
    """rows(전체 또는 분야) 안에서 키워드별 시계열 지표와 상태."""
    sc = cfg["status"]
    V = len(dt.vocab)
    sub = arts.iloc[rows]
    qp, qc, qt = period_counts(dt, arts["quarter"], rows)
    mp, mc, mt = period_counts(dt, arts["month"], rows)
    # 전체 기간 축에 맞춘다 (분야에 기사가 없는 분기 대비)
    qc, qt = _align(qp, qc, qt, win.quarters)
    mc, mt = _align(mp, mc, mt, win.months)
    qs = qc / np.maximum(qt, 1)
    ms = mc / np.maximum(mt, 1)

    qi = {q: i for i, q in enumerate(win.quarters)}
    ri = [qi[q] for q in win.recent_q]
    bi = [qi[q] for q in win.base_q]
    r_df, r_n = qc[:, ri].sum(1), qt[ri].sum()
    b_df, b_n = qc[:, bi].sum(1), qt[bi].sum()
    r_sh = r_df / max(r_n, 1)
    b_sh = b_df / max(b_n, 1)
    ratio = ((r_df + 1) / max(r_n, 1)) / ((b_df + 1) / max(b_n, 1))

    # 월 단위 급증 (Event-driven)
    M = ms.shape[1]
    peak = ms.argmax(axis=1)
    peak_share = ms[np.arange(V), peak]
    peak_df = mc[np.arange(V), peak]
    mask = np.ones_like(ms, dtype=bool)
    for d in (-1, 0, 1):
        idx = np.clip(peak + d, 0, M - 1)
        mask[np.arange(V), idx] = False
    base_level = np.where(mask.any(1), (ms * mask).sum(1) / np.maximum(mask.sum(1), 1), 0)
    floor = 1.0 / max(np.median(mt[mt > 0]), 1)
    peak_mult = peak_share / np.maximum(base_level, floor)
    win_df = np.zeros(V)
    after = np.full(V, np.nan)
    for v in range(V):
        lo, hi = max(0, peak[v] - 3), min(M, peak[v] + 4)
        win_df[v] = mc[v, lo:hi].sum()
        a = ms[v, peak[v] + 2: peak[v] + 5]
        if a.size:
            after[v] = a.mean() / max(base_level[v], floor)
    conc = peak_df / np.maximum(win_df, 1)
    decayed = np.where(np.isnan(after), False, after < 2.0)
    ongoing = peak >= M - 2
    event = (peak_mult >= sc["event_peak_multiple"]) & (peak_df >= sc["event_min_peak_df"]) & \
            (conc >= sc["event_concentration"]) & (decayed | ongoing)

    # 급증 월을 뺀 최근/기준 비율 (급증 하나로 증가처럼 보이는지 확인)
    peak_label = np.array(win.months)[peak]
    peak_q = pd.PeriodIndex(peak_label, freq="M").asfreq("Q").astype(str).to_numpy()
    in_recent = np.isin(peak_q, win.recent_q)
    r_df2 = np.where(in_recent, r_df - peak_df, r_df)
    r_n2 = np.where(in_recent, r_n - mt[peak], r_n)
    ratio_wo_peak = ((r_df2 + 1) / np.maximum(r_n2, 1)) / ((b_df + 1) / max(b_n, 1))

    trend = _spearman_rows(qs[:, -6:])
    coverage = (qc > 0).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.nan_to_num(qs.std(1) / qs.mean(1), nan=9.9)
    mean_share = qs.mean(1)

    # 증가 지속기간: 3개월 이동평균이 3개월 전보다 줄지 않은 최근 연속 개월 수
    ma = pd.DataFrame(ms.T).rolling(3, min_periods=3).mean().to_numpy().T
    grow = np.zeros_like(ma, dtype=bool)
    grow[:, 3:] = ma[:, 3:] >= ma[:, :-3] * 0.95
    grow &= ma > 0
    duration = np.zeros(V, dtype=int)
    for v in range(V):
        g = grow[v][::-1]
        duration[v] = len(g) if g.all() else int(np.argmin(g))

    first_idx = np.argmax(mc > 0, axis=1)
    first_month = np.array(win.months)[first_idx]

    s = pd.DataFrame({
        "keyword": dt.vocab,
        "articles": qc.sum(1).astype(int),
        "recent_df": r_df.astype(int), "base_df": b_df.astype(int),
        "recent_share": r_sh, "base_share": b_sh, "ratio": ratio, "ratio_wo_peak": ratio_wo_peak,
        "trend6q": trend, "coverage": coverage, "cv": cv, "mean_share": mean_share,
        "peak_month": peak_label, "peak_df": peak_df.astype(int), "peak_multiple": peak_mult,
        "peak_concentration": conc, "event_ongoing": ongoing & event, "growth_months": duration,
        "first_month": first_month,
    })
    emerging = (r_df >= sc["emerging_min_recent_df"]) & (b_sh <= sc["emerging_max_base_share"]) & \
               (ratio >= sc["emerging_min_ratio"]) & (ratio_wo_peak >= sc["emerging_min_ratio"])
    growing = (ratio >= sc["growing_min_ratio"]) & (ratio_wo_peak >= sc["growing_min_ratio"]) & \
              (r_df >= sc["growing_min_recent_df"]) & (trend >= sc["growing_min_trend"]) & \
              (b_df >= sc.get("growing_min_base_df", 20))
    established = (mean_share >= sc["established_min_share"]) & (cv <= sc["established_max_cv"]) & (coverage >= 0.9)
    declining = (ratio <= sc["declining_max_ratio"]) & (b_df >= sc["declining_min_base_df"]) & (trend <= -sc["growing_min_trend"])
    # 급증 하나로만 설명되는 증가는 Event-driven으로만 남긴다
    s["Emerging"] = emerging
    s["Growing"] = growing & ~established
    s["Established"] = established
    s["Declining"] = declining
    s["Event-driven"] = event & ~(emerging | growing)
    s["event_spike"] = event
    s["Spreading"] = False
    s["Converging"] = False
    # 기업 집중도: 최근 구간 기사 중 한 기업 검색에서 온 비율 (설계 14장, 일시적·국지적 이슈 구분)
    if company_X is not None:
        rrows = rows[np.isin(arts["quarter"].to_numpy()[rows], win.recent_q)]
        cm = np.asarray((dt.X[rrows].T @ company_X[rrows]).todense())
        tot = np.maximum(cm.sum(1), 1)
        top = cm.argmax(1)
        has = cm.sum(1) > 0                              # 최근 구간에 기업 연결이 없으면 '해당 없음' (argmax 0번 기업을 쓰지 않는다)
        s["top_company"] = np.where(has, np.array(company_names)[top], "")
        s["top_company_share"] = np.where(has, cm[np.arange(V), top] / tot, np.nan)
        s["n_companies"] = (cm > 0).sum(1)
        s["company_specific"] = (np.nan_to_num(s["top_company_share"]) >= sc.get("company_specific_share", 0.6)) & (r_df > 0)
    return s


def _align(periods: list[str], counts: np.ndarray, totals: np.ndarray, axis: list[str]):
    idx = {p: i for i, p in enumerate(periods)}
    c = np.zeros((counts.shape[0], len(axis)))
    t = np.zeros(len(axis))
    for k, p in enumerate(axis):
        if p in idx:
            c[:, k] = counts[:, idx[p]]
            t[k] = totals[idx[p]]
    return c, t


# ---------------------------------------------------------------- 분야 확산 (11장)

def sector_matrix(dt: DocTerm, arts: pd.DataFrame, sector_rows: dict[str, np.ndarray], months: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(키워드 × 분야) 기사 수와 분야별 기사 수, months 구간 안에서."""
    inwin = arts["month"].isin(months).to_numpy()
    cols, tots = [], []
    for sec, rows in sector_rows.items():
        r = rows[inwin[rows]]
        cols.append(np.asarray(dt.X[r].sum(axis=0)).ravel())
        tots.append(len(r))
    return np.stack(cols, axis=1), np.array(tots)


def arrow(ratio: float, sig_now: bool, sig_before: bool) -> str:
    if not sig_now and not sig_before:
        return "-"
    if ratio >= 2.0:
        return "↑↑"
    if ratio >= 1.25:
        return "↑"
    if ratio <= 0.5:
        return "↓↓"
    if ratio <= 0.8:
        return "↓"
    return "→"


def diffusion(dt: DocTerm, arts: pd.DataFrame, sector_rows: dict[str, np.ndarray], win: Windows, cfg: dict) -> dict:
    sc = cfg["sectors"]
    sectors = list(sector_rows)

    def sig(c, t):
        return (c >= sc["significant_min_df"]) & (c / np.maximum(t, 1) >= sc["significant_min_share"])

    cr, tr = sector_matrix(dt, arts, sector_rows, win.recent12)
    cp, tp = sector_matrix(dt, arts, sector_rows, win.prev12)
    cf, tf = sector_matrix(dt, arts, sector_rows, win.first12)
    sr, sp = sig(cr, tr), sig(cp, tp)
    sf = sig(cf, tf)
    ratio = ((cr + 1) / np.maximum(tr, 1)) / ((cp + 1) / np.maximum(tp, 1))
    # 반기별 유의미 분야 (확산 경로)
    half_sig = {}
    for h in win.halves:
        ms = sorted(arts.loc[arts["half"] == h, "month"].unique())
        c, t = sector_matrix(dt, arts, sector_rows, ms)
        half_sig[h] = sig(c, t)
    return {"sectors": sectors, "recent_df": cr, "recent_tot": tr, "prev_df": cp, "prev_tot": tp,
            "sig_recent": sr, "sig_prev": sp, "sig_first": sf, "ratio": ratio, "half_sig": half_sig}


def spread_path(dif: dict, j: int) -> list[tuple[str, str]]:
    """키워드가 각 분야에서 처음 유의미해진 반기 (확산 순서)."""
    out = []
    for i, sec in enumerate(dif["sectors"]):
        for h, m in dif["half_sig"].items():
            if m[j, i]:
                out.append((sec, h))
                break
    return sorted(out, key=lambda x: x[1])


# ---------------------------------------------------------------- 구조 변화 (15장)

def association(dt: DocTerm, rows: np.ndarray, cols: np.ndarray):
    """cols 키워드와 전체 어휘 사이의 동시등장·코사인 (구간 rows 안)."""
    X = dt.X[rows]
    df = np.asarray(X.sum(axis=0)).ravel()
    co = np.asarray((X[:, cols].T @ X).todense())
    cos = co / np.sqrt(np.outer(np.maximum(df[cols], 1), np.maximum(df, 1)))
    return df, co, cos


def top_assoc(cos_row: np.ndarray, co_row: np.ndarray, df: np.ndarray, self_j: int, n: int, min_co: int, min_df: int,
              exclude: np.ndarray | None = None, n_docs: int | None = None, min_lift: float = 1.2) -> list[int]:
    """코사인 상위 연관어. 어디에나 나오는 단어가 끼지 않도록 우연 기대치보다 min_lift배 이상 함께 나온 것만."""
    ok = (co_row >= min_co) & (df >= min_df)
    if n_docs:
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = co_row * n_docs / (np.maximum(df[self_j], 1) * np.maximum(df, 1))
        ok &= lift >= min_lift
    ok[self_j] = False
    if exclude is not None:
        ok &= ~exclude
    idx = np.flatnonzero(ok)
    return list(idx[np.argsort(-cos_row[idx], kind="stable")][:n])


def structural(dt: DocTerm, early_rows: np.ndarray, recent_rows: np.ndarray, words: list[str],
               early_net: Network, recent_net: Network, cfg: dict, top_n: int = 20) -> pd.DataFrame:
    sc = cfg["status"]
    cols = np.array([dt.index[w] for w in words])
    dfe, coe, cose = association(dt, early_rows, cols)
    dfr, cor, cosr = association(dt, recent_rows, cols)
    cent_e = early_net.centrality.set_index("keyword") if early_net.centrality is not None else None
    cent_r = recent_net.centrality.set_index("keyword") if recent_net.centrality is not None else None
    rows = []
    for i, w in enumerate(words):
        j = cols[i]
        te = top_assoc(cose[i], coe[i], dfe, j, top_n, 3, 10, n_docs=len(early_rows))
        tr = top_assoc(cosr[i], cor[i], dfr, j, top_n, 3, 10, n_docs=len(recent_rows))
        se, sr = set(te), set(tr)
        union = se | sr
        overlap = len(se & sr) / len(union) if union else np.nan
        u = np.array(sorted(union), dtype=int)
        if len(u):
            a, b = cose[i, u], cosr[i, u]
            prof = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        else:
            prof = np.nan
        ctx_e = float(np.mean(cose[i, te[:10]])) if te else 0.0
        ctx_r = float(np.mean(cosr[i, tr[:10]])) if tr else 0.0
        new = [k for k in tr if k not in se]
        lost = [k for k in te if k not in sr]
        # 융합: 새 연관어가 이전에는 다른 군집(또는 네트워크 밖)에 있었고 코사인이 커졌다
        ce = early_net.membership.get(w, -1)
        cross_new = [k for k in new if (early_net.membership.get(dt.vocab[k], -2) != ce or ce == -1)
                     and cosr[i, k] - cose[i, k] >= 0.03 and cor[i, k] >= 5]
        ecl = _cluster_of(early_net, w)
        rcl = _cluster_of(recent_net, w)
        moved = None
        if ecl and rcl:
            s1, s2 = set(ecl["keywords"]), set(rcl["keywords"])
            moved = len(s1 & s2) / len(s1 | s2) < 0.2
        rows.append({
            "keyword": w, "early_df": int(dfe[j]), "recent_df": int(dfr[j]),
            "early_context": round(ctx_e, 3), "recent_context": round(ctx_r, 3),
            "assoc_overlap": overlap, "profile_similarity": prof,
            "early_assoc": [(dt.vocab[k], round(float(cose[i, k]), 3)) for k in te[:12]],
            "recent_assoc": [(dt.vocab[k], round(float(cosr[i, k]), 3)) for k in tr[:12]],
            "new_assoc": [(dt.vocab[k], round(float(cose[i, k]), 3), round(float(cosr[i, k]), 3)) for k in new[:10]],
            "lost_assoc": [(dt.vocab[k], round(float(cose[i, k]), 3), round(float(cosr[i, k]), 3)) for k in lost[:10]],
            "converging_links": [(dt.vocab[k], round(float(cose[i, k]), 3), round(float(cosr[i, k]), 3)) for k in cross_new[:10]],
            "n_converging": len(cross_new),
            "early_cluster": ecl["name_candidate"] if ecl else None,
            "recent_cluster": rcl["name_candidate"] if rcl else None,
            "cluster_moved": moved,
            "early_degree": float(cent_e.loc[w, "degree_centrality"]) if cent_e is not None and w in cent_e.index else None,
            "recent_degree": float(cent_r.loc[w, "degree_centrality"]) if cent_r is not None and w in cent_r.index else None,
        })
    out = pd.DataFrame(rows)
    out["structural_change"] = (out["early_df"] >= sc["structural_min_df"]) & (out["recent_df"] >= sc["structural_min_df"]) & \
                               (out["assoc_overlap"] <= sc["structural_max_overlap"]) & \
                               (out[["early_context", "recent_context"]].min(axis=1) >= sc.get("structural_min_context", 0.08))
    return out


def _cluster_of(net: Network, w: str) -> dict | None:
    cid = net.membership.get(w)
    return net.clusters[cid] if cid is not None else None


# ---------------------------------------------------------------- 융합 (12장)

def sector_convergence(dt: DocTerm, arts: pd.DataFrame, sector_rows: dict[str, np.ndarray],
                       art_company_sectors: list[dict[str, set]], periods: list[str], unit: str,
                       exclude: np.ndarray | None = None) -> dict:
    """분야 간 거리 변화. 분야별 '특징 프로필'(전체 대비 과대표현된 키워드)의 코사인 거리,
    서로 다른 기업을 통해 두 분야에 동시에 걸린 기사 비율."""
    sectors = list(sector_rows)
    S = len(sectors)
    dist = {p: np.full((S, S), np.nan) for p in periods}
    shared = {p: np.zeros((S, S)) for p in periods}
    lab = arts[unit].to_numpy()
    profiles = {}
    for p in periods:
        prow = np.flatnonzero(lab == p)
        all_sh = np.asarray(dt.X[prow].sum(axis=0)).ravel() / max(len(prow), 1)
        vecs = []
        for sec in sectors:
            r = sector_rows[sec]
            r = r[lab[r] == p]
            c = np.asarray(dt.X[r].sum(axis=0)).ravel()
            sh = c / max(len(r), 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                lift = np.where((c >= 3) & (all_sh > 0), np.log(sh / all_sh), 0.0)
            v = np.sqrt(sh) * np.maximum(lift, 0)
            if exclude is not None:
                v[exclude] = 0
            vecs.append(v)
        V = np.stack(vecs)
        profiles[p] = V
        nrm = np.linalg.norm(V, axis=1)
        sim = (V @ V.T) / np.maximum(np.outer(nrm, nrm), 1e-12)
        dist[p] = 1 - sim
        # 교차 기사: 서로 다른 기업을 통해 두 분야에 걸린 기사
        cnt = np.array([len(sector_rows[s][lab[sector_rows[s]] == p]) for s in sectors])
        for r in prow:
            m = art_company_sectors[r]
            if len(m) < 2:
                continue
            secs = sorted({sectors.index(s) for co, ss in m.items() for s in ss})
            for a in secs:
                for b in secs:
                    if a < b and any(sectors[a] in m[c1] and sectors[b] in m[c2] for c1 in m for c2 in m if c1 != c2):
                        shared[p][a, b] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            shared[p] = shared[p] / np.maximum(np.minimum.outer(cnt, cnt), 1)
    return {"sectors": sectors, "periods": periods, "distance": dist, "shared_rate": shared, "profiles": profiles}


def keyword_pair_convergence(dt: DocTerm, early_rows: np.ndarray, recent_rows: np.ndarray, words: list[str],
                             early_net: Network, max_pairs: int = 60, min_early_df: int = 20) -> pd.DataFrame:
    """이전에는 거리가 멀던 키워드 쌍이 최근 가까워진 경우."""
    cols = np.array([dt.index[w] for w in words])

    def cos(rows):
        X = dt.X[rows][:, cols]
        co = np.asarray((X.T @ X).todense())
        d = np.sqrt(np.maximum(np.diag(co), 1))
        return co, co / np.outer(d, d)

    coe, ce = cos(early_rows)
    cor, cr = cos(recent_rows)
    iu, ju = np.triu_indices(len(words), 1)
    gain = cr[iu, ju] - ce[iu, ju]
    dfe = np.diag(coe)
    # 두 키워드 모두 이전에도 있었는데(각 20건 이상) 거의 같이 나오지 않다가 최근 가까워진 쌍
    ok = (ce[iu, ju] < 0.05) & (cr[iu, ju] >= 0.08) & (cor[iu, ju] >= 10) & (gain > 0) & \
         (dfe[iu] >= min_early_df) & (dfe[ju] >= min_early_df)
    iu, ju, gain = iu[ok], ju[ok], gain[ok]
    order = np.argsort(-gain)[: max_pairs * 3]
    rows = []
    for k in order:
        a, b = words[iu[k]], words[ju[k]]
        ca, cb = early_net.membership.get(a, -1), early_net.membership.get(b, -2)
        rows.append({"a": a, "b": b, "early_distance": round(1 - ce[iu[k], ju[k]], 3),
                     "recent_distance": round(1 - cr[iu[k], ju[k]], 3), "early_co": int(coe[iu[k], ju[k]]),
                     "recent_co": int(cor[iu[k], ju[k]]), "different_early_cluster": ca != cb})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["different_early_cluster", "recent_distance"], ascending=[False, True]).head(max_pairs)


def bridge_keywords(net: Network, top: int = 25) -> list[dict]:
    """여러 군집을 잇는 키워드 (이웃이 3개 이상 군집에 걸치고 매개중심성이 높음)."""
    if net.centrality is None:
        return []
    out = []
    for r in net.centrality.sort_values("betweenness", ascending=False).itertuples():
        nb = list(net.graph.neighbors(r.keyword))
        cl = {net.membership.get(n, -1) for n in nb} - {-1}
        if len(cl) >= 3:
            out.append({"keyword": r.keyword, "betweenness": round(r.betweenness, 4), "clusters": len(cl),
                        "cluster_names": [net.clusters[c]["name_candidate"] for c in sorted(cl)][:6]})
        if len(out) >= top:
            break
    return out


# ---------------------------------------------------------------- 시계열 군집 (18장)

def quarterly_networks(dt: DocTerm, arts: pd.DataFrame, win: Windows, cfg: dict, exclude: set[str]) -> list[dict]:
    out = []
    prev = None
    for q in win.quarters:
        rows = win.rows(arts, "quarter", [q])
        net = build_network(dt, rows, q, cfg, exclude=exclude)
        entry = {"period": q, "articles": int(len(rows)), "n_clusters": len(net.clusters),
                 "top_keywords": net.words[:15],
                 "clusters": [{"id": c["id"], "name_candidate": c["name_candidate"], "size": c["size"],
                               "keywords": c["keywords"][:12]} for c in net.clusters[:12]]}
        if prev is not None:
            m = match_clusters(prev, net)
            entry["new_clusters"] = [dict(x, name=net.clusters[x["cluster"]]["name_candidate"]) for x in m if x["status"] == "new"][:8]
            entry["disappeared_clusters"] = [dict(x, name=prev.clusters[x["prev_cluster"]]["name_candidate"]) for x in m if x["status"] == "disappeared"][:8]
        out.append(entry)
        prev = net
    return out
