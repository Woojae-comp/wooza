"""E5b 주제 트렌드 확정 + E4.3 주제 계보 (v2 설계: 최종 트렌드 유형은 E4 주제와 결합해 확정).

1) 고정 모델 월별 배정: E4.2 기본 모델(LSA100 K120)의 기사 배정을 그대로 쓴다 (월마다 다시 군집하지 않음 → 주제 번호가 흔들리지 않는다).
   주제 × 월 가중 기사 수 W (HIGH·LOW 배정, 판정 가중 INCLUDE 1 / REVIEW 0.5), 월별 가중 기사 합 T. 층화 복구 주제(tr_*)도 같은 방식.
2) 주제 신호: E5a와 같은 규칙(signal_row)을 주제 계열에 적용. 불완전월 제외(직전 완전월 확정), 좌측 절단(수집 첫 3개월 등장 → Emerging 금지).
   민감도: p_rel 가중 계열로 다시 판정 → 다르면 sensitivity_flag.
3) 최종 유형 trend_type:
   - E4.2 잡음 후보(noise_reason_codes 또는 NOISE_CANDIDATE) → Noise (트렌드 목록에서 제외, 값은 보존)
   - 그 밖은 주제 신호 유형. 월 기사가 부족하면(Insufficient) 같은 규칙을 분기 단위로 다시 적용 (signal_resolution=quarter),
     분기로도 부족하면 Low volume (트렌드 판정에서 빼고 목록에는 남김). 데이터가 바뀌어도 사람이 다시 정하지 않는 자동 규칙. 보조 표시: cross_sector(최근 12개월 활성 분야 수·정규화 엔트로피), entity_driven(최근 12개월 상위 기업 점유),
     keyword_support(주제 상위 키워드 중 E5a Emerging·Growing·Event Spike 비율).
4) 계보(E4.3): window_months 창을 step_months 간격으로 밀며, 같은 LSA 공간에서 창 안 기사만 K-means (K = 창에서 가중 min_topic_weight 이상인 고정 주제 수).
   이웃 창 군집을 중심 코사인으로 잇는다. 연결 기준은 서로 최선 짝인 쌍(분명한 이어짐) 코사인 분포의 하위 분위(link_quantile).
   사건: continued(1:1) / split(1:다) / merged(다:1) / new(선행 없음) / ended(후행 없음). 창 군집은 가중 다수 고정 주제에 대응(purity 기록).
   고정 주제별로 최근 창의 사건을 붙인다 → 고정 모델이 최근 구조 변화를 놓치는지 점검.
   안정성: 창 군집을 시드 lineage_seeds개로 다시 만들어 과반 시드에서 나온 사건만 lineage_recent(안정 사건)로 쓴다 (한 시드라도 나온 사건은 lineage_any_seed).
   무작위성 기준: 최근 창을 시드만 바꿔 재군집해 서로 이었을 때 생기는 사건 수 → 실제 창 이동의 사건 수가 이보다 얼마나 많은지(excess_over_null).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .e5 import month_axis, safe_share, signal_row

CONF_OK = ("HIGH", "LOW")


# ---------------------------------------------------------------- 주제 × 월 계열

def topic_months(asg: pd.DataFrame, months: list[str], weight_col: str = "weight_half",
                 topic_col: str = "topic_id") -> tuple[list[str], np.ndarray, np.ndarray]:
    """(주제 목록, 주제 × 월 가중 기사 수, 월별 가중 기사 합). 합계 T는 배정 여부와 무관한 유입 기사 전체."""
    mi = {m: i for i, m in enumerate(months)}
    col = pd.to_datetime(asg["date"]).dt.to_period("M").astype(str).map(mi).to_numpy()
    w = asg[weight_col].to_numpy(float)
    T = np.bincount(col, weights=w, minlength=len(months))
    ok = asg["assignment_confidence"].isin(CONF_OK).to_numpy() & (asg[topic_col].fillna("") != "").to_numpy()
    topics = sorted(asg.loc[ok, topic_col].unique())
    ti = {t: i for i, t in enumerate(topics)}
    W = np.zeros((len(topics), len(months)))
    np.add.at(W, (asg.loc[ok, topic_col].map(ti).to_numpy(), col[ok]), w[ok])
    return topics, W, T


def to_quarters(W: np.ndarray, T: np.ndarray, tc: int) -> tuple[np.ndarray, np.ndarray]:
    """확정월 tc에서 끝나는 3개월 묶음 (앞쪽 모자란 달은 버린다). W: (주제 × 월) 또는 (월,)."""
    n = (tc + 1) // 3
    s0 = tc + 1 - 3 * n
    Wq = W[..., s0: tc + 1].reshape(*W.shape[:-1], n, 3).sum(-1)
    return Wq, T[s0: tc + 1].reshape(n, 3).sum(-1)


def resolve_signal(Wm: np.ndarray, Tm: np.ndarray, tc: int, rules: dict) -> tuple[dict, str]:
    """월 단위로 판정하고, 월 기사가 부족하면(Insufficient) 분기 단위로 다시 판정한다. 분기로도 부족하면 저빈도.
    반환: (신호, 판정 단위 month / quarter / low_volume)."""
    s = signal_row(Wm[: tc + 1], Tm[: tc + 1], rules)
    if s["signal_type"] != "Insufficient":
        return s, "month"
    Wq, Tq = to_quarters(Wm, Tm, tc)
    sq = signal_row(Wq, Tq, rules, per_year=4)
    if sq["signal_type"] != "Insufficient":
        return sq, "quarter"
    return sq, "low_volume"


def norm_title(t: str) -> str:
    import re as _re
    return _re.sub(r"[^0-9A-Za-z가-힣]", "", _re.sub(r"\[[^\]]*\]|\([^)]*\)", "", str(t))).lower()


def evidence_articles(asg: pd.DataFrame, col: str, topic: str, months: list[str], tc: int, summaries: pd.Series,
                      n: int = 5, window: int = 6) -> list[dict]:
    """판정 근거 기사: 판정 기간(확정월까지 최근 window개월) 안의 배정 기사만. HIGH 먼저, 같은 기간이면 중심 유사도 순,
    제목 정규화로 재전송·중복 보도는 한 건만. 부족하면 최근 12개월로 넓힌다 (선정 사유에 기록). 확정월 이후 기사는 쓰지 않는다."""
    mon = asg["date"].dt.to_period("M").astype(str)
    base = asg[(asg[col].fillna("") == topic) & asg["assignment_confidence"].isin(CONF_OK)]
    out, seen = [], set()
    for span, why in ((window, f"판정 기간 최근 {window}개월"), (12, "판정 기간 최근 12개월 (최근 6개월 부족)")):
        win = set(months[max(0, tc - span + 1): tc + 1])
        cand = base[mon.loc[base.index].isin(win)].assign(_h=lambda d: (d["assignment_confidence"] != "HIGH").astype(int)) \
            .sort_values(["_h", "centroid_similarity"], ascending=[True, False])
        for r in cand.itertuples():
            key = norm_title(r.title)
            if key in seen:
                continue
            seen.add(key)
            out.append({"gid": r.gid, "date": str(pd.Timestamp(r.date).date()), "title": r.title,
                        "summary": str(summaries.get(r.gid, ""))[:200], "e2_weight": float(r.weight_half),
                        "e2_decision": "INCLUDE" if r.weight_half >= 1 else "REVIEW", "assignment_confidence": r.assignment_confidence,
                        "centroid_similarity": float(r.centroid_similarity), "selection_reason": f"{why}·배정 신뢰도·중심 유사도 순, 중복 제목 제외"})
            if len(out) >= n:
                return out
    return out


def entropy_norm(x: np.ndarray) -> float:
    x = x[x > 0]
    if len(x) <= 1:
        return 0.0
    p = x / x.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


# ---------------------------------------------------------------- 계보

def windows(months: list[str], tc: int, size: int, step: int) -> list[tuple[int, int]]:
    """[start, end] 월 인덱스 (end 포함), 마지막 창이 확정월 tc에서 끝나도록 뒤에서부터 자른다."""
    out, end = [], tc
    while end - size + 1 >= 0:
        out.append((end - size + 1, end))
        end -= step
    return out[::-1]


def cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    na = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    nb = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return na @ nb.T


def mutual_best(S: np.ndarray) -> np.ndarray:
    """서로가 최선 짝인 쌍의 코사인 (분명한 이어짐의 분포)."""
    ba, bb = S.argmax(1), S.argmax(0)
    return np.array([S[i, ba[i]] for i in range(S.shape[0]) if bb[ba[i]] == i])


def link_windows(Ca: np.ndarray, Cb: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """이웃 창 군집 연결: 코사인 ≥ thr 인 모든 쌍 (한 군집이 둘 이상과 이어지면 분할·병합)."""
    S = cos(Ca, Cb)
    return [(int(i), int(j), float(S[i, j])) for i, j in zip(*np.where(S >= thr))]


def lineage_events(n_a: int, n_b: int, links: list[tuple[int, int, float]]) -> dict[str, list]:
    out_deg, in_deg = collections.Counter(i for i, _, _ in links), collections.Counter(j for _, j, _ in links)
    ev = collections.defaultdict(list)
    for i in range(n_a):
        if out_deg[i] == 0:
            ev["ended"].append(i)
        elif out_deg[i] >= 2:
            ev["split"].append(i)
    for j in range(n_b):
        if in_deg[j] == 0:
            ev["new"].append(j)
        elif in_deg[j] >= 2:
            ev["merged"].append(j)
    ev["continued"] = [(i, j) for i, j, _ in links if out_deg[i] == 1 and in_deg[j] == 1]
    return dict(ev)


def lineage_inputs(emb: dict, asg: pd.DataFrame, months: list[str]) -> dict:
    a = asg.set_index("gid").loc[emb["gid"]]
    mon = pd.to_datetime(a["date"]).dt.to_period("M").astype(str).map({m: i for i, m in enumerate(months)}).to_numpy()
    return {"Z": emb["Z"], "mon": mon, "ok": a["assignment_confidence"].isin(CONF_OK).to_numpy(),
            "w": a["weight_half"].to_numpy(float), "fixed": a["topic_id"].to_numpy()}


def fit_window(li: dict, s: int, e: int, minw: float, seed: int) -> tuple[np.ndarray, list[tuple]]:
    """창 [s, e] 기사만 같은 LSA 공간에서 K-means. K = 창 안 가중 minw 이상 고정 주제 수. 군집별 (가중 다수 고정 주제, 순도, 가중)."""
    from sklearn.cluster import KMeans

    r = np.where(li["ok"] & (li["mon"] >= s) & (li["mon"] <= e))[0]
    w, fixed = li["w"][r], li["fixed"][r]
    K = max(int((pd.Series(w).groupby(fixed).sum() >= minw).sum()), 2)
    km = KMeans(n_clusters=K, random_state=seed, n_init=3).fit(li["Z"][r], sample_weight=w)
    maj = []
    for c in range(K):
        m = km.labels_ == c
        share = pd.Series(w[m]).groupby(fixed[m]).sum().sort_values(ascending=False)
        maj.append((share.index[0], float(share.iloc[0] / share.sum()), float(w[m].sum())))
    return km.cluster_centers_, maj


def step_rows(A: dict, B: dict, thr: float, latest: bool) -> list[dict]:
    ev = lineage_events(len(A["centers"]), len(B["centers"]), link_windows(A["centers"], B["centers"], thr))
    rows = []
    for kind, items in ev.items():
        for it in items:
            i, j = it if kind == "continued" else ((it, None) if kind in ("ended", "split") else (None, it))
            src = A["major"][i] if i is not None else (None, None, None)
            dst = B["major"][j] if j is not None else (None, None, None)
            rows.append({"from_window": A["window"], "to_window": B["window"], "event": kind,
                         "from_cluster": i, "to_cluster": j, "from_fixed_topic": src[0], "from_purity": src[1],
                         "to_fixed_topic": dst[0], "to_purity": dst[1], "weight": dst[2] if dst[2] is not None else src[2],
                         "is_latest_step": latest})
    return rows


def run_lineage(li: dict, months: list[str], tc: int, rules: dict, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], float]:
    size, step = rules.get("window_months", 6), rules.get("step_months", 3)
    minw, q = rules.get("min_topic_weight", 10.0), rules.get("link_quantile", 0.05)
    clus = []
    for s, e in windows(months, tc, size, step):
        C, maj = fit_window(li, s, e, minw, seed)
        clus.append({"window": f"{months[s]}~{months[e]}", "span": (s, e), "centers": C, "major": maj})
    # 연결 기준: 서로 최선 짝인 쌍(분명한 이어짐) 코사인 분포의 하위 분위 — 이어졌다고 하려면 전형적인 이어짐만큼은 닮아야 한다
    mb = np.concatenate([mutual_best(cos(A["centers"], B["centers"])) for A, B in zip(clus, clus[1:])]) if len(clus) > 1 else np.array([1.0])
    thr = float(np.quantile(mb, q))
    rows = []
    for k, (A, B) in enumerate(zip(clus, clus[1:])):
        rows += step_rows(A, B, thr, k == len(clus) - 2)
    purity = pd.DataFrame([{"window": c["window"], "clusters": len(c["major"]),
                            "weighted_purity": round(float(np.average([m[1] for m in c["major"]], weights=[m[2] for m in c["major"]])), 4),
                            "link_threshold": round(thr, 4)} for c in clus])
    return pd.DataFrame(rows), purity, clus, thr


def topic_events(lin: pd.DataFrame, latest_only: bool = True) -> dict[str, set]:
    """고정 주제별 사건 집합 (continued 외)."""
    if lin.empty:
        return {}
    last = lin[(lin["is_latest_step"] if latest_only else True) & (lin["event"] != "continued")]
    out = collections.defaultdict(set)
    for _, r in last.iterrows():
        t = r["to_fixed_topic"] if r["event"] in ("new", "merged") else r["from_fixed_topic"]
        if t:
            out[t].add(r["event"])
    return out


def lineage_recent(lin: pd.DataFrame) -> dict[str, str]:
    """고정 주제별 최근 창 사건 (continued 외)."""
    return {t: ";".join(sorted(v)) for t, v in topic_events(lin).items()}


def null_events(li: dict, span: tuple[int, int], rules: dict, seeds: list[int], thr: float) -> dict[str, float]:
    """같은 창을 시드만 바꿔 다시 군집해 이었을 때 생기는 사건 수 = 군집 무작위성만으로 생기는 사건 (창 쌍 평균)."""
    minw = rules.get("min_topic_weight", 10.0)
    fits = [fit_window(li, span[0], span[1], minw, s)[0] for s in seeds]
    tot, n = collections.Counter(), 0
    for a in range(len(fits)):
        for b in range(a + 1, len(fits)):
            ev = lineage_events(len(fits[a]), len(fits[b]), link_windows(fits[a], fits[b], thr))
            tot.update({k: len(v) for k, v in ev.items()})
            n += 1
    return {k: tot[k] / max(n, 1) for k in ("continued", "split", "merged", "new", "ended")}


def lineage_stability(li: dict, months: list[str], tc: int, rules: dict, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """시드별 계보를 다시 만들고 (1) 주제별 최근 사건이 몇 개 시드에서 나오는지, (2) 같은 창 재군집만으로 생기는 사건 수(무작위성 기준)를 비교.
    안정 사건 = 과반 시드에서 나온 사건."""
    per_seed, step_counts, thr0, last_span = [], [], None, None
    for s in seeds:
        lin, _, clus, thr = run_lineage(li, months, tc, rules, s)
        thr0 = thr if thr0 is None else thr0
        last_span = clus[-1]["span"]
        per_seed.append(topic_events(lin))
        c = lin[lin["is_latest_step"]]["event"].value_counts()
        step_counts.append({"seed": s, "link_threshold": round(thr, 4), **{k: int(c.get(k, 0)) for k in ("continued", "split", "merged", "new", "ended")}})
    topics = sorted(set().union(*[set(p) for p in per_seed]))
    rows = []
    for t in topics:
        cnt = collections.Counter(ev for p in per_seed for ev in p.get(t, set()))
        stable = sorted(k for k, v in cnt.items() if v > len(seeds) / 2)
        rows.append({"topic_id": t, "seeds": len(seeds), **{f"n_{k}": cnt.get(k, 0) for k in ("split", "merged", "new", "ended")},
                     "stable_events": ";".join(stable), "any_seed_events": ";".join(sorted(cnt))})
    null = null_events(li, last_span, rules, seeds, thr0)
    counts = pd.DataFrame(step_counts)
    obs = counts[["split", "merged", "new", "ended"]].mean().to_dict()
    summary = {"seeds": seeds, "latest_step_mean": {k: round(v, 1) for k, v in obs.items()},
               "same_window_null_mean": {k: round(v, 1) for k, v in null.items()},
               "excess_over_null": {k: round(obs[k] - null.get(k, 0), 1) for k in obs},
               "topics_with_stable_event": int(sum(bool(r["stable_events"]) for r in rows)),
               "topics_with_any_event": len(rows)}
    return pd.DataFrame(rows), counts, summary


# ---------------------------------------------------------------- 실행

def run_e5b(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot

    rules = {**cfg.get("e5", {}), **cfg.get("e5b", {})}
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e5b_topic_trend", cfg, out_root, snap["data_snapshot_id"])
    recs = [json.loads(l) for l in (REGISTRY / "run_registry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def latest(exp: str, need: str) -> str:
        def half_mode(rid: str) -> bool:     # 가중 민감도 실행(half_x_pcontent)은 제외
            sp = next((out_root / "runs" / rid).glob("*_summary.json"), None)
            return sp is None or json.loads(sp.read_text(encoding="utf-8")).get("relevance_weighting", "half") == "half"
        ok = [r["run_id"] for r in recs if r.get("kind") == exp and r.get("status") == "SUCCESS"
              and (out_root / "runs" / r["run_id"] / need).exists() and half_mode(r["run_id"])]
        if not ok:
            raise FileNotFoundError(f"{exp} 실행 결과({need})가 없다")
        return ok[-1]

    e42_run = cfg.get("e5b", {}).get("e42_run") or latest("e42_topic_final", "07c_lsa_embedding.npz")
    e5_run = cfg.get("e5b", {}).get("e5a_run") or latest("e5a_keyword_signal", "06a_keyword_signal.csv")
    d42 = out_root / "runs" / e42_run

    with run.stage("load") as st:
        asg = pd.read_parquet(d42 / "07c_article_assignment.parquet")
        reg = pd.read_csv(d42 / "07c_topic_registry.csv")
        emb = dict(np.load(d42 / "07c_lsa_embedding.npz", allow_pickle=False))
        sig = pd.read_csv(out_root / "runs" / e5_run / "06a_keyword_signal.csv")
        corpus = build_corpus(raw, cfg)
        st["rows_out"] = len(asg)

    with run.stage("series") as st:
        asg["date"] = pd.to_datetime(asg["date"])
        months = month_axis(corpus.articles["date"])
        last_day = corpus.articles["date"].max()
        partial = bool(last_day.day < last_day.days_in_month)
        tc = len(months) - 1 - int(partial)
        topics, W, T = topic_months(asg, months)
        _, Ws, Ts = topic_months(asg, months, "p_rel")
        r_asg = asg[asg["rescue_topic_id"].fillna("") != ""]
        if len(r_asg):
            rt, Wr, _ = topic_months(r_asg, months, topic_col="rescue_topic_id")
            _, Wrs, _ = topic_months(r_asg, months, "p_rel", "rescue_topic_id")
            topics, W, Ws = topics + rt, np.vstack([W, Wr]), np.vstack([Ws, Wrs])
        st["rows_out"] = len(topics)

    with run.stage("lineage") as st:
        li = lineage_inputs(emb, asg, months)
        lin, purity, _, _ = run_lineage(li, months, tc, rules, seed=0)
        seeds = list(range(rules.get("lineage_seeds", 5)))
        stab, seed_counts, stab_sum = lineage_stability(li, months, tc, rules, seeds)
        stab.to_csv(run.dir / "07d_lineage_stability.csv", index=False, encoding="utf-8-sig")
        seed_counts.to_csv(run.dir / "07d_lineage_seed_counts.csv", index=False, encoding="utf-8-sig")
        recent = stab.set_index("topic_id")["stable_events"].to_dict() if len(stab) else {}
        recent_any = stab.set_index("topic_id")["any_seed_events"].to_dict() if len(stab) else {}
        lin.to_csv(run.dir / "07d_topic_lineage.csv", index=False, encoding="utf-8-sig")
        purity.to_csv(run.dir / "07d_window_purity.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07d_topic_lineage.csv", "topic_lineage", rows=len(lin))
        st["rows_out"] = len(lin)

    with run.stage("signals") as st:
        sec = corpus.article_sector[corpus.article_sector["sector"] != cfg["cleaning"].get("related_label", "연관산업")]
        sec_of = sec.groupby("gid")["sector"].agg(list)
        comp_of = corpus.article_company.groupby("gid")["company"].agg(list)
        recent12 = asg["date"].dt.to_period("M").astype(str).isin(months[max(0, tc - 11): tc + 1]).to_numpy()
        okc = asg["assignment_confidence"].isin(CONF_OK).to_numpy()
        # 주제 키워드 지지: 트렌드 신호로 쓸 수 있는 키워드(개념·작품)만 센다 (기업명·일반어·시세·서식어 제외)
        sig_ok = sig[sig["trend_eligible"]] if "trend_eligible" in sig.columns else sig
        kw_type = sig_ok.set_index("keyword")["signal_type"].to_dict()
        reg_i = reg.set_index("topic_id")
        rows = []
        for i, t in enumerate(topics):
            col = "rescue_topic_id" if t.startswith("tr_") else "topic_id"
            m = okc & recent12 & (asg[col].fillna("") == t).to_numpy()
            ww = asg.loc[m, "weight_half"].to_numpy()
            gids = asg.loc[m, "gid"].to_numpy()
            sec_w = collections.Counter()
            co_w = collections.Counter()
            for g, x in zip(gids, ww):
                for s in sec_of.get(g, []):
                    sec_w[s] += x
                for c in set(comp_of.get(g, [])):
                    co_w[c] += x
            sw_ = np.array(list(sec_w.values()), float)
            active = int((sw_ >= rules.get("sector_min_df", 3)).sum())
            ent = entropy_norm(sw_)
            top_co, top_n = co_w.most_common(1)[0] if co_w else ("", 0.0)
            s_conf, resolution = resolve_signal(W[i], T, tc, rules)
            # Stable은 잔여 범주: 6개월 비가 성장·쇠퇴 기준을 넘었지만 다른 조건(Robust Z·지속)을 못 넘은 '방향 혼재'와 '변화 작음'을 구분
            r6 = s_conf["ratio_6m"]
            note = ("방향 혼재" if (r6 >= rules.get("growing_ratio", 1.3) or r6 <= rules.get("declining_ratio", 0.7)) else "변화 작음") \
                if s_conf["signal_type"] == "Stable" else ""
            s_prov = signal_row(W[i], T, rules)
            s_soft, _ = resolve_signal(Ws[i], Ts, tc, rules)
            r = reg_i.loc[t] if t in reg_i.index else pd.Series(dtype=object)
            noise = bool(str(r.get("noise_reason_codes", "") or "").strip() and str(r.get("noise_reason_codes")) != "nan") \
                or r.get("topic_type") == "NOISE_CANDIDATE"
            kws = [k.strip() for k in str(r.get("top_keywords", "") or "").split(",") if k.strip()][:10]
            known = [kw_type[k] for k in kws if k in kw_type]
            support = round(sum(x in ("Emerging", "Growing", "Event Spike") for x in known) / len(known), 3) if known else None
            rows.append({
                "topic_id": t, "topic_type_e42": r.get("topic_type"), "top_keywords": ", ".join(kws),
                "trend_type": "Noise" if noise else ("Low volume" if resolution == "low_volume" else s_conf["signal_type"]),
                "signal_resolution": resolution, "trend_note": note,
                "signal_type": s_conf["signal_type"], "signal_type_provisional": s_prov["signal_type"],
                "signal_type_soft": s_soft["signal_type"], "sensitivity_flag": int(s_conf["signal_type"] != s_soft["signal_type"]),
                "noise_excluded": int(noise), "noise_reason_codes": r.get("noise_reason_codes"),
                "noise_basis": ";".join(x for x in [("E4.2 유형 NOISE_CANDIDATE" if r.get("topic_type") == "NOISE_CANDIDATE" else ""),
                                                  (f"잡음 사유 {r.get('noise_reason_codes')}" if str(r.get("noise_reason_codes") or "").strip()
                                                   and str(r.get("noise_reason_codes")) != "nan" else "")] if x),
                "left_censored": s_conf["left_censored"], "signal_asof": months[tc], "provisional_month": months[-1] if partial else "",
                "growth_3m": round(s_conf["growth_3m"], 3), "ratio_6m": round(s_conf["ratio_6m"], 3), "robust_z": round(s_conf["robust_z"], 3),
                "persistence_12m": round(s_conf["persistence_12m"], 3), "burst_months_12m": s_conf["burst_months_12m"],
                "peak_dominance_12m": round(s_conf["peak_dominance_12m"], 3),
                "weighted_articles_12m": round(float(W[i, max(0, tc - 11): tc + 1].sum()), 1),
                "weighted_articles_last": round(float(W[i, tc]), 1),
                "share_per_1000_last": round(float(safe_share(W[i, tc:tc + 1], T[tc:tc + 1])[0]), 3),
                "sectors_active_12m": active, "sector_entropy_12m": round(ent, 3),
                "cross_sector": bool(active >= rules.get("cross_min_sectors", 3) and ent >= rules.get("cross_min_entropy", 0.6)),
                "top_company_12m": top_co, "top_company_share_12m": round(top_n / max(ww.sum(), 1e-9), 3),
                "entity_driven": bool(top_n / max(ww.sum(), 1e-9) >= rules.get("entity_driven_share", 0.5)),
                "keyword_support": support, "lineage_recent": recent.get(t, ""), "lineage_any_seed": recent_any.get(t, ""),
            })
        out = pd.DataFrame(rows).sort_values(["noise_excluded", "trend_type", "weighted_articles_12m"], ascending=[True, True, False])
        summaries = corpus.articles.set_index("gid")["summary"]
        ev = [dict(topic_id=t, rank=j + 1, **e) for t in topics
              for j, e in enumerate(evidence_articles(asg, "rescue_topic_id" if t.startswith("tr_") else "topic_id", t, months, tc, summaries))]
        pd.DataFrame(ev).to_csv(run.dir / "06b_topic_evidence.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "06b_topic_evidence.csv", "topic_evidence", rows=len(ev))
        out.to_csv(run.dir / "06b_topic_trend.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "06b_topic_trend.csv", "topic_trend", rows=len(out))
        mon = pd.DataFrame(W, index=topics, columns=months).rename_axis("topic_id").reset_index() \
            .melt(id_vars="topic_id", var_name="month", value_name="weighted_articles")
        mon["share_per_1000"] = np.round(safe_share(mon["weighted_articles"].to_numpy(),
                                                    mon["month"].map(dict(zip(months, T))).to_numpy()), 4)
        mon["is_partial_month"] = (mon["month"] == months[-1]) & partial
        # 재현용: 원 기사 수(HIGH·LOW 배정), 가중 기사 수 W, 분모 N(같은 달 유입 기사 전체 가중 — 미배정·거부 포함), 판정 포함 여부
        _, R, _ = topic_months(asg.assign(_one=1.0), months, "_one")
        if len(r_asg):
            _, Rr, _ = topic_months(r_asg.assign(_one=1.0), months, "_one", "rescue_topic_id")
            R = np.vstack([R, Rr])
        mon["raw_articles"] = pd.DataFrame(R, index=topics, columns=months).stack().reindex(
            pd.MultiIndex.from_frame(mon[["topic_id", "month"]])).to_numpy()
        mon["month_total_weighted"] = mon["month"].map(dict(zip(months, T)))
        mon["in_signal_window"] = mon["month"].map({m: i <= tc for i, m in enumerate(months)})
        mon.to_parquet(run.dir / "06b_topic_trend_monthly.parquet", index=False)
        st["rows_out"] = len(out)

    keep = out[out["noise_excluded"] == 0]
    summary = {"e42_run": e42_run, "e5a_run": e5_run, "topics": len(out), "noise_excluded": int(out["noise_excluded"].sum()),
               "signal_asof": months[tc], "partial_month": months[-1] if partial else None,
               "trend_type_counts": keep["trend_type"].value_counts().to_dict(),
               "signal_resolution_counts": keep["signal_resolution"].value_counts().to_dict(),
               "sensitivity_flag": int(keep["sensitivity_flag"].sum()), "cross_sector": int(keep["cross_sector"].sum()),
               "entity_driven": int(keep["entity_driven"].sum()),
               "lineage_events_latest": lin[lin["is_latest_step"]]["event"].value_counts().to_dict() if len(lin) else {},
               "lineage_stability": stab_sum,
               "window_purity": purity.to_dict("records")}
    (run.dir / "e5b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_trend_v2_e5b", "run_id": run.run_id, "task": "topic_trend", "data_snapshot_id": snap["data_snapshot_id"],
        "e42_run": e42_run, "e5a_run": e5_run, "status": "SUCCESS", "result": summary["trend_type_counts"]})
    run.finish()
    return summary


# ---------------------------------------------------------------- 화면용

def latest_e5b_run(out_root: Path) -> str | None:
    from .runlog import REGISTRY

    p = REGISTRY / "run_registry.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r["run_id"] for r in recs if r.get("kind") == "e5b_topic_trend" and r.get("status") == "SUCCESS"
          and (out_root / "runs" / r["run_id"] / "06b_topic_trend.csv").exists()]
    return ok[-1] if ok else None


def topic_payload(out_root: Path, run_id: str | None = None, n_articles: int = 4) -> dict | None:
    """레이더 화면 '주제 트렌드' 탭 자료. E5b 실행이 없으면 None."""
    run_id = run_id or latest_e5b_run(out_root)
    if not run_id:
        return None
    d = out_root / "runs" / run_id
    summ = json.loads((d / "e5b_summary.json").read_text(encoding="utf-8"))
    t = pd.read_csv(d / "06b_topic_trend.csv", keep_default_na=False)
    mon = pd.read_parquet(d / "06b_topic_trend_monthly.parquet")
    months = sorted(mon["month"].unique())
    series = mon.pivot(index="topic_id", columns="month", values="share_per_1000").reindex(columns=months).fillna(0)
    reg = pd.read_csv(out_root / "runs" / summ["e42_run"] / "07c_topic_registry.csv", keep_default_na=False).set_index("topic_id")
    asg = pd.read_parquet(out_root / "runs" / summ["e42_run"] / "07c_article_assignment.parquet", columns=["gid", "date"]).set_index("gid")

    def reps(tid: str) -> list[dict]:
        raw = str(reg["representative_articles"].get(tid, "")) if tid in reg.index else ""
        out = []
        for part in [p.strip() for p in raw.split("||") if p.strip()][:n_articles]:
            gid, _, title = part.partition(" ")
            dt = asg["date"].get(gid)
            out.append({"t": title, "d": str(pd.Timestamp(dt).date()) if dt is not None else ""})
        return out

    evp = d / "06b_topic_evidence.csv"
    ev = pd.read_csv(evp, keep_default_na=False) if evp.exists() else pd.DataFrame(columns=["topic_id"])
    ev_by = {t: g.sort_values("rank").to_dict("records") for t, g in ev.groupby("topic_id")}
    asof_i = months.index(summ["signal_asof"]) if summ["signal_asof"] in months else len(months) - 1
    periods = {"recent6": [months[max(0, asof_i - 5)], months[asof_i]], "base6": [months[max(0, asof_i - 11)], months[max(0, asof_i - 6)]]}

    def stable_lineage(v: str) -> list[str]:
        # split·merged는 군집 무작위성과 구별되지 않아 화면에 쓰지 않는다 (E4.3 안정성 결과)
        return [e for e in str(v).split(";") if e in ("new", "ended")]

    num = lambda x: None if x in ("", None) else float(x)
    rows = []
    for r in t.to_dict("records"):
        tid = r["topic_id"]
        rows.append({"id": tid, "type": r["trend_type"], "res": r["signal_resolution"], "kws": r["top_keywords"].split(", ")[:10],
                     "e42": r["topic_type_e42"], "noise": int(r["noise_excluded"]), "noise_codes": r["noise_reason_codes"],
                     "ratio6": num(r["ratio_6m"]), "rz": num(r["robust_z"]), "pers12": num(r["persistence_12m"]),
                     "w12": num(r["weighted_articles_12m"]), "wlast": num(r["weighted_articles_last"]),
                     "share_last": num(r["share_per_1000_last"]), "sectors": int(r["sectors_active_12m"] or 0),
                     "cross": str(r["cross_sector"]) == "True", "entity": str(r["entity_driven"]) == "True",
                     "company": r["top_company_12m"], "company_share": num(r["top_company_share_12m"]),
                     "support": num(r["keyword_support"]), "sens": int(r["sensitivity_flag"]), "soft": r["signal_type_soft"],
                     "prov": r["signal_type_provisional"], "lineage": stable_lineage(r.get("lineage_recent", "")),
                     "lineage_all": [e for e in str(r.get("lineage_recent", "")).split(";") if e],
                     "note": r.get("trend_note", ""), "noise_basis": r.get("noise_basis", ""),
                     "evidence": [{"t": e["title"], "d": e["date"], "s": e["summary"], "dec": e["e2_decision"], "conf": e["assignment_confidence"]}
                                  for e in ev_by.get(tid, [])],
                     "series": [round(float(x), 3) for x in series.loc[tid].tolist()] if tid in series.index else [],
                     "arts": reps(tid)})
    return {"run_id": run_id, "e42_run": summ["e42_run"], "asof": summ["signal_asof"], "partial": summ.get("partial_month"),
            "months": months, "periods": periods, "counts": summ.get("trend_type_counts", {}), "resolution": summ.get("signal_resolution_counts", {}),
            "lineage": summ.get("lineage_stability", {}), "rows": rows}
