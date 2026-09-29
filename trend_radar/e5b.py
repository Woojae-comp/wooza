"""E5b 주제 트렌드 확정 + E4.3 주제 계보 (v2 설계: 최종 트렌드 유형은 E4 주제와 결합해 확정).

1) 고정 모델 월별 배정: E4.2 기본 모델(LSA100 K120)의 기사 배정을 그대로 쓴다 (월마다 다시 군집하지 않음 → 주제 번호가 흔들리지 않는다).
   주제 × 월 가중 기사 수 W (HIGH·LOW 배정, 판정 가중 INCLUDE 1 / REVIEW 0.5), 월별 가중 기사 합 T. 층화 복구 주제(tr_*)도 같은 방식.
2) 주제 신호: E5a와 같은 규칙(signal_row)을 주제 계열에 적용. 불완전월 제외(직전 완전월 확정), 좌측 절단(수집 첫 3개월 등장 → Emerging 금지).
   민감도: p_rel 가중 계열로 다시 판정 → 다르면 sensitivity_flag.
3) 최종 유형 trend_type:
   - E4.2 잡음 후보(noise_reason_codes 또는 NOISE_CANDIDATE) → Noise (트렌드 목록에서 제외, 값은 보존)
   - 그 밖은 주제 신호 유형. 보조 표시: cross_sector(최근 12개월 활성 분야 수·정규화 엔트로피), entity_driven(최근 12개월 상위 기업 점유),
     keyword_support(주제 상위 키워드 중 E5a Emerging·Growing·Event Spike 비율).
4) 계보(E4.3): window_months 창을 step_months 간격으로 밀며, 같은 LSA 공간에서 창 안 기사만 K-means (K = 창에서 가중 min_topic_weight 이상인 고정 주제 수).
   이웃 창 군집을 중심 코사인으로 잇는다. 연결 기준은 사람 값이 아니라 이웃 창 전체 쌍 코사인 분포의 상위 분위(null_quantile).
   사건: continued(1:1) / split(1:다) / merged(다:1) / new(선행 없음) / ended(후행 없음). 창 군집은 가중 다수 고정 주제에 대응(purity 기록).
   고정 주제별로 최근 창의 사건을 lineage_recent로 붙인다 → 고정 모델이 최근 구조 변화를 놓치는지 점검.
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


def link_windows(Ca: np.ndarray, Cb: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """이웃 창 군집 연결: 코사인 ≥ thr 이면서 한쪽의 최선 짝인 쌍."""
    na = Ca / np.maximum(np.linalg.norm(Ca, axis=1, keepdims=True), 1e-12)
    nb = Cb / np.maximum(np.linalg.norm(Cb, axis=1, keepdims=True), 1e-12)
    S = na @ nb.T
    ba, bb = S.argmax(1), S.argmax(0)
    links = {(i, int(ba[i])) for i in range(len(Ca)) if S[i, ba[i]] >= thr}
    links |= {(int(bb[j]), j) for j in range(len(Cb)) if S[bb[j], j] >= thr}
    return sorted((i, j, float(S[i, j])) for i, j in links)


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


def run_lineage(emb: dict, asg: pd.DataFrame, months: list[str], tc: int, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans

    size, step = rules.get("window_months", 6), rules.get("step_months", 3)
    minw, q = rules.get("min_topic_weight", 10.0), rules.get("null_quantile", 0.95)
    a = asg.set_index("gid").loc[emb["gid"]]
    mon = pd.to_datetime(a["date"]).dt.to_period("M").astype(str).map({m: i for i, m in enumerate(months)}).to_numpy()
    ok = a["assignment_confidence"].isin(CONF_OK).to_numpy()
    w, fixed = a["weight_half"].to_numpy(float), a["topic_id"].to_numpy()
    Z = emb["Z"]
    wins = windows(months, tc, size, step)
    clus = []
    for k, (s, e) in enumerate(wins):
        r = np.where(ok & (mon >= s) & (mon <= e))[0]
        tw = pd.Series(w[r]).groupby(fixed[r]).sum()
        K = max(int((tw >= minw).sum()), 2)
        km = KMeans(n_clusters=K, random_state=0, n_init=3).fit(Z[r], sample_weight=w[r])
        maj = []
        for c in range(K):
            rc = r[km.labels_ == c]
            share = pd.Series(w[rc]).groupby(fixed[rc]).sum().sort_values(ascending=False)
            maj.append((share.index[0], float(share.iloc[0] / share.sum()), float(w[rc].sum())))
        clus.append({"window": f"{months[s]}~{months[e]}", "centers": km.cluster_centers_, "major": maj})
    # 연결 기준: 이웃 창 전체 쌍 코사인 분포의 상위 분위 (대부분 쌍은 무관하므로 null 분포로 본다)
    sims = []
    for A, B in zip(clus, clus[1:]):
        na = A["centers"] / np.linalg.norm(A["centers"], axis=1, keepdims=True)
        nb = B["centers"] / np.linalg.norm(B["centers"], axis=1, keepdims=True)
        sims.append((na @ nb.T).ravel())
    thr = float(np.quantile(np.concatenate(sims), q)) if sims else 1.0
    rows = []
    for k, (A, B) in enumerate(zip(clus, clus[1:])):
        links = link_windows(A["centers"], B["centers"], thr)
        ev = lineage_events(len(A["centers"]), len(B["centers"]), links)
        for kind, items in ev.items():
            for it in items:
                i, j = it if kind == "continued" else ((it, None) if kind in ("ended", "split") else (None, it))
                src = A["major"][i] if i is not None else (None, None, None)
                dst = B["major"][j] if j is not None else (None, None, None)
                rows.append({"from_window": A["window"], "to_window": B["window"], "event": kind,
                             "from_cluster": i, "to_cluster": j, "from_fixed_topic": src[0], "from_purity": src[1],
                             "to_fixed_topic": dst[0], "to_purity": dst[1], "weight": dst[2] if dst[2] is not None else src[2],
                             "is_latest_step": k == len(clus) - 2})
    lin = pd.DataFrame(rows)
    purity = pd.DataFrame([{"window": c["window"], "clusters": len(c["major"]),
                            "weighted_purity": round(float(np.average([m[1] for m in c["major"]], weights=[m[2] for m in c["major"]])), 4),
                            "link_threshold": round(thr, 4)} for c in clus])
    return lin, purity


def lineage_recent(lin: pd.DataFrame) -> dict[str, str]:
    """고정 주제별 최근 창 사건 (continued 외)."""
    if lin.empty:
        return {}
    last = lin[lin["is_latest_step"] & (lin["event"] != "continued")]
    out = collections.defaultdict(set)
    for _, r in last.iterrows():
        t = r["to_fixed_topic"] if r["event"] in ("new", "merged") else r["from_fixed_topic"]
        if t:
            out[t].add(r["event"])
    return {t: ";".join(sorted(v)) for t, v in out.items()}


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
        lin, purity = run_lineage(emb, asg, months, tc, rules)
        recent = lineage_recent(lin)
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
        kw_type = sig.set_index("keyword")["signal_type"].to_dict()
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
            s_conf = signal_row(W[i, : tc + 1], T[: tc + 1], rules)
            s_prov = signal_row(W[i], T, rules)
            s_soft = signal_row(Ws[i, : tc + 1], Ts[: tc + 1], rules)
            r = reg_i.loc[t] if t in reg_i.index else pd.Series(dtype=object)
            noise = bool(str(r.get("noise_reason_codes", "") or "").strip() and str(r.get("noise_reason_codes")) != "nan") \
                or r.get("topic_type") == "NOISE_CANDIDATE"
            kws = [k.strip() for k in str(r.get("top_keywords", "") or "").split(",") if k.strip()][:10]
            known = [kw_type[k] for k in kws if k in kw_type]
            support = round(sum(x in ("Emerging", "Growing", "Event Spike") for x in known) / len(known), 3) if known else None
            rows.append({
                "topic_id": t, "topic_type_e42": r.get("topic_type"), "top_keywords": ", ".join(kws),
                "trend_type": "Noise" if noise else s_conf["signal_type"],
                "signal_type": s_conf["signal_type"], "signal_type_provisional": s_prov["signal_type"],
                "signal_type_soft": s_soft["signal_type"], "sensitivity_flag": int(s_conf["signal_type"] != s_soft["signal_type"]),
                "noise_excluded": int(noise), "noise_reason_codes": r.get("noise_reason_codes"),
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
                "keyword_support": support, "lineage_recent": recent.get(t, ""),
            })
        out = pd.DataFrame(rows).sort_values(["noise_excluded", "trend_type", "weighted_articles_12m"], ascending=[True, True, False])
        out.to_csv(run.dir / "06b_topic_trend.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "06b_topic_trend.csv", "topic_trend", rows=len(out))
        mon = pd.DataFrame(W, index=topics, columns=months).rename_axis("topic_id").reset_index() \
            .melt(id_vars="topic_id", var_name="month", value_name="weighted_articles")
        mon["share_per_1000"] = np.round(safe_share(mon["weighted_articles"].to_numpy(),
                                                    mon["month"].map(dict(zip(months, T))).to_numpy()), 4)
        mon["is_partial_month"] = (mon["month"] == months[-1]) & partial
        mon.to_parquet(run.dir / "06b_topic_trend_monthly.parquet", index=False)
        st["rows_out"] = len(out)

    keep = out[out["noise_excluded"] == 0]
    summary = {"e42_run": e42_run, "e5a_run": e5_run, "topics": len(out), "noise_excluded": int(out["noise_excluded"].sum()),
               "signal_asof": months[tc], "partial_month": months[-1] if partial else None,
               "trend_type_counts": keep["trend_type"].value_counts().to_dict(),
               "sensitivity_flag": int(keep["sensitivity_flag"].sum()), "cross_sector": int(keep["cross_sector"].sum()),
               "entity_driven": int(keep["entity_driven"].sum()),
               "lineage_events_latest": lin[lin["is_latest_step"]]["event"].value_counts().to_dict() if len(lin) else {},
               "window_purity": purity.to_dict("records")}
    (run.dir / "e5b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_trend_v2_e5b", "run_id": run.run_id, "task": "topic_trend", "data_snapshot_id": snap["data_snapshot_id"],
        "e42_run": e42_run, "e5a_run": e5_run, "status": "SUCCESS", "result": summary["trend_type_counts"]})
    run.finish()
    return summary
