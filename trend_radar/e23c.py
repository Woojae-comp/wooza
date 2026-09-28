"""E2.3c 사건 이슈 보강 (2026-09-29, 가이드 v1.1 개발용 200건 비교 뒤 진행).

배경: v1.1 합의 판정으로 보면 E2.3b H2는 비콘텐츠 유입을 절반으로 줄이지만, 합의 CONTENT 중 사건 이슈(EVENT_ISSUE) 기사를 EXCLUDE한다.
기존 근거 계열에 '콘텐츠 기업이 당사자인 수사·재판·제재' 근거가 없고, 이런 기사는 주가조작·불성실공시 같은 시장 어휘와 함께 나와
시장 계열 표가 이긴다.

- F_content_event (CONTENT 투표): 콘텐츠 분야 기업명(별칭 포함)과 사건 어휘가 같은 문장(제목 또는 요약 한 문장)에 함께 나온다.
  정치·사회 계열 또는 방송사 출처 계열이 표를 던진 기사는 기권 (방송사 출처·정치 기사의 사건 어휘 유입 차단).
- 하한(floor): 사건 계열이 표를 던졌는데 EXCLUDE로 떨어진 기사는 REVIEW로 올린다. 기사 목록형 제목(주요공시·브리핑 등)은 제외.
  라벨 모델이 시장 표와 사건 표를 가르지 못하는 구조적 한계를 판정 단계에서 막는 장치이며 INCLUDE로 올리지는 않는다.
- 변형: H2 (E2.3b Prior-M 그대로) / H2_event (계열 추가) / H2_event_floor (계열 + 하한).
- 선정: 개발용 200건 LLM 합의 판정(참고용, 정답 아님)과 대표표본 가중 유입률로 비교. 이 200건으로 규칙을 설계했으므로
  통과해도 상태는 'dev_passed_needs_holdout' — 보관 400건 확인 전에는 운영 포인터를 바꾸지 않는다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .e23 import AB, C, VC

EVENT_WORDS = ["수사", "기소", "재판", "압수수색", "제재", "불성실공시", "고발", "무죄", "유죄", "선고", "혐의", "소송", "가처분",
               "과징금", "구속", "영장", "송치", "벌점", "시정명령", "징계", "피소", "횡령", "배임"]
LIST_TITLE = r"주요\s*공시|기업\s*공시|공시\s*\]|브리핑|모닝|주요기사|外|이모저모"
SENT_SPLIT = re.compile(r"\s*\.\.\.\s*|(?<=[다요])\.\s+|[.!?]\s+")
VARIANTS = ("H2", "H2_event", "H2_event_floor")


def sentences(title: str, summary: str) -> list[str]:
    return [title] + [s for s in SENT_SPLIT.split(summary or "") if s]


def content_company_names(article_company: pd.DataFrame, include_sectors: set[str], aliases: dict) -> pd.Series:
    """gid → 콘텐츠 분야(include_sectors) 기업명·별칭 목록 (두 글자 이상)."""
    cc = article_company[article_company["sectors"].apply(lambda s: bool(set(s) & include_sectors))]
    return cc.groupby("gid")["company"].apply(
        lambda s: sorted({n for c in s for n in [c, *(aliases.get(c) or [])] if len(n) >= 2}))


def event_hits(arts: pd.DataFrame, names: pd.Series, words: list[str]) -> np.ndarray:
    """콘텐츠 기업명과 사건 어휘가 같은 문장에 있는가."""
    pat = re.compile("|".join(map(re.escape, words)))
    out = np.zeros(len(arts), bool)
    for i, (g, t, s) in enumerate(zip(arts["gid"], arts["title"].fillna(""), arts["summary"].fillna(""))):
        ns = names.get(g)
        if ns:
            out[i] = any(pat.search(x) and any(n in x for n in ns) for x in sentences(t, s))
    return out


def event_family(p: dict, cfg: dict) -> np.ndarray:
    """F_content_event 발화 여부 (정치·사회·방송사 출처 계열과 충돌하면 기권)."""
    r2 = cfg.get("e2", {})
    names = content_company_names(p["corpus"].article_company, set(cfg["cleaning"].get("include_sectors") or []),
                                  cfg["cleaning"].get("company_aliases") or {})
    hit = event_hits(p["arts"], names, r2.get("event_issue_words") or EVENT_WORDS)
    F = p["F"]
    return hit & (F["F_politics_society"].to_numpy() == AB) & (F["F_broadcaster_source"].to_numpy() == AB)


def apply_floor(dec: np.ndarray, ev: np.ndarray, titles: pd.Series, list_pattern: str = LIST_TITLE) -> np.ndarray:
    is_list = titles.fillna("").str.contains(list_pattern, regex=True).to_numpy()
    return np.where(ev & ~is_list & (dec == "EXCLUDE"), "REVIEW", dec)


def variant_decisions(p: dict, cfg: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """변형별 (판정, P(CONTENT)) 와 사건 계열 발화 여부."""
    from .e23b import PRIORS, decide, fit_h2

    ev = event_family(p, cfg)
    out = {}
    q, _ = fit_h2(p["F"], PRIORS["M"])
    out["H2"] = (decide(q, p["seed_layer"])[0], q[:, C])
    F2 = p["F"].assign(F_content_event=np.where(ev, VC, AB))
    q2, _ = fit_h2(F2, PRIORS["M"])
    dec2 = decide(q2, p["seed_layer"])[0]
    out["H2_event"] = (dec2, q2[:, C])
    out["H2_event_floor"] = (apply_floor(dec2, ev, p["arts"]["title"], cfg.get("e2", {}).get("list_title_pattern", LIST_TITLE)), q2[:, C])
    return out, ev


# ---------------------------------------------------------------- 검토 표본 대조 (LLM 합의 = 참고 판정)

def review_reference(review_dir: Path) -> pd.DataFrame:
    """merged/review_wide.csv 의 모델별 판정 → 참고 판정. internal/v11_reclass.csv (article_id, class)가 있으면
    가이드 v1.1 이전 응답의 CONTENT 판정을 그 클래스로 바꾼다 (실적 공시 → MARKET). 합의 못 하면 DISAGREE."""
    W = pd.read_csv(review_dir / "merged" / "review_wide.csv", dtype=str, keep_default_na=False)
    rc = review_dir / "internal" / "v11_reclass.csv"
    re_map = pd.read_csv(rc, dtype=str).set_index("article_id")["class"].to_dict() if rc.exists() else {}
    models = [c for c in W.columns if c.startswith("class::")]
    fixed = {m: [re_map.get(a, x) if x == "CONTENT" else x for a, x in zip(W["article_id"], W[m])] for m in models}
    F = pd.DataFrame(fixed)
    W["ref"] = np.where(F.nunique(axis=1) == 1, F.iloc[:, 0], "DISAGREE")
    W["any_content"] = (F == "CONTENT").any(axis=1)
    W["w"] = pd.to_numeric(W["weight"], errors="coerce")
    return W


def compare(W: pd.DataFrame, decisions: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict]:
    """후보별 참고 판정 클래스의 유입 (INCLUDE / INCLUDE+REVIEW / 기사 수)과 대표표본 가중 유입률."""
    rows = []
    rep = W["sample_set"] == "representative"
    for name, d in decisions.items():
        d = W["article_id"].map(d)
        row = {"candidate": name}
        for cl in ("CONTENT", "MARKET", "OTHER"):
            m = W["ref"] == cl
            row[f"{cl}_include"], row[f"{cl}_in"], row[f"{cl}_n"] = int((d[m] == "INCLUDE").sum()), int((d[m] != "EXCLUDE").sum()), int(m.sum())
        m = W["any_content"] & (W["ref"] != "CONTENT")
        row["content_disputed_in"], row["content_disputed_n"] = int((d[m] != "EXCLUDE").sum()), int(m.sum())
        r_in = rep & (d != "EXCLUDE")
        row["rep_in_rate_w"] = round(float(W.loc[r_in, "w"].sum() / W.loc[rep, "w"].sum()), 4)
        row["rep_content_share_in_w"] = round(float(W.loc[r_in & (W["ref"] == "CONTENT"), "w"].sum() / max(W.loc[r_in, "w"].sum(), 1e-9)), 4)
        rows.append(row)
    return pd.DataFrame(rows), {"n": len(W), "ref_counts": W["ref"].value_counts().to_dict()}


def gates_v11(row: dict, base: dict, g: dict) -> dict:
    """v1.1 기준 (개발용): 합의 CONTENT를 기준 모델보다 잃지 않고, 합의 비콘텐츠 유입을 충분히 줄인다."""
    nonc = lambda r: r["MARKET_in"] + r["OTHER_in"]
    out = {"gate_content_kept": row["CONTENT_in"] >= base["CONTENT_in"],
           "gate_noncontent_reduced": nonc(row) <= (1 - g.get("noncontent_reduction_min", 0.3)) * nonc(base),
           "gate_rep_content_share": row["rep_content_share_in_w"] >= base["rep_content_share_in_w"]}
    out["pass_all"] = all(out.values())
    return out


def run_e23c(cfg: dict, raw: pd.DataFrame, out_root: Path, review_dir: Path | None = None) -> dict:
    from .e23b import prepare
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .selection import selected_e2

    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e23c_event_issue", cfg, out_root, snap["data_snapshot_id"])
    base_run, base_path, _ = selected_e2(cfg, REGISTRY, out_root, use_candidate=False)
    prev = pd.read_parquet(base_path)

    with run.stage("prepare", rows_in=len(raw)) as st:
        p = prepare(cfg, raw, out_root)
        st["rows_out"] = len(p["arts"])
    with run.stage("label_models") as st:
        var, ev = variant_decisions(p, cfg)
        st["rows_out"] = len(var)

    gid = p["arts"]["gid"].to_numpy()
    dec_df = pd.DataFrame({"gid": gid, "event_family": ev, **{f"decision_{k}": v[0] for k, v in var.items()},
                           **{f"p_content_{k}": np.round(v[1], 5) for k, v in var.items()}})
    dec_df.to_parquet(run.dir / "02c_relevance_variants.parquet", index=False)
    run.artifact(run.dir / "02c_relevance_variants.parquet", "relevance_variants")
    counts = {k: pd.Series(v[0]).value_counts().to_dict() for k, v in var.items()}
    lifted = int((var["H2_event_floor"][0] != var["H2_event"][0]).sum())

    summary = {"baseline_run_id": base_run, "event_family_coverage": round(float(ev.mean()), 4), "event_family_articles": int(ev.sum()),
               "floor_lifted_to_review": lifted, "decision_counts": counts, "status": "no_review_reference"}
    if review_dir is not None and (review_dir / "merged" / "review_wide.csv").exists():
        W = review_reference(review_dir)
        decisions = {"E2.1_operational": pd.Series(prev["decision"].to_numpy(), index=prev["gid"]),
                     **{k: pd.Series(v[0], index=gid) for k, v in var.items()}}
        tab, info = compare(W, decisions)
        base = tab.set_index("candidate").loc["E2.1_operational"].to_dict()
        g = pd.DataFrame([gates_v11(r, base, cfg.get("e23c_gates", {})) for r in tab.to_dict("records")])
        tab = pd.concat([tab, g], axis=1)
        tab.to_csv(run.dir / "e23c_review_comparison.csv", index=False)
        passing = tab[(tab["candidate"] != "E2.1_operational") & tab["pass_all"]]["candidate"].tolist()
        summary.update({"review_dir": str(review_dir), "reference": info, "comparison": tab.to_dict("records"),
                        "passing_variants": passing,
                        "status": "dev_passed_needs_holdout" if passing else "failed"})
    (run.dir / "e23c_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_relevance_v2_e23c", "run_id": run.run_id, "task": "event_issue_family",
        "data_snapshot_id": snap["data_snapshot_id"], "baseline_run_id": base_run, "status": summary["status"],
        "result": {k: summary.get(k) for k in ("event_family_articles", "floor_lifted_to_review", "passing_variants")}})
    run.finish()
    return summary
