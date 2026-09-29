"""대시보드 평가 묶음 (`python -m trend_radar export`) — 화면의 판정 결과를 TSV로 내보내 사람·LLM이 평가할 수 있게.

- 01_topic_trends.tsv: E5b 주제 트렌드 (주제당 한 행, 대표 기사 제목 포함) + 평가 열
- 02_topic_monthly.tsv: 주제 × 월 기사 비중(1,000건당) — 판정 근거 확인용
- 03_keyword_signals.tsv: E5a 키워드 신호 (Insufficient 제외) + 평가 열
- 04_radar_lists.tsv: 키워드 레이더 화면의 목록(층·목록별 키워드와 근거 지표) + 평가 열
- README.md: 열 설명과 평가 방법
평가 열은 비워 둔다. LLM 판정은 참고 의견이며 정답·학습 라벨로 쓰지 않는다.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

TT_KO = {"Emerging": "부상", "Growing": "성장", "Established": "정착", "Stable": "유지", "Declining": "쇠퇴",
         "Event Spike": "급등", "Low volume": "저빈도", "Noise": "잡음 후보", "Insufficient": "자료 부족"}
RES_KO = {"month": "월", "quarter": "분기", "low_volume": "저빈도"}
LIST_KO = {"rising": "A 확대되는 이슈", "established": "A 지속 핵심", "cross_sector": "C 분야 간 공통", "emerging": "E 부상",
           "growing": "E 성장", "declining": "E 쇠퇴", "event": "E 이벤트성", "spreading": "D 확산", "converging": "D 융합",
           "structural": "F 구조 변화"}
EVAL_TOPIC = ["eval_topic_coherent", "eval_is_content_industry", "eval_trend_type_ok", "eval_suggested_type", "eval_comment"]
EVAL_KW = ["eval_meaningful", "eval_is_content_industry", "eval_signal_ok", "eval_comment"]

README = """# Trend Radar 평가 묶음

생성: {created} · 스냅샷 {snapshot} · 확정 기준월 {asof}{partial}
출처: E5b {e5b} · E4.2 {e42} · E5a {e5a} · 키워드 레이더 out/{{content,market}}/radar.json

## 파일
| 파일 | 행 | 내용 |
|---|---|---|
| 01_topic_trends.tsv | {n_topics} | 주제 트렌드 (화면 '주제 트렌드' 탭). 잡음 후보도 포함 (trend_type=잡음 후보) |
| 02_topic_monthly.tsv | {n_topics} | 주제별 월간 기사 비중 (유입 기사 1,000건당). 마지막 달이 불완전월이면 판정에서 제외됨 |
| 03_keyword_signals.tsv | {n_kw} | 키워드 시계열 신호 (자료 부족 제외) |
| 04_radar_lists.tsv | {n_list} | 키워드 레이더 화면 목록 (층: content=콘텐츠·사업, market=자본시장) |
| spec/SPEC_CURRENT.md | | 현행 명세 (운영 방식·규칙·검증 체계·알려진 한계) |
| spec/TREND_RADAR_V2.md | | 실험 기록 (시도·실패·수치 이력) |
| spec/trend_radar.yaml | | 설정값 전체 |
| spec/annotation_guide.md | | 콘텐츠 관련성 판정 가이드 v1.1 (LLM 검토 기준) |
| spec/decision_log.jsonl | | 결정 기록 (사용자·보조 연구자·파이프라인) |
| spec/improvement_prompt.md | | 개선안 검토 요청문 (이 묶음과 함께 LLM에 붙여 넣기) |

## 평가 방법 (eval_* 열에 기입, 비워 두면 미평가)
- eval_topic_coherent: 키워드·대표 기사가 하나의 이야기로 묶이는가 (Y / N / ?)
- eval_is_content_industry: 콘텐츠산업(작품·유통·정책·사건) 이야기인가, 주가·실적 등 자본시장 이야기인가 (CONTENT / MARKET / OTHER / ?)
- eval_trend_type_ok: 판정 유형(성장·급등·정착·유지·쇠퇴·저빈도)이 월별 비중 흐름과 대표 기사에 맞는가 (Y / N / ?)
- eval_suggested_type: N이면 맞다고 보는 유형
- eval_meaningful (키워드): 트렌드로 볼 만한 의미 있는 말인가, 일반어·기업명·시세 표현인가 (Y / N / ?)
- eval_signal_ok (키워드): 신호 유형이 맞는가 (Y / N / ?)
- eval_comment: 짧은 근거

## 판정 규칙 요약
- 주제: E4.2 고정 모델(LSA100 K120)의 월별 배정. 유형 규칙은 키워드 신호와 같다 (최근 6개월 대 직전 6개월 비중 비, Robust Z, 버스트, 12개월 지속).
  월 가중 기사 5건 미만이면 분기 단위로 다시 판정(signal_resolution=분기), 분기로도 부족하면 저빈도.
- lineage_recent: 6개월 창 재군집에서 시드 과반으로 확인된 사건 (new=새 흐름, ended=흐름 종료). 분할·병합은 군집 무작위성과 구별되지 않아 쓰지 않는다.
- entity_driven: 최근 12개월 주제 기사의 절반 이상이 한 기업 기사. cross_sector: 3개 이상 분야·정규화 엔트로피 0.6 이상.
- 판정은 제안이다. LLM 평가를 받으면 참고 의견으로만 쓴다 (정답·학습 라벨 아님).
"""


def _latest(kind: str) -> str | None:
    from .runlog import REGISTRY

    p = REGISTRY / "run_registry.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    ok = [r["run_id"] for r in recs if r.get("kind") == kind and r.get("status") == "SUCCESS"]
    return ok[-1] if ok else None


def topic_tables(out_root: Path, e5b_run: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from .e5b import topic_payload

    pl = topic_payload(out_root, e5b_run, n_articles=5)
    rows = []
    for r in pl["rows"]:
        rows.append({"topic_id": r["id"], "trend_type": TT_KO.get(r["type"], r["type"]), "trend_type_en": r["type"],
                     "signal_resolution": RES_KO.get(r["res"], r["res"]), "top_keywords": ", ".join(r["kws"]),
                     "weighted_articles_12m": r["w12"], "ratio_6m": r["ratio6"], "robust_z": r["rz"], "persistence_12m": r["pers12"],
                     "share_per_1000_last": r["share_last"], "sectors_active_12m": r["sectors"], "cross_sector": int(r["cross"]),
                     "entity_driven": int(r["entity"]), "top_company_12m": r["company"], "top_company_share_12m": r["company_share"],
                     "keyword_support": r["support"], "sensitivity_flag": r["sens"], "signal_type_soft": TT_KO.get(r["soft"], r["soft"]),
                     "lineage_recent": ";".join(r["lineage"]), "topic_type_e42": r["e42"], "noise_reason_codes": r["noise_codes"],
                     "representative_articles": " || ".join(f"{a['d']} {a['t']}" for a in r["arts"]),
                     **{c: "" for c in EVAL_TOPIC}})
    t = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(["Growing", "Emerging", "Event Spike", "Established", "Stable", "Declining", "Low volume", "Noise"])}
    t = t.sort_values(["trend_type_en", "weighted_articles_12m"], key=lambda s: s.map(order) if s.name == "trend_type_en" else -s.fillna(0))
    m = pd.DataFrame([{"topic_id": r["id"], **dict(zip(pl["months"], r["series"]))} for r in pl["rows"]])
    return t, m, pl


def keyword_table(out_root: Path, e5a_run: str) -> pd.DataFrame:
    s = pd.read_csv(out_root / "runs" / e5a_run / "06a_keyword_signal.csv")
    s = s[s["signal_type"] != "Insufficient"].copy()
    keep = ["keyword", "class_auto", "entity_type", "signal_type", "signal_type_provisional", "signal_type_soft", "sensitivity_flag",
            "ratio_6m", "growth_3m", "robust_z", "persistence_12m", "burst_months_12m", "left_censored", "weighted_df_last",
            "cross_sector", "top_company_12m", "top_company_share_12m", "first_month"]
    s = s[[c for c in keep if c in s.columns]]
    s.insert(4, "signal_type_ko", s["signal_type"].map(TT_KO).fillna(s["signal_type"]))
    for c in EVAL_KW:
        s[c] = ""
    return s.sort_values(["signal_type", "weighted_df_last"], ascending=[True, False])


def radar_table(out_root: Path) -> pd.DataFrame:
    rows = []
    for layer in ("content", "market"):
        p = out_root / layer / "radar.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        prof = r["profiles"]
        lists = {"rising": r["overall"]["rising"], "established": r["overall"]["established"],
                 **{k: r.get(k, []) for k in LIST_KO if k not in ("rising", "established")}}
        for key, kws in lists.items():
            for rank, k in enumerate(kws, 1):
                pr = prof.get(k, {})
                co = pr.get("company") or {}
                arts = pr.get("arts") or {}
                ids = (arts.get("rep") or arts.get("latest") or [])[:3]
                titles = " || ".join(f"{r['articles'][str(i)]['d']} {r['articles'][str(i)]['t']}" for i in ids if str(i) in r["articles"])
                rows.append({"layer": layer, "list": LIST_KO.get(key, key), "rank": rank, "keyword": k,
                             "status": ";".join(pr.get("status") or []), "articles": pr.get("articles"),
                             "recent_share": pr.get("recent_share"), "base_share": pr.get("base_share"), "ratio": pr.get("ratio"),
                             "sectors_recent": (pr.get("diffusion") or {}).get("recent"),
                             "top_company": co.get("top"), "top_company_share": co.get("share"),
                             "evidence": " | ".join((pr.get("evidence") or [])[:3]), "example_articles": titles,
                             **{c: "" for c in EVAL_KW}})
    return pd.DataFrame(rows)


def export_pack(out_root: Path, dest: Path | None = None) -> dict:
    from datetime import datetime

    e5b = _latest("e5b_topic_trend")
    if not e5b:
        raise FileNotFoundError("E5b 실행 결과가 없다 — python -m trend_radar all 먼저")
    summ = json.loads((out_root / "runs" / e5b / "e5b_summary.json").read_text(encoding="utf-8"))
    dest = dest or out_root / "eval" / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    t, m, pl = topic_tables(out_root, e5b)
    k = keyword_table(out_root, summ["e5a_run"])
    rl = radar_table(out_root)
    files = {"01_topic_trends.tsv": t, "02_topic_monthly.tsv": m, "03_keyword_signals.tsv": k, "04_radar_lists.tsv": rl}
    for name, df in files.items():
        df = df.round(4)
        df.to_csv(dest / name, sep="\t", index=False, encoding="utf-8-sig")
    snap = next((json.loads(l).get("data_snapshot_id") for l in (__import__("trend_radar.runlog", fromlist=["REGISTRY"]).REGISTRY / "run_registry.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l).get("run_id") == e5b), "")
    (dest / "README.md").write_text(README.format(
        created=datetime.now().strftime("%Y-%m-%d %H:%M"), snapshot=snap, asof=pl["asof"],
        partial=f" ({pl['partial']} 불완전월 제외)" if pl.get("partial") else "", e5b=e5b, e42=summ["e42_run"], e5a=summ["e5a_run"],
        n_topics=len(t), n_kw=len(k), n_list=len(rl)), encoding="utf-8")
    # 명세 묶음: 개선 작업용 (현행 명세, 실험 기록, 설정, 판정 가이드, 결정 기록, 개선 검토 요청문)
    import shutil

    from .runlog import REGISTRY
    root = Path(__file__).resolve().parent.parent
    spec = dest / "spec"
    spec.mkdir(exist_ok=True)
    for src in (root / "docs" / "SPEC_CURRENT.md", root / "docs" / "TREND_RADAR_V2.md", root / "trend_radar.yaml",
                root / "docs" / "review" / "annotation_guide.md", REGISTRY / "decision_log.jsonl",
                root / "docs" / "review" / "improvement_prompt.md"):
        if src.exists():
            shutil.copy2(src, spec / src.name)
    zp = dest.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dest.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(dest).as_posix())
    return {"dir": str(dest), "zip": str(zp), "rows": {n: len(d) for n, d in files.items()}}
