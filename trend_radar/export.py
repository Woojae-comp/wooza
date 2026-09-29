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

TT_KO = {"Emerging": "신규 부상", "Growing": "보도 비중 증가", "Established": "지속 보도", "Stable": "유지(잔여)", "Declining": "보도 비중 감소",
         "Event Spike": "단기 집중", "Low volume": "최근 자료 부족", "Noise": "잡음 후보", "Insufficient": "자료 부족"}
RES_KO = {"month": "월", "quarter": "분기", "low_volume": "저빈도"}
LIST_KO = {"rising": "A 확대되는 이슈", "established": "A 지속 핵심", "cross_sector": "C 여러 업종 공통", "emerging": "E 부상",
           "growing": "E 성장", "declining": "E 쇠퇴", "event": "E 이벤트성", "spreading": "D 관련 업종 증가", "converging": "D 연관어 연결 증가", "companies_rising": "A 기업별 보도 증가(탐색용)",
           "structural": "F 구조 변화"}
EVAL_TOPIC = ["eval_topic_coherent", "eval_is_content_industry", "eval_trend_type_ok", "eval_suggested_type", "eval_comment"]
EVAL_KW = ["eval_meaningful", "eval_is_content_industry", "eval_signal_ok", "eval_comment"]

README = """# Trend Radar 평가 묶음

생성: {created} · 스냅샷 {snapshot} · 확정 기준월 {asof}{partial}
출처: E5b {e5b} · E4.2 {e42} · E5a {e5a} · 키워드 레이더 out/{{content,market}}/radar.json

## 파일
| 파일 | 행 | 내용 |
|---|---|---|
| 00_manifest.tsv | | 표별 실행ID·스냅샷·설정 해시·커밋·사전 해시·집계 기준 |
| 01_topic_trends.tsv | {n_topics} | 주제 트렌드 (화면 '주제 트렌드' 탭). 잡음 후보도 포함 (trend_type=잡음 후보, 근거 noise_basis) |
| 02_topic_monthly.tsv | {n_monthly} | 주제 × 월: 원 기사 수, 가중 기사 수 W, 분모 N, 1,000건당 비중, 불완전월·판정 포함 여부 |
| 05_topic_evidence.tsv | {n_ev} | 판정 근거 기사 (판정 기간 안, 기사ID·요약·E2 판정·배정 신뢰도·선정 사유) |
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

## 계산식 (재현용)
- 비중: share = W / N × 1000. W = 주제에 HIGH·LOW로 배정된 기사의 판정 가중 합(INCLUDE 1, REVIEW 0.5), N = 같은 달 E2 유입 기사 전체의 판정 가중 합
  (미배정·거부 기사도 분모에 남는다 → 한 달 주제 비중 합이 1,000보다 작다).
- 증가 배율 ratio_6m = ((ΣW_최근6 + 1) / ΣN_최근6) / ((ΣW_직전6 + 1) / ΣN_직전6). 월 비중의 평균끼리 나눈 값이 아니라 **기간 합산 후 +1 평활**이다.
  최근 6개월 = 확정 기준월까지 6개월. growth_3m은 마지막 달 대 직전 3개월 평균의 같은 형식.
- Robust Z = (마지막 달 비중 − 직전 12개월 비중 중앙값) / (1.4826 × MAD), ±10에서 자름 (MAD=0이면 같으면 0, 다르면 ±10). 버스트 = Kleinberg 2상태 (s=2, γ=1).
- 분기 판정: 확정 기준월에서 끝나는 3개월 묶음으로 뒤에서부터 자르고, 앞쪽 모자란 달은 버린다 (2026-08 기준 마지막 분기 = 2026-06~08).
- 키워드 레이더(04)는 E5와 다른 체계: 최근 2분기 대 직전 4분기 분기 비중 비교 (compare 열).

## 표시 명칭 (산업 성장·융합으로 읽지 않도록)
| 내부 유형 | 표시 | 뜻 |
|---|---|---|
| Growing / Declining | 보도 비중 증가 / 감소 | 유입 기사 중 비중의 증감. 산업의 실물 성장·쇠퇴가 아님 |
| Established | 지속 보도 | 12개월 거의 매달 보도, 비중 변화 작음 |
| Stable | 유지(잔여) → trend_note: 방향 혼재 / 변화 작음 | 다른 유형 조건을 못 넘은 나머지. 방향 혼재 = 6개월 비는 기준을 넘었지만 Robust Z·지속 조건 불충족 |
| Event Spike | 단기 집중 | 짧은 버스트 |
| Converging (레이더) | 연관어 연결 증가 | 연관 키워드 연결이 늘었다는 뜻. 산업 간 융합을 입증하지 않음 |
| Spreading / 분야 확산 | 관련 업종 증가 / 관련 기업 업종 다수 | 기사 분야 = 연결 기업의 업종 합집합. 기사 내용상 분야 확산과 다름 |
- 기업명 키워드는 A·C·D·E 목록에서 빼고 'A 기업별 보도 증가(탐색용)'에 따로 둔다 (기업 안의 작품·사건을 찾는 출발점).
- 03의 keyword_role: company / general(보고서체 일반어) / market(시세·공시 표현) / format(방송 안내 등) / entity / concept. trend_eligible=False는 신호 목록에서 제외.

## 판정 규칙 요약
- 주제: E4.2 고정 모델(LSA100 K120)의 월별 배정. 유형 규칙은 키워드 신호와 같다 (최근 6개월 대 직전 6개월 비중 비, Robust Z, 버스트, 12개월 지속).
  월 가중 기사 5건 미만이면 분기 단위로 다시 판정(signal_resolution=분기), 분기로도 부족하면 저빈도.
- lineage_stable_all: 6개월 창 재군집에서 시드 과반으로 확인된 사건 전체(split·merged 포함). lineage_display: 그중 화면에 쓰는 new(새 흐름)·ended(흐름 종료)만.
  split·merged는 같은 창 재군집(무작위성 기준)과 건수가 비슷해 쓰지 않는다.
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
                     "trend_note": r.get("note", ""), "lineage_display": ";".join(r["lineage"]), "lineage_stable_all": ";".join(r.get("lineage_all", [])),
                     "topic_type_e42": r["e42"], "noise_reason_codes": r["noise_codes"], "noise_basis": r.get("noise_basis", ""),
                     "evidence_articles_signal_period": " || ".join(f"{a['d']} [{a['dec']}/{a['conf']}] {a['t']}" for a in r.get("evidence", [])),
                     "representative_articles_all_period": " || ".join(f"{a['d']} {a['t']}" for a in r["arts"]),
                     **{c: "" for c in EVAL_TOPIC}})
    t = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(["Growing", "Emerging", "Event Spike", "Established", "Stable", "Declining", "Low volume", "Noise"])}
    t = t.sort_values(["trend_type_en", "weighted_articles_12m"], key=lambda s: s.map(order) if s.name == "trend_type_en" else -s.fillna(0))
    mp = out_root / "runs" / e5b_run / "06b_topic_trend_monthly.parquet"
    m = pd.read_parquet(mp)
    cols = [c for c in ["topic_id", "month", "raw_articles", "weighted_articles", "month_total_weighted", "share_per_1000",
                        "is_partial_month", "in_signal_window"] if c in m.columns]
    m = m[cols].sort_values(["topic_id", "month"])
    return t, m, pl


def evidence_table(out_root: Path, e5b_run: str) -> pd.DataFrame:
    p = out_root / "runs" / e5b_run / "06b_topic_evidence.csv"
    return pd.read_csv(p, keep_default_na=False) if p.exists() else pd.DataFrame()


def keyword_table(out_root: Path, e5a_run: str) -> pd.DataFrame:
    s = pd.read_csv(out_root / "runs" / e5a_run / "06a_keyword_signal.csv")
    s = s[s["signal_type"] != "Insufficient"].copy()
    keep = ["keyword", "class_auto", "entity_type", "keyword_role", "trend_eligible", "signal_type", "signal_type_provisional", "signal_type_soft", "sensitivity_flag",
            "ratio_6m", "growth_3m", "robust_z", "persistence_12m", "burst_months_12m", "left_censored", "weighted_df_last",
            "cross_sector", "top_company_12m", "top_company_share_12m", "first_month"]
    s = s[[c for c in keep if c in s.columns]]
    s.insert(s.columns.get_loc("signal_type") + 1, "signal_type_ko", s["signal_type"].map(TT_KO).fillna(s["signal_type"]))
    for c in EVAL_KW:
        s[c] = ""
    by = [c for c in ["trend_eligible", "signal_type", "weighted_df_last"] if c in s.columns]
    return s.sort_values(by, ascending=[False, True, False][-len(by):])


def radar_table(out_root: Path) -> pd.DataFrame:
    rows = []
    for layer in ("content", "market"):
        p = out_root / layer / "radar.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        prof = r["profiles"]
        lists = {"rising": r["overall"]["rising"], "established": r["overall"]["established"],
                 "companies_rising": r["overall"].get("companies_rising", []),
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
                             "top_company": co.get("top") if co.get("share") else "", "top_company_share": co.get("share") or None,
                             "compare": f"최근 {r['meta'].get('recent_q', ['?'])[0]}~{r['meta'].get('recent_q', ['?'])[-1]} 대 직전 "
                                        f"{r['meta'].get('base_q', ['?'])[0]}~{r['meta'].get('base_q', ['?'])[-1]} (분기 보도 비중)",
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
    ev = evidence_table(out_root, e5b)
    files = {"01_topic_trends.tsv": t, "02_topic_monthly.tsv": m, "03_keyword_signals.tsv": k, "04_radar_lists.tsv": rl,
             "05_topic_evidence.tsv": ev}
    for name, df in files.items():
        df = df.round(4)
        df.to_csv(dest / name, sep="\t", index=False, encoding="utf-8-sig")
    snap = next((json.loads(l).get("data_snapshot_id") for l in (__import__("trend_radar.runlog", fromlist=["REGISTRY"]).REGISTRY / "run_registry.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l).get("run_id") == e5b), "")
    (dest / "README.md").write_text(README.format(
        created=datetime.now().strftime("%Y-%m-%d %H:%M"), snapshot=snap, asof=pl["asof"],
        partial=f" ({pl['partial']} 불완전월 제외)" if pl.get("partial") else "", e5b=e5b, e42=summ["e42_run"], e5a=summ["e5a_run"],
        n_topics=len(t), n_kw=len(k), n_list=len(rl), n_monthly=len(m), n_ev=len(ev)), encoding="utf-8")
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
    # 표별 출처 (같은 스냅샷·실행에서 나온 결과인지 판별)
    import hashlib

    from .runlog import REGISTRY as _R
    runs = {json.loads(l)["run_id"]: json.loads(l) for l in (_R / "run_registry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    lex = root / "lexicon.yaml"
    lex_sha = hashlib.sha256(lex.read_bytes()).hexdigest()[:12] if lex.exists() else ""
    mf = []
    for name, rid, basis in [("01_topic_trends.tsv", e5b, f"확정 기준월 {pl['asof']}, 최근 6개월 {pl['periods']['recent6'][0]}~{pl['periods']['recent6'][1]} 대 직전 6개월"),
                             ("02_topic_monthly.tsv", e5b, "월별 (in_signal_window=True인 달만 판정에 사용)"),
                             ("03_keyword_signals.tsv", summ["e5a_run"], f"확정 기준월 {pl['asof']}"),
                             ("04_radar_lists.tsv", "out/{content,market}/radar.json", "최근 2분기 대 직전 4분기 (키워드 레이더, E5와 비교 체계가 다름)"),
                             ("05_topic_evidence.tsv", e5b, "판정 기간 안의 배정 기사만")]:
        rr = runs.get(rid, {})
        mf.append({"table": name, "run_id": rid, "data_snapshot_id": rr.get("data_snapshot_id", ""), "config_hash": rr.get("config_hash", ""),
                   "git_commit": rr.get("git_commit", ""), "lexicon_sha256_12": lex_sha, "basis": basis})
    pd.DataFrame(mf).to_csv(dest / "00_manifest.tsv", sep="\t", index=False, encoding="utf-8-sig")
    if lex.exists():
        shutil.copy2(lex, spec / "lexicon.yaml")
    zp = dest.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dest.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(dest).as_posix())
    return {"dir": str(dest), "zip": str(zp), "rows": {n: len(d) for n, d in files.items()}}
