"""데이터 품질 점검 (v2 E0, 기획안 4.2). 결과는 01_data_quality.xlsx."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .load import Corpus


def _check(rows: list, name: str, value: float, threshold: str, bad: bool, note: str = "") -> None:
    rows.append({"점검 항목": name, "값": value, "경고 기준": threshold, "상태": "경고" if bad else "정상", "비고": note})


def data_quality(raw: pd.DataFrame, corpus: Corpus, cfg: dict, snapshot_date: str) -> dict[str, pd.DataFrame]:
    col = cfg["input"]["columns"]
    arts = corpus.articles
    rep = corpus.report
    checks: list[dict] = []

    # 요약 누락 (본문이 없으므로 요약이 분석 텍스트)
    s = raw[col["summary"]].fillna("").astype(str).str.strip()
    short = s.str.len() < 20
    day = pd.to_datetime(raw[col["date"]], errors="coerce").dt.date
    daily = short.groupby(day).mean()
    _check(checks, "요약 누락률 (20자 미만, 원자료 행)", round(float(short.mean()), 4), "일별 5% 초과",
           bool((daily > 0.05).any()), f"5% 초과 일수 {int((daily > 0.05).sum())}일")

    # 중복: 원자료 행 대비 기사, 상위 단계 중복 판정
    n_rows, n_art = len(raw), raw[col["article_id"]].nunique()
    dup_rate = 1 - n_art / n_rows
    _check(checks, "행 중복률 (같은 기사가 여러 기업·검색에 수집)", round(dup_rate, 4), "40% 초과", dup_rate > 0.4,
           f"원자료 {n_rows:,}행 → 기사 {n_art:,}건")
    if "중복판정" in raw:
        vc = raw.drop_duplicates(col["article_id"])["중복판정"].value_counts(normalize=True).round(4).to_dict()
        _check(checks, "상위 단계 중복 판정 분포", str(vc), "-", False, "원자료에 이미 적용됨 (대표 기사만 수록)")

    # 분야 미분류
    empty = raw[col["sectors"]].fillna("").astype(str).str.strip().eq("")
    _check(checks, "분야 미분류율 (원자료 행)", round(float(empty.mean()), 4), "10% 초과", float(empty.mean()) > 0.1)

    # 날짜 이상
    d = pd.to_datetime(raw[col["date"]], errors="coerce")
    future = d > pd.Timestamp(snapshot_date)
    _check(checks, "날짜 이상 (파싱 실패·스냅샷 이후)", int(d.isna().sum() + future.sum()), "1건 이상",
           bool(d.isna().sum() + future.sum() > 0))

    # 분야별 매체 편중·기업 편중 (정제 후 기사 기준)
    a_sec = corpus.article_sector.merge(arts[["gid", "press", "companies"]], on="gid")
    press = (a_sec.groupby("sector")["press"].agg(lambda x: x.value_counts(normalize=True).iloc[0]).rename("상위 매체 점유")
             .to_frame().join(a_sec.groupby("sector")["press"].agg(lambda x: x.value_counts().index[0]).rename("상위 매체")))
    co = corpus.article_company.merge(corpus.article_sector, on="gid")
    comp = (co.groupby("sector")["company"].agg(lambda x: x.value_counts(normalize=True).iloc[0]).rename("상위 기업 점유")
            .to_frame().join(co.groupby("sector")["company"].agg(lambda x: x.value_counts().index[0]).rename("상위 기업")))
    conc = press.join(comp).join(a_sec.groupby("sector").size().rename("기사 수")).reset_index()
    for _, r in conc.iterrows():
        if r["상위 매체 점유"] > 0.35:
            _check(checks, f"매체 편중: {r['sector']}", round(r["상위 매체 점유"], 3), "분야별 35% 초과", True, r["상위 매체"])
        if r["상위 기업 점유"] > 0.20:
            _check(checks, f"기업 편중: {r['sector']}", round(r["상위 기업 점유"], 3), "분야별 20% 초과", True, r["상위 기업"])

    # 정제 과정
    for k, label in (("rows_company_mentioned", "기업명 실제 등장 행"), ("rows_company_strong", "기업이 주인공인 행"),
                     ("articles", "정제 후 기사"), ("excluded_by_title_tag", "시세 태그로 뺀 기사"),
                     ("boilerplate_segment_count", "반복 문형 조각 제거")):
        if k in rep:
            _check(checks, label, rep[k], "-", False)

    # 월별 기사 수 (분야별)
    m = a_sec.merge(arts[["gid", "date"]], on="gid")
    m["month"] = m["date"].dt.to_period("M").astype(str)
    monthly = m.pivot_table(index="month", columns="sector", values="gid", aggfunc="count", fill_value=0)
    monthly["전체(기사)"] = arts.assign(month=arts["date"].dt.to_period("M").astype(str)).groupby("month").size()

    return {
        "점검 요약": pd.DataFrame(checks),
        "분야별 편중": conc,
        "월별 기사 수": monthly.reset_index(),
        "기업 등장률": pd.DataFrame(rep.get("company_mention", [])),
        "반복 문형": pd.DataFrame(rep.get("boilerplate_segments", [])),
        "제목 태그": pd.Series(rep.get("title_tags", {}), name="기사 수").rename_axis("태그").reset_index(),
    }


def write_quality(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path) as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
