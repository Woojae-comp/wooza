"""원자료 로드와 기사 단위 정제 (설계 3장).

원자료는 기업×검색구분 단위 행이다. 여기서 세 개의 표를 만든다.
- articles: 전역기사ID 기준 1행 (전체 산업 분석 단위)
- article_sector: 전역기사ID + 분야 (분야 분석 단위)
- article_company: 전역기사ID + 기업 (기업 레이어)
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from collections import Counter


@dataclass
class Corpus:
    articles: pd.DataFrame        # gid, date, title, summary, press, link, companies, sectors
    article_sector: pd.DataFrame  # gid, sector
    article_company: pd.DataFrame  # gid, company, sectors (그 기업 행에서 온 분야)
    report: dict                  # 정제 과정 기록


def read_raw(paths: list[str] | str) -> pd.DataFrame:
    if isinstance(paths, str):
        paths = [paths]
    files: list[str] = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) or [p])
    frames = []
    for f in files:
        suf = Path(f).suffix.lower()
        if suf in (".xlsx", ".xls"):
            frames.append(pd.read_excel(f, dtype=str))
        elif suf == ".parquet":
            frames.append(pd.read_parquet(f))
        else:
            frames.append(pd.read_csv(f, dtype=str))
    if not frames:
        raise FileNotFoundError(f"원자료가 없다: {paths}")
    return pd.concat(frames, ignore_index=True)


def split_sectors(value: str, separators: str) -> list[str]:
    parts = (p.strip() for p in re.split(rf"\s*{separators}\s*", str(value or "")))
    return sorted({p for p in parts if p and p.lower() != "nan"})


def company_patterns(companies: list[str], aliases: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """기업명 등장 판정 패턴. 앞 글자가 한글·영숫자가 아니어야 한다 ('삼천포대교'는 '대교'가 아님).
    뒤는 조사가 붙으므로 한글은 허용하고 영숫자만 막는다."""
    out = {}
    for c in companies:
        names = {c, c.replace(" ", "")} | set(aliases.get(c, []))
        alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        out[c] = re.compile(rf"(?<![가-힣A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])")
    return out


SEGMENT_SPLIT = re.compile(r"\.\.\.|…|(?<=[다요])\.\s+|▶|※|\|")
TITLE_TAG = re.compile(r"^\s*[\[【<]([^\]】>]{1,20})[\]】>]")


def title_tag(title: str) -> str | None:
    m = TITLE_TAG.match(title or "")
    return m.group(1).strip() if m else None


def strip_boilerplate(summaries: pd.Series, min_articles: int, names: list[str] | None = None
                      ) -> tuple[pd.Series, list[tuple[str, int]]]:
    """여러 기사에 반복되는 문장 조각(언론사 안내문, 자동 생성 시세 기사 문형, 종목 나열)을 요약에서 뺀다.
    숫자와 기업명을 가린 형태가 min_articles개 이상 기사에 나오면 반복 문형으로 본다."""
    name_pat = re.compile("|".join(re.escape(n) for n in sorted(set(names or []), key=len, reverse=True) if len(n) >= 2)) \
        if names else None

    def norm(x: str) -> str:
        x = re.sub(r"\d+", "0", x)
        return name_pat.sub("@", x) if name_pat else x

    segs = [[x.strip() for x in SEGMENT_SPLIT.split(s)] for s in summaries]
    keys = [[norm(x) if len(x) >= 10 else None for x in ss] for ss in segs]
    cnt = Counter()
    example = {}
    for ss, ks in zip(segs, keys):
        for x, k in zip(ss, ks):
            if k is not None:
                example.setdefault(k, x)
        cnt.update({k for k in ks if k is not None})
    bad = {k for k, c in cnt.items() if c >= min_articles}
    cleaned = [" ... ".join(x for x, k in zip(ss, ks) if x and k not in bad) for ss, ks in zip(segs, keys)]
    removed = sorted(((example[k], cnt[k]) for k in bad), key=lambda t: -t[1])
    return pd.Series(cleaned, index=summaries.index), removed


def build_corpus(raw: pd.DataFrame, cfg: dict) -> Corpus:
    col = cfg["input"]["columns"]
    clean = cfg.get("cleaning", {})
    rep: dict = {"raw_rows": int(len(raw))}

    df = pd.DataFrame({
        "gid": raw[col["article_id"]].astype(str),
        "date": pd.to_datetime(raw[col["date"]], errors="coerce").dt.normalize(),
        "title": raw[col["title"]].fillna("").astype(str),
        "summary": raw[col["summary"]].fillna("").astype(str),
        "company": raw[col["company"]].fillna("").astype(str).str.strip(),
        "sector_raw": raw[col["sectors"]].fillna("").astype(str),
        "press": raw.get(col.get("press", ""), pd.Series("", index=raw.index)).fillna("").astype(str),
        "link": raw.get(col.get("link", ""), pd.Series("", index=raw.index)).fillna("").astype(str),
    })
    df = df[df["date"].notna() & (df["gid"] != "")]
    rep["rows_with_date"] = int(len(df))

    excluded = set(clean.get("exclude_companies") or [])
    if excluded:
        df = df[~df["company"].isin(excluded)]
    rep["excluded_companies"] = sorted(excluded)

    # 기업 등장 여부 (행 단위: 이 기업으로 검색된 기사에 이 기업이 실제로 나오는가)
    pats = company_patterns(sorted(df["company"].unique()), clean.get("company_aliases") or {})
    text = df["title"] + " " + df["summary"]
    df["mentioned"] = [bool(pats[c].search(t)) for c, t in zip(df["company"], text)]
    rep["rows_company_mentioned"] = int(df["mentioned"].sum())
    by_co = df.groupby("company")["mentioned"].agg(["size", "mean"]).rename(columns={"size": "rows", "mean": "mention_rate"})
    rep["company_mention"] = by_co.sort_values("mention_rate").reset_index().to_dict("records")
    if clean.get("require_company_mention", True):
        df = df[df["mentioned"]]
    rep["rows_after_cleaning"] = int(len(df))

    sep = cfg["input"].get("sector_separators", "[·,|]")
    df["sectors"] = [split_sectors(v, sep) for v in df["sector_raw"]]

    # 분야 매핑: 전역기사ID + 분야 (같은 분야 여러 기업에 걸리면 한 번)
    a_sec = df[["gid", "sectors"]].explode("sectors").rename(columns={"sectors": "sector"})
    a_sec = a_sec.dropna().drop_duplicates().reset_index(drop=True)
    a_co = (df.explode("sectors").dropna(subset=["sectors"]).groupby(["gid", "company"])["sectors"]
            .agg(lambda x: sorted(set(x))).reset_index())

    first = df.sort_values(["gid", "date"]).drop_duplicates("gid", keep="first")
    arts = first[["gid", "date", "title", "summary", "press", "link"]].copy()
    arts = arts.merge(df.groupby("gid")["company"].agg(lambda s: sorted(set(s))).rename("companies"), on="gid")
    arts = arts.merge(a_sec.groupby("gid")["sector"].agg(lambda s: sorted(set(s))).rename("sectors"), on="gid", how="left")
    arts["sectors"] = arts["sectors"].apply(lambda v: v if isinstance(v, list) else [])
    arts = arts.sort_values(["date", "gid"]).reset_index(drop=True)
    # 반복 안내문 제거 (분석 텍스트만; 원래 요약은 summary_raw로 남긴다)
    arts["summary_raw"] = arts["summary"]
    min_rep = clean.get("boilerplate_min_articles", 20)
    if min_rep:
        names = sorted(set(df["company"]) | {a for al in (clean.get("company_aliases") or {}).values() for a in al})
        arts["summary"], removed = strip_boilerplate(arts["summary"], min_rep, names)
        rep["boilerplate_segments"] = [{"text": t[:120], "articles": c} for t, c in removed[:200]]
        rep["boilerplate_segment_count"] = len(removed)
    # 제목 머리말 태그 ([특징주], [마감시황] …): 사용자가 승인한 태그만 제외한다
    arts["tag"] = arts["title"].map(title_tag)
    rep["title_tags"] = arts["tag"].value_counts().head(80).to_dict()
    ex_tags = set(clean.get("exclude_title_tags") or [])
    if ex_tags:
        before = len(arts)
        arts = arts[~arts["tag"].isin(ex_tags)].reset_index(drop=True)
        keep = set(arts["gid"])
        a_sec = a_sec[a_sec["gid"].isin(keep)].reset_index(drop=True)
        a_co = a_co[a_co["gid"].isin(keep)].reset_index(drop=True)
        rep["excluded_by_title_tag"] = before - len(arts)
    rep["excluded_title_tags"] = sorted(ex_tags)
    rep["articles"] = int(len(arts))
    rep["date_min"] = str(arts["date"].min().date())
    rep["date_max"] = str(arts["date"].max().date())
    rep["sectors"] = a_sec["sector"].value_counts().to_dict()
    return Corpus(arts, a_sec, a_co, rep)
