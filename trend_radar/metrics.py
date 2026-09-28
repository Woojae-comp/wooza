"""키워드 지표와 시계열 (설계 6장).

절대 기사 수와 함께 반드시 같은 기간 전체 기사 대비 비중을 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .text import DocTerm


def add_periods(articles: pd.DataFrame) -> pd.DataFrame:
    a = articles.copy()
    a["month"] = a["date"].dt.to_period("M").astype(str)
    a["quarter"] = a["date"].dt.to_period("Q").astype(str)
    a["year"] = a["date"].dt.year.astype(str)
    return a


def period_counts(dt: DocTerm, labels: pd.Series, rows: np.ndarray | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """기간별 (키워드 × 기간) 기사 수와 기간별 전체 기사 수."""
    lab = labels.to_numpy() if rows is None else labels.to_numpy()[rows]
    X = dt.X if rows is None else dt.X[rows]
    periods = sorted(set(lab))
    idx = {p: i for i, p in enumerate(periods)}
    P = sparse.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), [idx[p] for p in lab])),
                          shape=(len(lab), len(periods)))
    counts = np.asarray((X.T @ P).todense())
    totals = np.asarray(P.sum(axis=0)).ravel()
    return periods, counts, totals


def keyword_table(dt: DocTerm, arts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """키워드별 요약 지표와 월·분기·연 시계열."""
    n = len(arts)
    dates = arts["date"].to_numpy()
    first, last = [], []
    for j in range(len(dt.vocab)):
        d = dates[dt.Xc.indices[dt.Xc.indptr[j]:dt.Xc.indptr[j + 1]]]
        first.append(d.min())
        last.append(d.max())
    tab = pd.DataFrame({
        "keyword": dt.vocab,
        "articles": dt.df,
        "share": dt.df / n,
        "occurrences": dt.tf,
        "first_seen": pd.to_datetime(first).date,
        "last_seen": pd.to_datetime(last).date,
    })
    series = {}
    for unit in ("month", "quarter", "year"):
        periods, counts, totals = period_counts(dt, arts[unit])
        series[unit] = {"periods": periods, "counts": counts, "totals": totals,
                        "shares": counts / np.maximum(totals, 1)}
    # 전기 대비·전년 동기 대비 (최근 분기 기준, 비중 변화)
    q = series["quarter"]
    sh = q["shares"]
    if sh.shape[1] >= 2:
        tab["qoq_share_change"] = _rel(sh[:, -1], sh[:, -2])
        tab["qoq_count_change"] = _rel(q["counts"][:, -1], q["counts"][:, -2])
    if sh.shape[1] >= 5:
        tab["yoy_share_change"] = _rel(sh[:, -1], sh[:, -5])
        tab["yoy_count_change"] = _rel(q["counts"][:, -1], q["counts"][:, -5])
    tab["last_quarter"] = q["periods"][-1]
    return tab, series


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, a / b - 1.0, np.nan)
    return r


def long_series(dt: DocTerm, series: dict, unit: str, top: int | None = None) -> pd.DataFrame:
    s = series[unit]
    k = len(dt.vocab) if top is None else min(top, len(dt.vocab))
    rows = []
    for j in range(k):
        for i, p in enumerate(s["periods"]):
            c = s["counts"][j, i]
            if c:
                rows.append((dt.vocab[j], p, int(c), float(s["shares"][j, i])))
    return pd.DataFrame(rows, columns=["keyword", unit, "articles", "share"])
