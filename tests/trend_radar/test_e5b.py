"""E5b: 주제 × 월 계열, 창 나누기, 이웃 창 연결과 계보 사건."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_radar.e5b import entropy_norm, lineage_events, lineage_recent, link_windows, topic_months, windows


def test_topic_months_counts_only_confident_and_total_all():
    asg = pd.DataFrame({"date": ["2024-01-03", "2024-01-20", "2024-02-02", "2024-02-05"],
                        "topic_id": ["tp_000", "tp_001", "tp_000", "tp_000"],
                        "assignment_confidence": ["HIGH", "REJECTED", "LOW", "HIGH"],
                        "weight_half": [1.0, 0.5, 0.5, 1.0]})
    topics, W, T = topic_months(asg, ["2024-01", "2024-02", "2024-03"])
    assert topics == ["tp_000"]                       # REJECTED만 있는 주제는 계열 없음
    assert W.tolist() == [[1.0, 1.5, 0.0]] and T.tolist() == [1.5, 1.5, 0.0]


def test_windows_end_at_confirmed_month():
    ms = [f"2024-{m:02d}" for m in range(1, 13)]
    assert windows(ms, 10, 6, 3) == [(2, 7), (5, 10)]   # 마지막 창은 확정월(인덱스 10)에서 끝난다


def test_link_and_events_split_merge_new_ended():
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    B = np.array([[1, 0.1, 0], [1, -0.1, 0], [0, 1, 0.05], [-1, -1, -1.0]])
    links = link_windows(A, B, 0.9)
    ev = lineage_events(3, 4, links)
    assert ev["split"] == [0]                         # A0 → B0, B1
    assert (1, 2) in ev["continued"]
    assert ev["ended"] == [2] and ev["new"] == [3]


def test_lineage_recent_maps_to_fixed_topics():
    lin = pd.DataFrame([{"event": "split", "from_fixed_topic": "tp_001", "to_fixed_topic": None, "is_latest_step": True},
                        {"event": "new", "from_fixed_topic": None, "to_fixed_topic": "tp_009", "is_latest_step": True},
                        {"event": "ended", "from_fixed_topic": "tp_002", "to_fixed_topic": None, "is_latest_step": False}])
    assert lineage_recent(lin) == {"tp_001": "split", "tp_009": "new"}


def test_entropy_norm():
    assert entropy_norm(np.array([5.0])) == 0.0
    assert abs(entropy_norm(np.array([1.0, 1.0, 1.0])) - 1.0) < 1e-9


def test_to_quarters_ends_at_confirmed_month():
    from trend_radar.e5b import to_quarters

    W = np.arange(10, dtype=float)[None, :]         # 월 0..9, 확정월 8
    Wq, Tq = to_quarters(W, np.ones(10), 8)
    assert Wq.tolist() == [[0 + 1 + 2, 3 + 4 + 5, 6 + 7 + 8]] and Tq.tolist() == [3, 3, 3]
    Wq2, _ = to_quarters(W, np.ones(10), 7)          # 앞쪽 모자란 달(0,1)은 버린다
    assert Wq2.tolist() == [[2 + 3 + 4, 5 + 6 + 7]]


def test_resolve_signal_month_quarter_low_volume():
    from trend_radar.e5b import resolve_signal

    rules = {"min_monthly_df": 5}
    T = np.full(24, 100.0)
    _, res = resolve_signal(np.full(24, 8.0), T, 23, rules)
    assert res == "month"
    s, res = resolve_signal(np.full(24, 2.0), T, 23, rules)          # 월 2건 → 분기 6건
    assert res == "quarter" and s["signal_type"] != "Insufficient"
    _, res = resolve_signal(np.r_[np.zeros(23), 1.0], T, 23, rules)  # 분기로도 부족
    assert res == "low_volume"


def test_signal_row_per_year_default_is_monthly():
    from trend_radar.e5 import signal_row

    rng = np.random.default_rng(1)
    W, T = rng.poisson(8, 30).astype(float), np.full(30, 200.0)
    assert signal_row(W, T, {}) == signal_row(W, T, {}, per_year=12)
