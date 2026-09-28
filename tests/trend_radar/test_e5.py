"""E5a 키워드 시계열 신호: 누락 월 0 채움, 유한 증가율, 상수 계열 Z, Kleinberg 재현성, 급등 대 증가 구분, half/soft 차이 기록."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from trend_radar.e5 import growth, kleinberg, month_axis, monthly, signal_row, zscores

RULES = {"min_monthly_df": 5}


def test_missing_months_filled_with_zero():
    dates = pd.to_datetime(pd.Series(["2025-01-10", "2025-01-20", "2025-04-05"]))
    months = month_axis(dates)
    assert months == ["2025-01", "2025-02", "2025-03", "2025-04"]
    X = sparse.csr_matrix(np.array([[1], [1], [1]], float))
    W, T = monthly(X, dates, np.ones(3), months)
    assert W.shape == (1, 4)
    assert list(W[0]) == [2, 0, 0, 1] and list(T) == [2, 0, 0, 1]


def test_growth_finite_when_previous_zero():
    g = growth(10, 100, 0, 100)
    assert np.isfinite(g) and g == 11
    assert growth(0, 0, 0, 0) == 1.0          # 기사 없는 달
    s = signal_row(np.array([0, 0, 0, 0, 12.0]), np.full(5, 100.0), RULES)
    assert all(np.isfinite(s[k]) for k in ("growth_mom", "growth_3m", "ratio_6m", "z", "robust_z"))


def test_constant_series_z_zero():
    z, rz = zscores(np.float64(5.0), np.full(12, 5.0))
    assert z == 0 and rz == 0
    s = signal_row(np.full(13, 20.0), np.full(13, 200.0), RULES)
    assert s["z"] == 0 and s["robust_z"] == 0
    assert s["signal_type"] == "Established"


def test_kleinberg_reproducible_and_detects_burst():
    r = np.array([2, 2, 3, 2, 30, 35, 2, 2, 3, 2, 2, 2], float)
    d = np.full(12, 100.0)
    a, b = kleinberg(r, d), kleinberg(r, d)
    assert (a == b).all()
    assert a[4] == 1 and a[5] == 1 and a[0] == 0 and a[-1] == 0
    assert kleinberg(np.zeros(5), np.full(5, 10.0)).sum() == 0


def test_event_spike_vs_growing_by_persistence():
    T = np.full(24, 1000.0)
    spike = np.array([0.0] * 22 + [60, 3])         # 한 달 급등, 앞뒤로 거의 없음
    grow = np.array([5.0] * 12 + [6, 8, 10, 12, 14, 17, 20, 23, 26, 30, 34, 38])   # 꾸준히 증가
    s1 = signal_row(spike, T, RULES)
    s2 = signal_row(grow, T, RULES)
    assert s1["signal_type"] == "Event Spike" and s1["persistence_12m"] < 0.5
    assert s2["signal_type"] == "Growing" and s2["persistence_12m"] == 1.0


def test_half_vs_soft_difference_recorded():
    # REVIEW 기사(half 0.5)가 최근에 몰린 키워드: soft 확률이 낮으면 판정이 달라질 수 있다 → sensitivity_flag
    T = np.full(24, 1000.0)
    half = np.array([5.0] * 12 + [6, 8, 10, 12, 14, 17, 20, 23, 26, 30, 34, 38])
    soft = half * 0.1
    a = signal_row(half, T, RULES)["signal_type"]
    b = signal_row(soft, T * 0.8, RULES)["signal_type"]
    flag = int(a != b)
    assert a == "Growing" and b == "Insufficient" and flag == 1
