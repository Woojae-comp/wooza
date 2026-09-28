"""E2.3b: 사전분포 있는 Dawid-Skene, 계층형 H2, 계열 통합, 게이트."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_radar.e23 import AB, C, M, O, VC, VM, VO
from trend_radar.e23b import ds_fit, fit_g3, fit_h2, lm_gates


def _fam(n=4000, seed=0, other_cov=0.03):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.35, 0.45, 0.2])
    def lf(t, vote, hit, false):
        return np.where(rng.random(n) < np.where(y == t, hit, false), vote, AB)
    F = pd.DataFrame({"F_content": lf(C, VC, 0.7, 0.05), "F_market": lf(M, VM, 0.8, 0.05),
                      "F_other_industry": lf(O, VO, other_cov * 5, 0.005), "F_politics_society": lf(O, VO, other_cov * 4, 0.005),
                      "F_broadcaster_source": lf(O, VO, other_cov * 3, 0.005), "F_content_policy": lf(C, VC, 0.05, 0.005)})
    return F, y


def test_prior_strength_shrinks_toward_accuracy():
    F, _ = _fam()
    V = F.to_numpy()
    _, th0, _ = ds_fit(V, 3, {VC: C, VM: M, VO: O}, strength=0.0)
    _, th1, _ = ds_fit(V, 3, {VC: C, VM: M, VO: O}, strength=1e6)
    # 강한 사전분포 → 대상 클래스 투표 확률 / 다른 클래스 = acc/((1-acc)/2) = 4.67
    j = list(F.columns).index("F_content")
    r = th1[j, C, VC] / th1[j, M, VC]
    assert abs(r - 0.7 / 0.15) < 0.2
    assert not np.allclose(th0, th1)


def test_h2_probabilities_sum_to_one_and_market_first():
    F, y = _fam(seed=1)
    q, info = fit_h2(F, 5.0)
    assert np.allclose(q.sum(1), 1)
    covered = (F.to_numpy() != AB).any(1)
    assert (q.argmax(1)[covered & (y == M)] == M).mean() > 0.8
    assert np.allclose(info["pi"].sum(), 1, atol=1e-6)


def test_g3_vs_h2_run_and_class_order():
    F, y = _fam(seed=2)
    for fit in (fit_g3, fit_h2):
        q, info = fit(F, 5.0)
        assert q.shape == (len(F), 3) and info["pi"].shape == (3,)
        covered = (F.to_numpy() != AB).any(1)
        assert (q.argmax(1)[covered] == y[covered]).mean() > 0.6


def test_lm_gates_all_conditions():
    r = {"probe_in_rate_drop_pp": 8, "anchor_in_rate_drop_pp": 1, "policy_to_exclude_rate": 0.005, "broadcast_excess_drop_pp": 1,
         "content_anchor_pred_content": 0.7, "max_family_removal_content_change_pp": 5, "prior_LM_agreement": 0.97}
    assert lm_gates(r, {})["pass_all"]
    assert not lm_gates({**r, "content_anchor_pred_content": 0.32}, {})["pass_all"]      # E2.3 붕괴
    assert not lm_gates({**r, "prior_LM_agreement": 0.9}, {})["pass_all"]
    assert not lm_gates({**r, "max_family_removal_content_change_pp": 15}, {})["pass_all"]
