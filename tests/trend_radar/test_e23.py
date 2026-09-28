"""E2.3 3분류 라벨 모델: 클래스 복원, 방향 제약, NONCONTENT 처리, 클래스별 복제 학습, 통과 기준, 운영 포인터."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trend_radar.e23 import AB, C, M, O, VC, VM, VN, VO, gates3, label_model3, lf_agreement, lf_stats3, soft_rows3


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.4, 0.4, 0.2])
    def lf(target_cls, vote, hit, false):
        p = np.where(y == target_cls, hit, false)
        return np.where(rng.random(n) < p, vote, AB)
    L = pd.DataFrame({
        "c1": lf(C, VC, 0.6, 0.05), "c2": lf(C, VC, 0.5, 0.05),
        "m1": lf(M, VM, 0.6, 0.05), "m2": lf(M, VM, 0.5, 0.05),
        "o1": lf(O, VO, 0.5, 0.03), "o2": lf(O, VO, 0.4, 0.03),
        "n1": np.where((y != C) & (rng.random(n) < 0.5) | (y == C) & (rng.random(n) < 0.03), VN, AB),
    })
    return L, y


def test_label_model3_recovers_classes_and_other_separately():
    L, y = _synthetic()
    q, th, pi, tgt = label_model3(L)
    covered = (L.to_numpy() != AB).any(1)
    acc = (q.argmax(1)[covered] == y[covered]).mean()
    assert acc > 0.8
    # OTHER가 CONTENT 쪽으로 흡수되지 않는다 (이진 모델의 실패 양상)
    assert (q[(y == O) & covered].argmax(1) == C).mean() < 0.15
    assert abs(pi[O] - 0.2) < 0.1


def test_direction_constraints_hold():
    L, _ = _synthetic(seed=1)
    _, th, _, tgt = label_model3(L)
    cols = list(L.columns)
    for j, name in enumerate(cols):
        v = tgt[name]
        if v == VC:
            assert th[j, C, v] >= th[j, :, v].max() - 1e-12
        if v == VM:
            assert th[j, M, v] >= th[j, :, v].max() - 1e-12
        if v == VO:
            assert th[j, O, v] >= th[j, :, v].max() - 1e-12
        if v == VN:
            assert th[j, C, v] <= th[j, :, v].min() + 1e-12


def test_lf_stats_flags_correlated_overconfident_rules():
    L, _ = _synthetic(seed=2)
    L["c1_dup"] = L["c1"]                                     # 완전히 상관된 규칙 → 독립 가정 위반 의심
    q, th, pi, tgt = label_model3(L)
    s = lf_stats3(L, th, pi, tgt).set_index("lf")
    assert s.loc["c1_dup", "max_corr_same_direction"] > 0.9
    assert s.loc["n1", "vote"] == "NONCONTENT"


def test_agreement_noncontent_compatible_with_market_and_other():
    L = pd.DataFrame({"a": [VN] * 30, "b": [VM] * 15 + [VO] * 15, "c": [VC] * 30})
    ag = lf_agreement(L)
    assert ag.loc["a", "b"] == 1.0 and ag.loc["a", "c"] == 0.0


def test_soft_rows3_class_duplication_and_confidence():
    q = np.array([[0.8, 0.1, 0.1], [1 / 3, 1 / 3, 1 / 3], [0.2, 0.2, 0.6]])
    rows, y, w = soft_rows3(q, np.array([True, True, False]), confidence=False)
    assert set(rows) == {0, 1} and np.isclose(w[(rows == 0) & (y == C)][0], 0.8)
    rows, y, w = soft_rows3(q, np.array([True, True, False]), confidence=True)
    assert pd.Series(w).groupby(rows).sum().get(1, 0.0) < 1e-6   # 균등 분포 → 신뢰도 0
    assert set(y) == {C, M, O}


def test_gates3_and_auto_confirm_guard():
    df = pd.DataFrame([
        {"model": "a", "probe_in_rate_drop_pp": 7, "anchor_in_rate_drop_pp": 1, "policy_to_exclude_rate": 0.005,
         "broadcast_excess_drop_pp": 1, "direction_flip_rate": 0.05, "include_change_rel": 0.1},
        {"model": "b", "probe_in_rate_drop_pp": 9, "anchor_in_rate_drop_pp": 1, "policy_to_exclude_rate": 0.005,
         "broadcast_excess_drop_pp": 1, "direction_flip_rate": 0.05, "include_change_rel": 0.45},
        {"model": "c", "probe_in_rate_drop_pp": 9, "anchor_in_rate_drop_pp": 3, "policy_to_exclude_rate": 0.005,
         "broadcast_excess_drop_pp": 1, "direction_flip_rate": 0.05, "include_change_rel": 0.1},
    ])
    g = gates3(df, 0.072, {}).set_index("model")
    assert g.loc["a", "pass_all"] and not g.loc["a", "auto_confirm_blocked"]
    assert g.loc["b", "pass_all"] and g.loc["b", "auto_confirm_blocked"]    # INCLUDE 30% 이상 변화 → 자동 확정 금지
    assert not g.loc["c", "pass_all"]


def test_selection_pointer_blocks_failed_and_unselected(tmp_path):
    from trend_radar.selection import SelectionError, mark_failed, selected_e2, selected_e3

    reg, out = tmp_path / "registry", tmp_path / "out"
    reg.mkdir()
    for rid in ("run_ok", "run_bad"):
        d = out / "runs" / rid
        d.mkdir(parents=True)
        pd.DataFrame({"gid": ["a"]}).to_parquet(d / "02_article_relevance.parquet")
        (d / "e2_summary.json").write_text(json.dumps({"otsu_threshold": 0.5}))
    (reg / "experiment_registry.jsonl").write_text(
        json.dumps({"task": "keyword_dictionary", "run_id": "e3_old", "input_e2_run_id": "run_bad"}) + "\n" +
        json.dumps({"task": "keyword_dictionary", "run_id": "e3_ok", "input_e2_run_id": "run_ok"}) + "\n")
    cfg = {"e2": {"selected_run_id": "run_ok", "selection_status": "approved", "candidate_run_id": None}}
    assert selected_e2(cfg, reg, out)[0] == "run_ok"
    assert selected_e3(reg, "run_ok") == "e3_ok"                      # 최신이 아니라 선정 E2에서 만든 E3
    with pytest.raises(SelectionError):
        selected_e2(cfg, reg, out, use_candidate=True)               # 후보 미지정
    mark_failed(reg, "run_bad", "e2_2", ["x"], "test")
    with pytest.raises(SelectionError):
        selected_e2({"e2": {"selected_run_id": "run_bad", "selection_status": "approved"}}, reg, out)
    with pytest.raises(SelectionError):
        selected_e3(reg, "run_none")
