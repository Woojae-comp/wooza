"""E2.2: 확률 라벨 복제 학습, 확신도 가중, 검색 기업명 처리, 출처 토큰 분리, 선정 기준."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_radar.e22 import agreement_stats, drop_search_companies, select, separate_source_tokens, soft_rows


def test_soft_rows_duplicate_with_probability_weights():
    p = np.array([0.9, 0.3, 0.5, 0.7])
    covered = np.array([True, True, True, False])            # 기권 기사는 학습 제외
    rows, y, w = soft_rows(p, covered, "softA")
    assert sorted(set(rows)) == [0, 1, 2]
    assert np.isclose(w[(rows == 1) & (y == 1)][0], 0.3) and np.isclose(w[(rows == 1) & (y == 0)][0], 0.7)


def test_soft_b_downweights_uncertain():
    p = np.array([0.95, 0.5, 0.3])
    rows, y, w = soft_rows(p, np.ones(3, bool), "softB")
    tot = pd.Series(w).groupby(rows).sum()
    assert tot.get(1, 0.0) < 1e-6                                   # p=0.5 → 엔트로피 1 → 가중 0 (학습 지배 방지)
    assert tot[0] > tot[2] > 0                              # 확신 높을수록 큰 가중, p=0.3도 학습에 들어감


def test_drop_search_company_unless_industry_action():
    arts = pd.DataFrame({"title": ["YTN 정치 인터뷰", "YTN 매각 본격화"], "summary": ["의원이 말했다", "지분 인수 경쟁"],
                         "companies": [["YTN"], ["YTN"]]})
    kw = [["YTN", "의원"], ["YTN", "지분"]]
    out, dropped = drop_search_companies(kw, arts, {}, {"매각", "인수"})
    assert out[0] == ["의원"] and dropped[0]                 # 기업명만 있고 산업 행위 없음 → 콘텐츠 근거 제외
    assert out[1] == ["YTN", "지분"] and not dropped[1]      # 매각·인수가 기사 주제 → 유지
    subj = pd.DataFrame({"title": ["SBS가 사과문 발표"], "summary": ["논란 이후 입장을 냈다"], "companies": [["SBS"]]})
    out2, d2 = drop_search_companies([["SBS", "사과문"]], subj, {}, {"매각"})
    assert out2[0] == ["SBS", "사과문"] and not d2[0]       # 기업이 문장 주체 → 유지


def test_separate_source_tokens():
    arts = pd.DataFrame({"title": ["[자막뉴스] 후보자 청문회", "SBS 드라마 편성"],
                         "summary": ["여야 공방. YTN 기자", "SBS가 새 드라마를 편성했다."], "companies": [["YTN"], ["SBS"]]})
    kw = [["자막뉴스", "후보자", "청문회", "YTN"], ["SBS", "드라마", "편성"]]
    out = separate_source_tokens(kw, arts, {"YTN": ["YTN"], "SBS": ["SBS"]})
    assert "자막뉴스" not in out[0] and "YTN" not in out[0] and "SRC_BROADCASTER" in out[0]
    assert out[1] == ["SBS", "드라마", "편성"]                # 방송사가 기사 대상이면 그대로


def test_agreement_stats_flip():
    p_clf = np.array([0.8, 0.2, 0.9, 0.1])
    p_lm = np.array([0.3, 0.1, 0.9, 0.6])
    s = agreement_stats(p_clf, 0.5, p_lm, np.ones(4, bool), np.array([True, False, False, False]))
    assert s["direction_flip_rate"] == 0.5 and s["probe_direction_flip_rate"] == 1.0
    assert s["lm_irrelevant_but_clf_relevant"] == 0.25


def test_select_requires_all_gates():
    df = pd.DataFrame([
        {"variant": "a", "probe_in_rate_drop_pp": 8, "anchor_in_rate_drop_pp": 1, "broadcast_excess_drop_pp": 1,
         "policy_to_exclude_rate": 0.001, "direction_flip_rate": 0.1, "base_flip_rate": 0.2, "relevant_rate_change_rel": -0.05},
        {"variant": "b", "probe_in_rate_drop_pp": 15, "anchor_in_rate_drop_pp": 4, "broadcast_excess_drop_pp": 1,
         "policy_to_exclude_rate": 0.001, "direction_flip_rate": 0.1, "base_flip_rate": 0.2, "relevant_rate_change_rel": -0.2},
    ])
    s = select(df, {})
    assert s.iloc[0]["variant"] == "a" and s.iloc[0]["pass_all"]
    assert not s.set_index("variant").loc["b", "gate_anchor"] and s.set_index("variant").loc["b", "relevant_rate_shift_flag"]
