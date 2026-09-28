"""E4.2: 연결어·복합어 처리, 배정 거부 경계, 잡음 점수·사유, 하위 주제 선택 규칙, 층화 복구 조건."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from trend_radar.e42 import (apply_bridge, classify_assignment, is_noise, noise_components, noise_reasons, noise_score,
                             pick_k, rejection_bounds, rescue_flags, subtopic_gate, top2_similarity)

RULES = {}


def test_bridge_single_word_and_compounds():
    nt = pd.DataFrame({"keyword": ["AI", "플랫폼", "생성형 AI", "팬덤 플랫폼", "사업", "AI 에이전트"],
                       "node_type": ["general_low_specificity", "general_low_specificity", "general_low_specificity",
                                     "general_low_specificity", "general_low_specificity", "concept"],
                       "reason": [""] * 6})
    out = dict(zip(*apply_bridge(nt, {"AI", "플랫폼", "방송"})[["keyword", "node_type"]].T.to_numpy()))
    assert out["AI"] == "generic_bridge" and out["플랫폼"] == "generic_bridge"
    assert out["생성형 AI"] == "concept" and out["팬덤 플랫폼"] == "concept"      # 복합어는 개념 노드
    assert out["사업"] == "general_low_specificity"                               # 연결어가 아닌 일반어는 그대로
    assert out["AI 에이전트"] == "concept"


def test_rejection_boundary_per_cluster():
    # 군집 0은 조밀(0.9 근처), 군집 1은 느슨(0.5 근처) → 경계가 군집마다 다르다
    lab = np.array([0] * 20 + [1] * 20)
    sim = np.r_[np.linspace(0.85, 0.95, 20), np.linspace(0.3, 0.7, 20)]
    b = rejection_bounds(sim, lab, 0.05)
    assert b[0] > 0.8 and b[1] < 0.35
    conf = classify_assignment(np.array([0, 1, 0]), np.array([0.6, 0.6, 0.9]), np.array([0.2, 0.2, 0.001]), b, 0.01)
    assert list(conf) == ["REJECTED", "HIGH", "LOW"]     # 같은 유사도라도 군집 0에선 거부, 군집 1에선 수용; 차이 작으면 LOW


def test_top2_similarity_normalizes_centers():
    Z = np.array([[1.0, 0.0], [0.6, 0.8]])
    C = np.array([[10.0, 0.0], [0.0, 3.0]])
    t1, s1, t2, s2 = top2_similarity(Z, C)
    assert list(t1) == [0, 1] and np.isclose(s1[0], 1) and np.isclose(s2[1], 0.6)


def _comp(**kw):
    base = dict(generic_share=0.1, market_phrase_share=0.0, external_entity_share=0.0, content_concept_share=0.6,
                bridge_share=0.05, market_article_share=0.1, content_anchor_share=0.2, entity_list_share=0.0)
    base.update(kw)
    return base


def test_noise_score_and_reasons_theme_stock_list():
    # 테마주 나열: 외부 기업·종목명 많고, 시장 앵커 기사 많고, 콘텐츠 개념어 적고, 나열 기사 반복
    c = _comp(external_entity_share=0.45, content_concept_share=0.2, market_article_share=0.6, entity_list_share=0.8,
              content_anchor_share=0.0)
    s = noise_score(c)
    r = noise_reasons(c, 0.7, 0.3, 0.0, RULES, 0.49, 0.05)
    assert s > 0 and "EXTERNAL_ENTITY_LIST" in r and "MARKET_ARTICLE_DOMINANT" in r
    assert is_noise(s, r, RULES)


def test_content_topic_not_noise_even_with_generic_words():
    c = _comp(generic_share=0.25, content_concept_share=0.55, content_anchor_share=0.3)
    s = noise_score(c)
    r = noise_reasons(c, 0.8, 0.3, 0.0, RULES, 0.49, 0.05)
    assert s < 0 and r == [] and not is_noise(s, r, RULES)


def test_single_soft_reason_is_not_noise():
    c = _comp()
    r = noise_reasons(c, 0.8, 0.01, 0.0, RULES, 0.49, 0.05)       # 일관성만 낮음
    assert r == ["LOW_COHERENCE"] and not is_noise(noise_score(c), r, RULES)


def test_noise_components_weighted_shares():
    Xd = sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], float))
    types = np.array(["concept", "general_low_specificity", "proper_noun_noncontent"])
    c = noise_components(np.array([0, 1]), np.array([1.0, 1.0]), Xd, types, np.array([1.0, 0]), np.array([0, 1.0]), np.array([0, 1.0]))
    assert np.isclose(c["content_concept_share"], 0.25) and np.isclose(c["generic_share"], 0.5)
    assert np.isclose(c["external_entity_share"], 0.25) and np.isclose(c["market_article_share"], 0.5)


def test_pick_k_smallest_within_tolerance():
    sils = {2: 0.100, 3: 0.105, 4: 0.108, 8: 0.109}
    assert pick_k(sils, 0.01) == 2
    assert pick_k(sils, 0.003) == 4


def test_subtopic_gate_rules():
    ok, _ = subtopic_gate(0.8, np.array([100.0, 80.0]), 0.08, RULES)
    assert ok
    assert not subtopic_gate(0.5, np.array([100.0, 80.0]), 0.08, RULES)[0]          # 시드 ARI
    assert not subtopic_gate(0.8, np.array([500.0, 20.0]), 0.08, RULES)[0]          # 최소 기사 수
    assert not subtopic_gate(0.8, np.array([3000.0, 40.0]), 0.08, RULES)[0]         # 최소 비중 0.03
    assert not subtopic_gate(0.8, np.array([100.0, 80.0]), 0.01, RULES)[0]          # 일관성 개선


def test_rescue_requires_low_similarity_size_and_coherence():
    f = rescue_flags(np.array([0.2, 0.6, 0.2, 0.2]), np.array([50, 50, 10, 50]), np.array([0.3, 0.3, 0.3, 0.05]), RULES, 0.1)
    assert list(f) == [True, False, False, False]
