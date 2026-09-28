"""E4.1: 노드 유형 순서, Leiden 기사 배정(분할·저신뢰·미배정), 규모 보정, 분야 가중, 분야 포착, 통과 기준, 주제 유형, 캐시."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from trend_radar.e4 import cache_key, cached_edges, npmi_edges
from trend_radar.e41 import (assign_leiden, community_matrix, gates, node_types, sector_capture, sector_weights,
                             topic_type)

RULES = {"min_keywords": 2, "min_coverage": 0.3, "margin_min": 0.1, "split_min_mass": 0.25}


def _dic(rows):
    return pd.DataFrame(rows, columns=["keyword", "entity_type", "class_auto", "z_B_half", "doc_freq"])


def test_node_types_entity_first_then_market_then_general():
    dic = _dic([
        ("넷마블", "기업", "CORE", 1.0, 900),          # 콘텐츠 기업 (빈도·특이도와 무관하게 먼저)
        ("SBS", "기업", "EXTENDED", 0.5, 800),         # 연관산업만 → 비콘텐츠 기업
        ("붉은사막", "작품·개체", "CORE", 9.0, 300),
        ("목표주가", "개념", "EXTENDED", 0.1, 700),     # 시장 단어
        ("실적 개선", "개념", "EXTENDED", 0.5, 600),    # 시장 앵커 z 유의 + 콘텐츠 비유의
        ("사업", "개념", "EXTENDED", 0.2, 5000),       # 저특이도 일반어
        ("게임", "개념", "CORE", 30.0, 9000),          # 고빈도 콘텐츠 개념 → 유지
        ("신작 출시", "개념", "CORE", 12.0, 400),
    ])
    ref = pd.concat([dic, _dic([(f"w{i}", "개념", "EXTENDED", float(i % 7), i) for i in range(40)])])
    z_market = np.array([0, 0, 0, 5.0, 4.0, 0.5, 0.2, -2.0])
    nt = node_types(dic, z_market, {"목표주가"}, {"넷마블": {"게임"}, "SBS": {"연관산업"}}, {"게임"}, ref, {})
    t = dict(zip(nt["keyword"], nt["node_type"]))
    assert t["넷마블"] == "content_company" and t["SBS"] == "noncontent_company"
    assert t["붉은사막"] == "work_person_policy"
    assert t["목표주가"] == "market_expression" and t["실적 개선"] == "market_expression"
    assert t["사업"] == "general_low_specificity"
    assert t["게임"] == "concept" and t["신작 출시"] == "concept"
    keep = nt.set_index("keyword")
    assert not keep.loc["넷마블", "concept_keep"] and keep.loc["넷마블", "extended_keep"]      # 콘텐츠 기업: 개념 제외, 확장 유지
    assert not keep.loc["SBS", "extended_keep"]
    assert keep.loc["게임", "concept_keep"] and keep.loc["게임", "extended_keep"]


def test_node_types_proper_noun_split_by_specificity():
    from kiwipiepy import Kiwi

    dic = _dic([("LG전자", "개념", "EXTENDED", 0.3, 200), ("김범수", "개념", "EXTENDED", 3.0, 150),
                ("박진영", "개념", "EXTENDED", 0.8, 120), ("교보증권", "개념", "EXTENDED", 0.1, 90)])
    nt = node_types(dic, np.array([2.5, 0.5, 1.0, 3.0]), {"증권"}, {}, set(), dic, {}, kiwi=Kiwi())
    t = dict(zip(nt["keyword"], nt["node_type"]))
    assert t["LG전자"] == "proper_noun_noncontent"      # 형태(NNP) + 콘텐츠 비유의 + 시장 쪽
    assert t["김범수"] == "proper_noun_content"
    assert t["박진영"] == "proper_noun_content"         # 콘텐츠 특이도가 낮아도 시장 쪽이 유의하지 않으면 콘텐츠
    assert t["교보증권"] == "market_expression"          # 붙여 쓴 복합어의 시장 단어 끝


def test_community_matrix_size_normalized():
    memb = np.array([0, 0, 0, 0, 1, -1])
    M, Mb, ids = community_matrix(memb)
    assert list(ids) == [0, 1]
    assert np.isclose(M[0, 0], 0.5) and np.isclose(M[4, 1], 1.0)   # 1/√4, 1/√1
    assert M[5].nnz == 0                                            # 고립 노드는 배정에 쓰지 않음
    assert Mb.sum() == 5


def test_assign_leiden_high_split_low_unassigned():
    memb = np.array([0, 0, 1, 1, -1])
    V = sparse.csr_matrix(np.array([
        [0.9, 0.4, 0.0, 0.0, 0.0],     # 공동체 0만 → HIGH
        [0.6, 0.0, 0.6, 0.0, 0.0],     # 두 공동체 반반 → SPLIT
        [0.1, 0.0, 0.0, 0.0, 0.99],    # 대부분 고립 노드 → LOW (커버리지)
        [0.0, 0.0, 0.0, 0.0, 1.0],     # 공동체 핵심어 없음 → UNASSIGNED
        [0.9, 0.4, 0.0, 0.0, 0.0],     # 핵심어 1개 → LOW
    ]))
    nkw = np.array([2, 2, 2, 1, 1])
    a = assign_leiden(V, memb, nkw, RULES)
    assert list(a["assignment_confidence"]) == ["HIGH", "SPLIT", "LOW", "UNASSIGNED", "LOW"]
    assert np.isclose(a.loc[1, "top_weight"] + a.loc[1, "second_weight"], 1)
    assert a.loc[0, "second_weight"] == 0 and a.loc[3, "top_community"] == -1


def test_large_community_not_favored():
    # 기사 핵심어 1개가 큰 공동체(10개 노드), 1개가 작은 공동체(2개 노드)에 속할 때 작은 공동체가 이긴다 (규모 보정)
    memb = np.array([0] * 10 + [1] * 2)
    v = np.zeros(12); v[0] = 0.7; v[10] = 0.7
    a = assign_leiden(sparse.csr_matrix(v), memb, np.array([2]), RULES)
    assert a.loc[0, "top_community"] == 1


def test_sector_weights_sqrt_inverse():
    w = sector_weights([["게임"]] * 100 + [["만화"]] * 4)
    assert np.isclose(w.mean(), 1)
    assert np.isclose(w[-1] / w[0], np.sqrt(100 / 4))


def test_sector_capture_counts_enriched_topics():
    # 분야 1(소규모)이 주제 1에 몰리면 포착, 흩어지면 0
    Ssec = np.array([[1, 0]] * 8 + [[0, 1]] * 2, float)
    w = np.ones(10)
    lab = np.array([0] * 8 + [1] * 2)
    cap = sector_capture(lab, np.ones(10, bool), w, Ssec)
    assert np.isclose(cap[1], 1.0)
    lab2 = np.array([0] * 4 + [1] * 4 + [0, 1])
    assert sector_capture(lab2, np.ones(10, bool), w, Ssec)[1] == 0


def test_gates_require_improvement_over_baseline():
    base = pd.Series({"seed_ari": 0.27, "small_sector_capture_min": 0.1, "general_topic_share": 0.2,
                      "noise_exclude_highconf": 0.5, "noise_probe_highconf": 0.5})
    m = pd.DataFrame([
        {"model": "good", "seed_ari": 0.9, "largest_share": 0.05, "low_or_unassigned_rate": 0.2, "small_sector_capture_min": 0.3,
         "general_topic_share": 0.1, "noise_exclude_highconf": 0.2, "noise_probe_highconf": 0.3},
        {"model": "unstable", "seed_ari": 0.4, "largest_share": 0.05, "low_or_unassigned_rate": 0.2, "small_sector_capture_min": 0.3,
         "general_topic_share": 0.1, "noise_exclude_highconf": 0.2, "noise_probe_highconf": 0.3},
    ])
    g = gates(m, base, {}).set_index("model")
    assert g.loc["good", "pass_all"] and not g.loc["unstable", "pass_all"]
    assert not g.loc["unstable", "gate_stability"]


def test_topic_type_priority():
    cs = {"게임": 0.9, "음악": 0.1}
    assert topic_type(0.5, 0.8, False, 0, 0.1, cs, {}, 0.49)[0] == "NOISE_CANDIDATE"
    assert topic_type(0.0, 0.8, False, 3, 0.9, cs, {}, 0.49)[0] == "POLICY_EVENT"
    assert topic_type(0.0, 0.8, False, 0, 0.9, cs, {}, 0.49)[0] == "ENTITY_EVENT"
    assert topic_type(0.0, 0.8, False, 0, 0.2, {"게임": 0.5, "음악": 0.4, "만화": 0.1}, {}, 0.49)[0] == "CROSS_SECTOR"
    primary, flags = topic_type(0.0, 0.8, False, 0, 0.9, {"게임": 0.5, "음악": 0.5}, {}, 0.49)
    assert primary == "ENTITY_EVENT" and "CROSS_SECTOR" in flags        # 기업 편중이어도 교차 분야 표시는 보존
    assert topic_type(0.0, 0.8, False, 0, 0.2, cs, {}, 0.49)[0] == "INDUSTRY_TOPIC"


def test_edge_cache_key_and_reuse(tmp_path):
    key = {"data_snapshot_id": "ds_x", "dictionary": "run_a", "layer": "extended", "unit": "article", "min_co": 5, "npmi": 0.1, "topk": 15}
    assert cache_key(key) != cache_key({**key, "npmi": 0.2}) and cache_key(key) != cache_key({**key, "dictionary": "run_b"})
    B = sparse.csr_matrix(np.array([[1, 1, 0]] * 20 + [[0, 1, 1]] * 20, float))
    calls = []

    def build():
        calls.append(1)
        return npmi_edges(B, np.ones(40), 5, 0.0, 15)

    e1 = cached_edges(tmp_path, key, build)
    e2 = cached_edges(tmp_path, key, build)
    assert len(calls) == 1 and e1.equals(e2)


def test_general_words_use_effect_size_not_z():
    # '사업'은 z가 커도(빈도 때문) 효과 크기가 작으면 일반어, '신작'은 효과 크기가 커서 개념어
    dic = pd.DataFrame({"keyword": ["사업", "신작"] + [f"w{i}" for i in range(18)], "entity_type": "개념", "class_auto": "CORE",
                        "z_B_half": [25.0, 20.0] + [3.0] * 18, "delta_B": [0.05, 1.5] + list(np.linspace(0.1, 1.0, 18)),
                        "doc_freq": [9000, 8000] + list(range(10, 28))})
    nt = node_types(dic, np.zeros(20), set(), {}, set(), dic, {})
    t = dict(zip(nt["keyword"], nt["node_type"]))
    assert t["사업"] == "general_low_specificity" and t["신작"] == "concept"
