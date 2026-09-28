"""E4 주제 군집·네트워크: NPMI 연결 기준, Leiden 재현성, 문장 분할, 일반어 분위수, 군집 안정성, 선택 규칙."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from trend_radar.e4 import (best_match_jaccard, borda, centroid_similarity, community_metrics, general_words, leiden,
                            npmi_edges, sentence_spans, tfidf)


def _two_blocks(n=200):
    # 단어 0·1은 항상 함께, 2·3은 항상 함께, 두 묶음은 서로 다른 기사
    rows = [[1, 1, 0, 0]] * n + [[0, 0, 1, 1]] * n + [[1, 0, 0, 1]] * 3
    return sparse.csr_matrix(np.array(rows, float))


def test_npmi_edges_threshold_and_min_co():
    B = _two_blocks()
    e = npmi_edges(B, np.ones(B.shape[0]), min_co=5, min_npmi=0.1, topk=15)
    pairs = set(zip(e["i"], e["j"]))
    assert (0, 1) in pairs and (2, 3) in pairs
    assert (0, 3) not in pairs                     # 동시출현 3건 < 5
    assert (e["npmi"] <= 1 + 1e-9).all()


def test_npmi_edges_topk_limits_degree():
    rng = np.random.default_rng(0)
    B = sparse.csr_matrix((rng.random((2000, 30)) < 0.3).astype(float))
    e = npmi_edges(B, np.ones(2000), min_co=5, min_npmi=-1, topk=3)
    deg = np.bincount(np.r_[e["i"], e["j"]], minlength=30)
    # 어느 한쪽의 상위 3에 들면 유지되므로 최대 차수는 제한되지만 3을 조금 넘을 수 있다
    assert len(e) <= 30 * 3 and deg.max() < 30 - 1


def test_review_weight_changes_npmi_not_min_co():
    B = _two_blocks()
    w = np.ones(B.shape[0]); w[:200] = 0.5        # 첫 묶음이 REVIEW 기사
    e = npmi_edges(B, w, min_co=5, min_npmi=0.0, topk=15)
    assert (0, 1) in set(zip(e["i"], e["j"]))     # 최소 동시출현은 가중 전 기사 수


def test_leiden_reproducible_and_recovers_blocks():
    B = _two_blocks()
    e = npmi_edges(B, np.ones(B.shape[0]), 5, 0.1, 15)
    m1, q1 = leiden(4, e, 1.0, seed=3)
    m2, q2 = leiden(4, e, 1.0, seed=3)
    assert (m1 == m2).all() and q1 == q2
    assert m1[0] == m1[1] and m1[2] == m1[3] and m1[0] != m1[2]
    met = community_metrics([m1, m2], [q1, q2])
    assert met["seed_ari"] == 1.0 and met["isolated_rate"] == 0


def test_isolated_nodes_marked():
    e = pd.DataFrame({"i": [0], "j": [1], "npmi": [0.5], "co_units": [10]})
    m, _ = leiden(3, e, 1.0, 0)
    assert m[2] == -1 and m[0] >= 0


def test_sentence_spans():
    t = "제목 한 줄\n첫 문장이다. 둘째 문장! 셋째"
    s = [t[a:b] for a, b in sentence_spans(t)]
    assert s == ["제목 한 줄", "첫 문장이다.", "둘째 문장!", "셋째"]


def test_general_words_by_quantile():
    n = 20
    dic = pd.DataFrame({"keyword": [f"w{i}" for i in range(n)] + ["넷마블"], "entity_type": ["개념"] * n + ["기업"],
                        "z_B_half": list(range(n)) + [0], "doc_freq": list(range(n))[::-1] + [10_000]})
    g = general_words(dic, {})
    assert g == {"w0", "w1", "w2"}                  # 특이도 하위 절반이면서 문서빈도 상위 10% (20개 중 3개)
    assert "넷마블" not in g                         # 개체는 대상 아님


def test_best_match_jaccard_identical_and_relabelled():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert np.allclose(best_match_jaccard(a, a), 1)
    assert np.allclose(best_match_jaccard(a, np.array([5, 5, 7, 7, 9, 9])), 1)   # 번호만 다름
    assert best_match_jaccard(a, np.array([0, 1, 0, 1, 0, 1])).max() < 1


def test_borda_prefers_balanced():
    df = pd.DataFrame({"q": [0.9, 0.5, 0.6], "ari": [0.2, 0.9, 0.8], "big": [0.5, 0.1, 0.1]})
    r = borda(df, ["q", "ari"], ["big"])
    assert r.idxmin() in (1, 2) and r.idxmax() == 0


def test_tfidf_reuses_idf_and_centroid_similarity():
    X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], float))
    V, idf = tfidf(X)
    V2, idf2 = tfidf(X[:1], idf)
    assert np.allclose(idf, idf2) and np.allclose(V[:1].toarray(), V2.toarray())
    c = np.asarray(V.mean(0))
    sim = centroid_similarity(V, np.vstack([c, c]), np.array([0, 0, 1]))
    assert np.all((sim > 0) & (sim <= 1 + 1e-9))
