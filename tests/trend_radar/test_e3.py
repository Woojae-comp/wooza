"""E3 핵심어 사전: 가중치, 로그오즈, 등급 경계, 구문 조각 제한, 보호 개체, 재현성."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_radar.config import Lexicon
from trend_radar.e3 import c_value, candidates, doc_term, e3_input, grade, phrase_quality_fail, weighted_log_odds
from trend_radar.text import Prep, tokenize_texts

QUALITY = {"time_words": ["이날", "오전", "올해", "하반기"], "form_words": ["홈페이지", "리포트"],
           "title_words": ["부장", "대표"]}


def test_e3_input_weights_and_review_reason():
    rel = pd.DataFrame({"gid": ["a", "b", "c"], "decision": ["INCLUDE", "REVIEW", "EXCLUDE"],
                        "p_ws_lr": [0.9, 0.55, 0.1], "seed_layer": [True, False, False], "p_labelmodel": [0.8, 0.6, 0.1]})
    x = e3_input(rel, "run_x", 0.49)
    assert list(x["content_weight_half"]) == [1.0, 0.5, 0.0]
    assert np.allclose(x["content_weight_soft"] + x["general_weight_soft"], 1)
    assert list(x["model_agreement_count"]) == [3, 2, 0]
    assert x.loc[1, "review_reason"] == "ws:+ seed:- lm:+" and x.loc[0, "review_reason"] == ""
    assert (x["e2_run_id"] == "run_x").all()


def test_weighted_log_odds_direction_and_null():
    from scipy import sparse

    # 단어0: 관련 기사에만, 단어1: 양쪽 고르게
    block = np.array([[1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1]], dtype=float)
    X = sparse.csr_matrix(np.tile(block, (200, 1)))   # 기사 1,200건 (사전분포 α0=1000이 신호를 덮지 않는 규모)
    rel = np.tile(np.array([1, 1, 1, 0, 0, 0], float), 200)
    z = weighted_log_odds(X, rel, 1 - rel)
    assert z[0] > 1.96 and z[1] < z[0]
    same = weighted_log_odds(X, np.ones(1200), np.ones(1200))
    assert np.allclose(same, 0, atol=1e-6)


def test_half_weight_counts_review_as_half():
    from scipy import sparse

    X = sparse.csr_matrix(np.array([[1], [1]], dtype=float))
    half = np.array([1.0, 0.5])
    assert float(np.asarray(X.T @ half).ravel()[0]) == 1.5


def test_parent_share_marks_fragment():
    vocab = ["역할", "역할 수행", "역할 수행 게임", "게임"]
    f = np.array([100.0, 60.0, 55.0, 500.0])
    cval, share = c_value(vocab, f)
    assert share[0] >= 0.5          # '역할'은 과반이 상위 복합어 안에서 쓰임 → 조각
    assert share[3] < 0.5           # '게임'은 독립적으로 쓰임
    assert cval[2] > 0


def test_phrase_quality_rules():
    assert phrase_quality_fail("이날 오전", QUALITY)
    assert phrase_quality_fail("공식 홈페이지", QUALITY)
    assert phrase_quality_fail("김 부장", QUALITY)
    assert not phrase_quality_fail("신작 출시", QUALITY)
    assert not phrase_quality_fail("하반기 신작", QUALITY)   # 시간어 + 개념어는 유지


def _grade(**kw):
    n = 7
    base = dict(zA=np.full(n, 5.0), zB=np.full(n, 5.0), zC=np.full(n, 5.0), shift_AB=np.zeros(n), shift_AC=np.zeros(n),
                persistence=np.ones(n), growth=np.ones(n), recent_df=np.full(n, 50.0), parent_share=np.zeros(n),
                is_stop=np.zeros(n, bool), market_phrase=np.zeros(n, bool), quality_fail=np.zeros(n, bool))
    base.update(kw)
    return grade(**base, rules={})


def test_grade_boundaries():
    n = 7
    zA = np.array([5, 5, 5, 5, 5, -1, 1.0])
    zB = np.array([5, 5, 5, 5, 5, -1, 3.0])
    zC = np.array([5, 5, 5, 5, 5, -1, 1.0])
    parent = np.array([0, 0.6, 0, 0, 0, 0, 0])
    market = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    qual = np.array([0, 0, 0, 1, 0, 0, 0], bool)
    stop = np.array([0, 0, 0, 0, 1, 0, 0], bool)
    cls, reason, _ = _grade(zA=zA, zB=zB, zC=zC, parent_share=parent, market_phrase=market, quality_fail=qual, is_stop=stop)
    assert cls[0] == "CORE"
    assert cls[1] == "REVIEW" and "복합어" in reason[1]       # 조각은 최대 REVIEW
    assert cls[2] == "REVIEW" and "시장" in reason[2]         # 시장 구문은 최대 REVIEW
    assert cls[3] == "DROP" and "품질" in reason[3]
    assert cls[4] == "DROP" and "불용어" in reason[4]
    assert cls[5] == "DROP"                                   # 세 방식 모두 음수
    assert cls[6] in ("REVIEW", "EXTENDED")                  # B만 유의 → A·B 불일치


def test_grade_emerging_needs_growth_and_volume():
    n = 7
    zA = np.full(n, 0.5); zB = np.full(n, 0.5); zC = np.full(n, 0.5)
    growth = np.array([3, 3, 1, 3, 3, 3, 3.0])
    recent = np.array([20, 5, 20, 20, 20, 20, 20.0])
    cls, _, _ = _grade(zA=zA, zB=zB, zC=zC, growth=growth, recent_df=recent)
    assert cls[0] == "EMERGING"
    assert cls[1] != "EMERGING" and cls[2] != "EMERGING"


def test_protected_entity_restored_and_not_fragmented():
    names = ["P의 거짓", "네오위즈"]
    text = Prep(names)("네오위즈 P의 거짓 흥행 신작 출시")
    toks = tokenize_texts([text], names)
    restore = {"P의거짓": "P의 거짓"}
    terms = candidates(toks, [text], Lexicon(), {"형", "화"}, restore=restore)[0]
    assert "P의 거짓" in terms
    assert "거짓" not in terms


def test_candidates_reproducible():
    names = ["네오위즈"]
    texts = [Prep(names)(t) for t in ["네오위즈 신작 출시 글로벌 진출", "생성형 AI 콘텐츠 제작 확대"]]
    toks = tokenize_texts(texts, names)
    a = candidates(toks, texts, Lexicon(), {"형"})
    b = candidates(toks, texts, Lexicon(), {"형"})
    assert a == b
    X1, v1 = doc_term(a, 1)
    X2, v2 = doc_term(b, 1)
    assert v1 == v2 and (X1 != X2).nnz == 0
    assert "신작 출시" in a[0] and "생성형 AI" in a[1]
