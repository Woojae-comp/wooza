"""E2 비교·검토: 비례 배분, 묶음 단위 분할(누수 없음), 응답 파싱·검증, 근거 문구 대조, 병합·우선 검토."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trend_radar.review import (allocate, draw, duplicate_groups, evidence_found, merge_responses, norm_title,
                                parse_response, split_by_group, validate)


def test_allocate_proportional_largest_remainder():
    a = allocate({"a": 500, "b": 300, "c": 200}, 10)
    assert a == {"a": 5, "b": 3, "c": 2} and sum(allocate({"a": 1, "b": 1, "c": 1}, 200, cap=False).values()) == 200
    assert allocate({"a": 1, "b": 5}, 4) == {"a": 1, "b": 3}      # 모집단보다 많이 배분하지 않음


def test_duplicate_groups_titles_and_group_ids():
    gids = pd.Series(["1", "2", "3", "4"])
    titles = pd.Series(["[단독] 넷마블 신작 출시", "넷마블 신작 출시!", "크래프톤 실적", "크래프톤 실적 발표"])
    g = duplicate_groups(gids, titles, pd.Series({"3": "G9", "4": "G9"}))
    assert g[0] == g[1]                        # 말머리·기호만 다른 재전송 → 같은 묶음
    assert g[2] == g[3] and g[0] != g[2]       # 원자료 중복기사군
    assert norm_title("[속보] A-B (종합)") == "ab"


def test_draw_and_split_never_cross_groups():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"gid": [str(i) for i in range(200)], "group_id": [f"g{i // 3}" for i in range(200)]})
    s, n_avail = draw(df, 40, rng, set())
    assert s["group_id"].is_unique and len(s) == 40 and n_avail == 200
    s2, n2 = draw(df, 10, rng, set(s["group_id"]))
    assert not set(s2["group_id"]) & set(s["group_id"]) and n2 < 200
    both = pd.concat([s, s2, df[df["group_id"] == s["group_id"].iat[0]]]).drop_duplicates("gid")
    split = split_by_group(both, 20, rng)
    assert (both.assign(split=split).groupby("group_id")["split"].nunique() == 1).all()


def test_parse_response_ignores_prose_and_code_fences():
    text = "검토 결과입니다.\n```\narticle_id\tcontent_relevance\tmarket_focus\treview_class\tcontent_evidence\texclusion_evidence\treason_short\tneeds_more_context\n" \
           "a1\tSUBSTANTIVE\tNO\tCONTENT\t신작 출시\t\t출시 정보\tNO\na2\tNONE\tYES\tMARKET\t\t목표주가\t시세\tNO\n```\n처리 완료."
    df, errs = parse_response(text)
    assert errs == [] and list(df["article_id"]) == ["a1", "a2"]
    bad, errs2 = parse_response("article_id\treview_class\na1\tCONTENT")
    assert any("필수 열" in e for e in errs2)


def test_evidence_found_normalization():
    src = "넷마블, ‘세븐나이츠’ 신작  출시… 글로벌 흥행"
    assert evidence_found("'세븐나이츠' 신작 출시", src) is True
    assert evidence_found("신작 출시 / 글로벌 흥행", src) is True
    assert evidence_found("해외 매출 급증", src) is False
    assert evidence_found("", src) is None


def test_validate_flags_invalid_unknown_duplicate():
    df = pd.DataFrame([{"article_id": "a1", "content_relevance": "substantive", "market_focus": "NO", "review_class": "CONTENT",
                        "content_evidence": "없는 문구", "exclusion_evidence": "", "reason_short": "", "needs_more_context": "NO"},
                       {"article_id": "zz", "content_relevance": "MAYBE", "market_focus": "NO", "review_class": "OTHER",
                        "content_evidence": "", "exclusion_evidence": "", "reason_short": "", "needs_more_context": "NO"}])
    v = validate(df, {"a1", "a2"}, {"a1": "넷마블 신작 출시"})
    assert v.loc[0, "content_relevance"] == "SUBSTANTIVE" and not v.loc[0, "invalid_content_relevance"]
    assert v.loc[0, "evidence_problem"] and v.loc[1, "unknown_id"] and v.loc[1, "invalid_content_relevance"]


def test_merge_responses_end_to_end(tmp_path):
    d = tmp_path / "review_x"
    (d / "upload" / "stage1").mkdir(parents=True)
    (d / "internal").mkdir()
    (d / "model_responses").mkdir()
    pd.DataFrame({"article_id": ["a1", "a2", "a3"], "title": ["넷마블 신작 출시", "주가 급등 종목", "여야 청문회 공방"],
                  "summary": ["글로벌 출시 일정 공개", "목표주가 상향", "YTN 라디오 출연"]}) \
        .to_csv(d / "upload" / "stage1" / "review_input_batch_001.tsv", sep="\t", index=False)
    pd.DataFrame({"gid": ["a1", "a2", "a3"], "sample_set": "representative", "reason": "x", "split": "dev", "batch": "batch_001",
                  "has_sid": ["True", "False", "False"], "e21_decision": ["INCLUDE", "EXCLUDE", "REVIEW"], "weight": "1",
                  "cand::E2.1_operational": ["INCLUDE", "EXCLUDE", "REVIEW"], "flag_content_policy": "False",
                  "flag_mixed_content_market": "False"}).to_csv(d / "internal" / "sample_manifest.csv", index=False)
    h = "article_id\tcontent_relevance\tmarket_focus\treview_class\tcontent_evidence\texclusion_evidence\treason_short\tneeds_more_context\n"
    (d / "model_responses" / "batch_001__m1.tsv").write_text(h + "a1\tSUBSTANTIVE\tNO\tCONTENT\t신작 출시\t\tx\tNO\n"
                                                             "a2\tNONE\tYES\tMARKET\t\t목표주가\tx\tNO\na3\tNONE\tNO\tOTHER\t\t청문회\tx\tNO\n", encoding="utf-8")
    (d / "model_responses" / "batch_001__m2.tsv").write_text(h + "a1\tSUBSTANTIVE\tYES\tCONTENT\t신작 출시\t\tx\tNO\n"
                                                             "a2\tINCIDENTAL\tYES\tMARKET\t\t급등\tx\tNO\n", encoding="utf-8")   # a3 누락
    r = merge_responses(d)
    assert r["models"] == ["m1", "m2"] and r["issues"] >= 1
    W = pd.read_csv(d / "merged" / "review_wide.csv", dtype=str).set_index("article_id")
    assert W.loc["a1", "consensus_class"] == "CONTENT" and W.loc["a2", "consensus_class"] == "MARKET"
    issues = pd.read_csv(d / "merged" / "parse_issues.csv")
    assert issues["issue"].str.contains("누락 ID a3").any()
    assert (d / "e2_llm_comparison_report.md").read_text(encoding="utf-8").count("LLM 참고 판정") >= 1


def test_v10_responses_without_content_type_still_merge():
    text = "article_id\tcontent_relevance\tmarket_focus\treview_class\tcontent_evidence\texclusion_evidence\treason_short\tneeds_more_context\n" \
           "a1\tSUBSTANTIVE\tNO\tCONTENT\t신작\t\tx\tNO\n"
    df, errs = parse_response(text)
    assert errs == [] and df.loc[0, "content_type"] == ""
    text11 = "article_id\tcontent_relevance\tmarket_focus\treview_class\tcontent_type\tcontent_evidence\texclusion_evidence\treason_short\tneeds_more_context\n" \
             "a1\tSUBSTANTIVE\tNO\tCONTENT\tEVENT_ISSUE\t압수수색\t\tx\tNO\n"
    df2, errs2 = parse_response(text11)
    v = validate(df2, {"a1"}, {"a1": "컴투스 압수수색"})
    assert errs2 == [] and not v.loc[0, "invalid_content_type"]
