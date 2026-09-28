"""E2.1 잡음 유입 보정 라벨 함수 회귀 테스트 (사용자 지정 사례). 규칙 수준의 투표 방향을 고정한다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_radar.e2 import ABSTAIN, NEG, lf_broadcaster_source_only, lf_politics_society_section, strip_source_mentions

BRO = {"YTN": ["YTN"], "SBS": ["SBS", "SBS Biz"]}
TOPIC = {"드라마", "제작", "편성", "민영화", "매각", "방송법", "콘텐츠", "OTT", "시청률"}
POLICY = {"방송법", "OTT", "저작권", "문체부", "민영화"}

CASES = [
    # (제목, 요약, 검색 기업, 섹션, 콘텐츠 앵커, 기대: 방송사 규칙, 기대: 섹션 규칙)
    ("[자막뉴스] 대통령 후보자 청문회 공방", "여야가 후보자 자질을 두고 격돌했다. YTN 기자", ["YTN"], "100", False, NEG, NEG),
    ("與 의원 \"특검 불가피\"", "YTN 라디오 '뉴스킹'에 출연해 정치 현안을 두고 이같이 말했다.", ["YTN"], "100", False, NEG, NEG),
    ("YTN 민영화와 방송산업 구조 변화", "YTN 매각 이후 보도 채널 구조가 바뀌고 있다.", ["YTN"], "102", False, ABSTAIN, ABSTAIN),
    ("SBS 새 드라마 제작 발표", "SBS가 하반기 새 드라마 편성을 발표했다.", ["SBS"], None, False, ABSTAIN, ABSTAIN),
    ("방송법 개정안 국회 통과", "방송법 개정안이 본회의를 통과했다. SBS 뉴스", ["SBS"], "100", False, ABSTAIN, ABSTAIN),
    ("정부, OTT 규제 강화 추진", "OTT 사업자 규제를 강화하는 방안이 나왔다.", ["SBS"], "100", False, ABSTAIN, ABSTAIN),
    ("가수 A씨 음주운전 적발", "경찰에 따르면 A씨는 새벽 음주 상태로 운전하다 적발됐다.", ["SBS"], "102", False, NEG, NEG),
]


def _frame():
    return pd.DataFrame([{"title": t, "summary": s, "companies": c} for t, s, c, *_ in CASES])


def test_broadcaster_source_only_cases():
    arts = _frame()
    anc = np.array([c[4] for c in CASES])
    got = lf_broadcaster_source_only(arts, BRO, TOPIC, anc)
    assert list(got) == [c[5] for c in CASES]


def test_politics_society_section_cases():
    texts = [f"{c[0]} {c[1]}" for c in CASES]
    got = lf_politics_society_section(np.array([c[3] for c in CASES], dtype=object), np.array([c[4] for c in CASES]),
                                      texts, POLICY, {"100", "102"})
    assert list(got) == [c[6] for c in CASES]


def test_missing_section_abstains_and_anchor_blocks():
    got = lf_politics_society_section(np.array([None, "100"], dtype=object), np.array([False, True]),
                                      ["정치 기사", "정치 기사"], POLICY, {"100", "102"})
    assert list(got) == [ABSTAIN, ABSTAIN]          # 섹션 없음 → 기권, 콘텐츠 앵커 → 기권


def test_name_alone_is_not_source_only():
    # 단순히 방송사명이 들어갔다는 이유로 제외하지 않는다: 주어·대상으로 쓰이면 기권
    arts = pd.DataFrame([{"title": "SBS, 2분기 영업손실", "summary": "SBS의 광고 매출이 줄었다.", "companies": ["SBS"]}])
    assert lf_broadcaster_source_only(arts, BRO, TOPIC, np.array([False]))[0] == ABSTAIN


def test_strip_source_mentions():
    t = strip_source_mentions("[자막뉴스] 여야 격돌 (서울=YTN) YTN 라디오 출연 ⓒYTN YTN '뉴스킹' 사진=SBS 캡처", ["YTN", "SBS"])
    assert "YTN" not in t and "SBS" not in t
