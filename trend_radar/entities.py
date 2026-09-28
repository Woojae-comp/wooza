"""보호 개체 사전 (E3.1): 작품명·정책명 등을 형태소 분석 전에 한 토큰으로 묶는다.

'P의 거짓'처럼 조사가 끼거나 띄어 쓴 이름은 형태소 분석에서 쪼개져 '거짓' 같은 조각이 독립 핵심어가 된다.
불용어로 막지 않고, 탐지 → 보호 토큰 치환 → 형태소 분석·구문 추출 → 원래 표기로 복원 순서로 처리한다.

탐지 기준 (데이터에서 계산, 사람 판정 없음)
- 제목 따옴표 표현 (인용문 형식 제외)
- 3건 이상 기사에 등장
- 한 기업 집중도 0.5 이상 (강조 표현 '껑충'·'급등'은 여러 기업에 퍼져 0.1~0.4)
- 시장 구문 단어를 포함하지 않음
"""
from __future__ import annotations

import collections
import re

import pandas as pd

from .text import _space_free

QUOTE = re.compile(r"[‘'\"“《〈『「<]([^’'\"”》〉』」>]{2,25})[’'\"”》〉』」>]")


GENERIC_TAGS = {"NNG", "NNB", "NR", "XSN"}


def entity_status(name: str, kiwi) -> tuple[bool, bool]:
    """(개체로 표시할지, 보호 토큰이 필요한지).
    - 명사가 없는 말(뛰어), 일반명사로만 된 말(흥행, 탈세 의혹, 역사적 저점) → 개체 아님 (1~3그램 후보로 잡힘)
    - 한 형태소 고유명사·외국어(니케, BTS) → 개체, 쪼개지지 않으므로 보호 불필요
    - 여러 형태소로 쪼개지는 이름(P의 거짓, 붉은사막, 오징어게임2, 제우스: 오만의 신) → 개체이면서 보호"""
    tags = [t.tag for t in kiwi.tokenize(name) if t.tag not in ("SP", "SS", "SF", "SE", "SO", "SW")]
    if not tags or not any(t.startswith("N") or t in ("SL", "SH") for t in tags):
        return False, False
    if len(tags) == 1:
        return tags[0] in ("NNP", "SL", "SH", "UN"), False
    if all(t in GENERIC_TAGS for t in tags):
        return False, False
    return True, True


def needs_protection(name: str, kiwi) -> bool:
    return entity_status(name, kiwi)[1]


def quoted(title: str) -> list[str]:
    out = []
    for q in QUOTE.findall(title or ""):
        q = q.strip(" ,.·…")
        if q.count(" ") >= 3 or re.search(r"(다|요|죠|까|네|나|자)$", q) or re.search(r"[?!…~]", q) \
                or re.fullmatch(r"[\d.,%]+", q):
            continue
        out.append(q)
    return out


def protected_entities(arts: pd.DataFrame, market_words: set[str], min_articles: int = 3,
                       min_concentration: float = 0.5) -> pd.DataFrame:
    q = arts["title"].map(quoted)
    cnt = collections.Counter(x for l in q for x in set(l))
    co = collections.defaultdict(collections.Counter)
    first = {}
    for l, cs, d in zip(q, arts["companies"], arts["date"]):
        for x in set(l):
            if cnt[x] >= min_articles:
                co[x].update(cs)
                m = str(d)[:7]
                first[x] = min(first.get(x, m), m)
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    rows = []
    for x, c in co.items():
        top, n = c.most_common(1)[0]
        conc = n / cnt[x]
        words = set(re.split(r"\s+", x))
        # 종목군 표현(게임주, 엔터주)은 시장 구문
        stockish = len(x) >= 3 and x.endswith("주") and " " not in x
        if conc < min_concentration or words & market_words or stockish or len(_space_free(x)) < 2 or re.match(r"^\d", x):
            continue
        is_entity, protect = entity_status(x, kiwi)
        if not is_entity:
            continue
        rows.append({"entity": x, "token": _space_free(x), "protect": protect, "articles": cnt[x],
                     "company_concentration": round(conc, 3), "top_company": top, "first_month": first[x]})
    df = pd.DataFrame(rows, columns=["entity", "token", "protect", "articles", "company_concentration", "top_company", "first_month"])
    # 같은 토큰으로 겹치면 기사 수가 많은 표기를 대표로
    return df.sort_values("articles", ascending=False).drop_duplicates("token").reset_index(drop=True)
