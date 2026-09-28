"""불용어·동의어·복합어 후보 제안 (설계 5장).

자동 적용하지 않는다. 후보와 근거를 표로 내고, 사용자가 승인한 것만 lexicon.yaml에 옮긴다.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .text import HANGUL, JOINERS, DocTerm, Token

# 기사체에 흔한 일반 명사. 제거를 확정하는 목록이 아니라 '제거 후보' 판단에 참고하는 표시일 뿐이다.
GENERIC_NEWS = set("""
기자 뉴스 이날 올해 지난해 작년 내년 이번 당시 관련 기준 대비 최근 가운데 이후 이전 이상 이하 오전 오후 전일 당일 금일
해당 사진 제공 무단 전재 배포 금지 사실 경우 부분 정도 가능 예정 진행 대상 통해 위해 때문 한편 가량 여기 이곳 그동안 동안
기간 수준 내용 결과 모습 상황 방침 설명 입장 발표 소식 주요 전체 일부 대부분 각각 현재 향후 오늘 내일 어제 연합뉴스 뉴시스
""".split())

# 설계 4장: 기사에 실제 등장하면 분석 대상으로 유지하는 유형 (주가·공시·투자·실적·정책·기술 등)
DESIGN_KEEP = set("""
주가 공시 투자 실적 정책 기술 매출 매출액 영업이익 영업익 순이익 적자 흑자 규제 인수 합병 상장 배당 지분 투자자 IPO
""".split())

# 알려진 표기 쌍 (둘 다 어휘에 있을 때만 후보로 낸다)
KNOWN_PAIRS = [
    ("NAVER", "네이버"), ("카카오", "Kakao"), ("AI", "인공지능"), ("넷플릭스", "Netflix"), ("유튜브", "YouTube"),
    ("매출", "매출액"), ("영업이익", "영업익"), ("순이익", "순익"), ("KPOP", "케이팝"), ("KPOP", "K팝"),
    ("IP", "지식재산"), ("IP", "지식재산권"), ("엔터", "엔터테인먼트"), ("OTT", "온라인동영상서비스"),
    ("NFT", "대체불가토큰"), ("하이브", "HYBE"), ("SM", "에스엠"), ("YG", "와이지"), ("JYP", "제이와이피"),
    ("웹툰", "Webtoon"), ("스팀", "Steam"), ("닌텐도", "Nintendo"), ("메타버스", "Metaverse"),
    ("생성형AI", "생성AI"), ("게임사", "게임업체"), ("CGV", "씨지브이"),
]


def _examples(dt: DocTerm, arts: pd.DataFrame, word: str, n: int = 3) -> list[str]:
    rows = dt.docs_with(word)
    if len(rows) == 0:
        return []
    pick = rows[np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)]
    return [f"{arts['date'].iat[r].date()} {arts['title'].iat[r]}" for r in pick]


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def stopword_candidates(dt: DocTerm, arts: pd.DataFrame, series: dict, sector_rows: dict[str, np.ndarray],
                        centrality: pd.DataFrame | None, companies: set[str], top: int = 600) -> pd.DataFrame:
    n = len(arts)
    sectors = list(sector_rows)
    sec_df = np.stack([np.asarray(dt.X[r].sum(axis=0)).ravel() for r in sector_rows.values()], axis=1)
    sec_n = np.array([len(r) for r in sector_rows.values()])
    sec_sh = sec_df / sec_n
    overall = dt.df / n
    q = series["quarter"]["shares"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.nan_to_num(q.std(1) / q.mean(1), nan=9.9)
        max_lift = sec_sh.max(1) / overall
    cent = centrality.set_index("keyword") if centrality is not None else pd.DataFrame()
    idx = list(range(min(top, len(dt.vocab))))
    idx += [dt.index[w] for w in GENERIC_NEWS if w in dt.index and dt.index[w] >= top]
    rows = []
    for j in idx:
        w = dt.vocab[j]
        ent = _entropy(sec_sh[j])
        deg = float(cent.loc[w, "degree_centrality"]) if w in cent.index else np.nan
        reasons = []
        if w in companies:
            rec, reasons = "유지 권고", ["분석 대상 기업명"]
        elif w in DESIGN_KEEP:
            rec, reasons = "유지 권고", ["설계상 분석 대상 (주가·공시·투자·실적·정책·기술 유형)"]
        elif max_lift[j] >= 2.0:
            top_sec = sectors[int(sec_sh[j].argmax())]
            rec, reasons = "유지 권고", [f"'{top_sec}' 분야에 집중 (전체 대비 {max_lift[j]:.1f}배)"]
        elif cv[j] >= 0.6:
            rec, reasons = "유지 권고", [f"시기별 변동이 큼 (분기 비중 변동계수 {cv[j]:.2f}) → 이슈를 구분하는 정보"]
        else:
            flat = ent >= 0.93 and cv[j] <= 0.35 and max_lift[j] < 1.5
            if w in GENERIC_NEWS and flat:
                rec = "제거 후보"
                reasons.append("기사체 일반어")
            elif flat and overall[j] >= 0.02:
                rec = "제거 후보"
                reasons.append("모든 분야·시기에 고르게 많이 등장 (구분 정보 적음)")
            elif w in GENERIC_NEWS:
                rec = "검토 필요"
                reasons.append("기사체 일반어이나 분포가 고르지 않음")
            else:
                rec = "검토 필요"
            reasons.append(f"분야 분포 균등도 {ent:.2f}, 분기 변동계수 {cv[j]:.2f}, 최대 분야 집중 {max_lift[j]:.1f}배")
        rows.append({"keyword": w, "recommendation": rec, "reason": "; ".join(reasons), "articles": int(dt.df[j]),
                     "share": round(float(overall[j]), 4), "sector_evenness": round(ent, 3), "quarter_cv": round(float(cv[j]), 3),
                     "max_sector_lift": round(float(max_lift[j]), 2), "degree_centrality": deg,
                     "examples": " | ".join(_examples(dt, arts, w))})
    out = pd.DataFrame(rows)
    order = {"제거 후보": 0, "검토 필요": 1, "유지 권고": 2}
    return out.sort_values(["recommendation", "articles"], key=lambda s: s.map(order) if s.name == "recommendation" else -s)


def _context_similarity(dt: DocTerm, words: list[str]) -> dict[str, np.ndarray]:
    cols = np.array([dt.index[w] for w in words])
    co = np.asarray((dt.X[:, cols].T @ dt.X).todense())
    cos = co / np.sqrt(np.outer(np.maximum(dt.df[cols], 1), np.maximum(dt.df, 1)))
    return {w: cos[i] for i, w in enumerate(words)}


def _ctx_sim(ctx: dict[str, np.ndarray], dt: DocTerm, a: str, b: str) -> float:
    va, vb = ctx[a].copy(), ctx[b].copy()
    for w in (a, b):
        va[dt.index[w]] = 0
        vb[dt.index[w]] = 0
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def synonym_candidates(dt: DocTerm, arts: pd.DataFrame, texts: list[str], top: int = 3000) -> pd.DataFrame:
    vocab = set(dt.vocab)
    pairs: dict[tuple[str, str], dict] = {}

    # (1) 괄호 병기: 인공지능(AI)
    paren = Counter()
    pat = re.compile(r"([가-힣A-Za-z][가-힣A-Za-z0-9]{1,15})\(([가-힣A-Za-z][가-힣A-Za-z0-9]{1,15})\)")
    for t in texts:
        for a, b in pat.findall(t):
            if a != b:
                paren[(a, b)] += 1
    for (a, b), c in paren.items():
        if c >= 5 and a in vocab and b in vocab and (HANGUL.search(a) is None) != (HANGUL.search(b) is None):
            key = tuple(sorted((a, b)))
            pairs.setdefault(key, {"evidence": []})["evidence"].append(f"괄호 병기 {a}({b}) {c}회")

    # (2) 줄임 표기: 가운데 한 글자가 빠진 형태 (영업이익/영업익, 순이익/순익). 맥락 유사도가 높아야 후보로 낸다.
    #     접미사가 붙은 형태(이사/이사회, 게임/게임사)는 다른 개념인 경우가 많아 후보로 내지 않는다.
    head = [w for w in dt.vocab[:top] if HANGUL.search(w)]
    hs = set(head)
    abbrev = set()
    for w in head:
        for k in range(1, len(w) - 1):
            v = w[:k] + w[k + 1:]
            if len(v) >= 2 and v in hs:
                key = tuple(sorted((w, v)))
                abbrev.add(key)
                pairs.setdefault(key, {"evidence": []})["evidence"].append(f"'{w}'에서 가운데 한 글자 뺀 형태")

    # (3) 알려진 표기 쌍
    for a, b in KNOWN_PAIRS:
        if a in vocab and b in vocab:
            key = tuple(sorted((a, b)))
            pairs.setdefault(key, {"evidence": []})["evidence"].append("알려진 표기 쌍")

    if not pairs:
        return pd.DataFrame(columns=["canonical", "variant", "evidence"])
    words = sorted({w for p in pairs for w in p})
    ctx = _context_similarity(dt, words)
    rows = []
    for (a, b), info in pairs.items():
        sim = _ctx_sim(ctx, dt, a, b)
        canon, var = (a, b) if dt.df[dt.index[a]] >= dt.df[dt.index[b]] else (b, a)
        known = "알려진 표기 쌍" in info["evidence"] or any(e.startswith("괄호") for e in info["evidence"])
        if (sim < 0.7 if (a, b) in abbrev else sim < 0.35) and not known:
            continue  # 쓰이는 맥락이 다르면 같은 개념으로 보기 어렵다
        rows.append({"canonical": canon, "variant": var, "canonical_articles": int(dt.df[dt.index[canon]]),
                     "variant_articles": int(dt.df[dt.index[var]]), "context_similarity": round(sim, 3),
                     "evidence": "; ".join(dict.fromkeys(info["evidence"])),
                     "recommendation": "통합 후보" if sim >= 0.6 or known else "검토 필요",
                     "variant_examples": " | ".join(_examples(dt, arts, var, 2))})
    if not rows:
        return pd.DataFrame(columns=["canonical", "variant", "evidence"])
    out = pd.DataFrame(rows)
    out["_o"] = out["recommendation"].map({"통합 후보": 0, "검토 필요": 1})
    return out.sort_values(["_o", "context_similarity"], ascending=[True, False]).drop(columns="_o")


def compound_candidates(tokens: list[list[Token]], texts: list[str], arts: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """붙어서 나온 형태소 쌍/세 쌍. 결합 비율(앞 형태소 뒤에 이 형태소가 붙어 나온 비율)이 높을수록 한 개념일 가능성."""
    uni = Counter()
    big = Counter()
    tri = Counter()
    example = defaultdict(list)
    for r, (toks, text) in enumerate(zip(tokens, texts)):
        uni.update(t[0] for t in toks)
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if b[1] >= a[2] and text[a[2]:b[1]] in JOINERS:
                if a[3] == "XSN":
                    continue
                key = (a[0], b[0])
                big[key] += 1
                if len(example[key]) < 2 and r % 7 == 0:
                    example[key].append(arts["title"].iat[r])
                if i + 2 < len(toks):
                    c = toks[i + 2]
                    if c[1] >= b[2] and text[b[2]:c[1]] in JOINERS:
                        tri[(a[0], b[0], c[0])] += 1
    rows = []
    for (a, b), c in big.items():
        if c < min_count:
            continue
        left, right = c / uni[a], c / uni[b]
        if max(left, right) < 0.4:
            continue
        one_char = len(a) == 1 or len(b) == 1
        strong = min(left, right) >= 0.5 or (max(left, right) >= 0.75 and min(left, right) >= 0.3)
        rec = "결합 후보" if strong or (one_char and max(left, right) >= 0.6) else "검토 필요"
        rows.append({"compound": f"{a}+{b}", "merged": a + b, "count": c, "left_ratio": round(left, 3),
                     "right_ratio": round(right, 3), "recommendation": rec,
                     "note": "한 글자 형태소 포함 (결합하지 않으면 분석에서 빠짐)" if one_char else "",
                     "examples": " | ".join(example.get((a, b), []))})
    for (a, b, c3), c in tri.items():
        if c < min_count:
            continue
        ratio = c / max(big[(a, b)], 1)
        if ratio >= 0.6 and c / uni[c3] >= 0.4:
            rows.append({"compound": f"{a}+{b}+{c3}", "merged": a + b + c3, "count": c, "left_ratio": round(ratio, 3),
                         "right_ratio": round(c / uni[c3], 3), "recommendation": "검토 필요", "note": "세 형태소", "examples": ""})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = out["recommendation"].map({"결합 후보": 0, "검토 필요": 1})
    return out.sort_values(["_o", "count"], ascending=[True, False]).drop(columns="_o")
