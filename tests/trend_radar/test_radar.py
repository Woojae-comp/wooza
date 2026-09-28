"""합성 뉴스로 Trend Radar 규칙 검증.

패턴을 알고 만든 기사에서 상태 분류·정제·구조 변화가 기대대로 나오는지 본다.
- 플랫폼: 비중 일정 (Established), 전체 기사량이 3배로 늘어도 Growing이 아니어야 한다
- 웹툰: 비중이 꾸준히 증가 (Growing)
- 판호: 한 달만 급증 (Event-driven)
- 로봇: 최근에만 등장 (Emerging)
- AI: 빈도는 일정한데 연관어가 기술·개발 → 규제·정책으로 바뀜 (구조 변화)
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from trend_radar.config import Lexicon, load_config
from trend_radar.load import build_corpus, split_sectors, strip_boilerplate
from trend_radar.pipeline import run
from trend_radar.text import merge_tokens


FILLER = ("서울 부산 제주 여름 겨울 봄 가을 아침 저녁 주말 연휴 축제 행사 발표회 간담회 전시회 박람회 설명회 "
          "학교 도서관 극장 공원 광장 시청 거리 항구 공항 역사 미술관 박물관").split()


def month_range():
    return pd.period_range("2024-01", "2026-09", freq="M")


def make_raw(seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    months = month_range()
    rows = []
    gid = 0
    T = len(months)
    for t, m in enumerate(months):
        n = int(60 + 120 * t / (T - 1))  # 전체 기사량 3배 증가
        for i in range(n):
            gid += 1
            words = ["콘텐츠"]
            if rnd.random() < 0.30:
                words.append("플랫폼")
            if rnd.random() < 0.01 + 0.25 * t / (T - 1):
                words.append("웹툰")
            if str(m) == "2025-03" and rnd.random() < 0.5:
                words.append("판호")
            if t >= T - 6 and rnd.random() < 0.15:
                words.append("로봇")
            if rnd.random() < 0.25:
                words.append("AI")
                if t < 12:
                    words += rnd.sample(["기술", "개발", "자동화"], 2)
                elif t >= T - 12:
                    words += rnd.sample(["규제", "정책", "수익"], 2)
            else:
                words.append(rnd.choice(["기술", "개발", "자동화", "규제", "정책", "수익", "시장", "공연"]))
            company, sector = rnd.choice([("넥슨게임즈", "게임"), ("하이브", "음악"), ("스튜디오드래곤", "방송 및 영상 · 영화")])
            rnd.shuffle(words)
            title = f"{company} " + " ".join(words[:3])
            filler = " ".join(rnd.sample(FILLER, 3))
            summary = f"{company}는 " + ", ".join(words) + f" {filler} 소식을 전했다... 무단 전재 및 재배포 금지 안내 문구입니다"
            day = min(28, 1 + i % 28)
            rows.append({"전역기사ID": f"g{gid}", "기사발행일": f"{m.year}-{m.month:02d}-{day:02d}", "제목": title,
                         "요약": summary, "회사명": company, "분야목록": sector, "검색구분": "회사명+주가",
                         "언론사도메인": "news.test", "원문링크": f"https://news.test/{gid}"})
            if i % 10 == 0:  # 같은 기사가 같은 분야 다른 기업으로도 수집됨 (중복)
                rows.append(dict(rows[-1], 회사명="넥슨게임즈" if company != "넥슨게임즈" else "하이브",
                                 분야목록="게임" if company != "넥슨게임즈" else "음악"))
    # 기업명이 다른 뜻으로 걸린 기사
    for k in range(30):
        rows.append({"전역기사ID": f"x{k}", "기사발행일": "2025-05-10", "제목": "삼천포대교 관광 명소",
                     "요약": "삼천포대교공원 바다 풍경 여행", "회사명": "대교", "분야목록": "출판", "검색구분": "회사명+주가",
                     "언론사도메인": "news.test", "원문링크": ""})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"keywords": {"min_df": 5}, "output": {"profile_top_n": 200}, "layers": {"enabled": False}, "cleaning": {"include_sectors": []},
                                  "network": {"top_n": 50, "min_cooccurrence": 3}})


@pytest.fixture(scope="module")
def raw():
    return make_raw()


@pytest.fixture(scope="module")
def result(tmp_path_factory, cfg, raw):
    out = tmp_path_factory.mktemp("radar")
    return run(cfg, out_dir=str(out), raw=raw, with_candidates=True), out


def test_sector_split():
    assert split_sectors("게임 · 콘텐츠솔루션 | 게임, 콘텐츠솔루션", "[·,|]") == ["게임", "콘텐츠솔루션"]
    assert split_sectors("방송 및 영상", "[·,|]") == ["방송 및 영상"]


def test_dedup_and_company_mention(cfg, raw):
    c = build_corpus(raw, cfg)
    # 전체 분석: 전역기사ID 한 번
    assert c.articles["gid"].is_unique
    assert len(c.articles) == raw.loc[raw["회사명"] != "대교", "전역기사ID"].nunique()
    # 삼천포대교 기사는 '대교'가 실제로 등장한 것이 아니므로 빠진다
    assert not c.articles["gid"].str.startswith("x").any()
    # 분야 분석: 전역기사ID + 분야 한 번
    assert not c.article_sector.duplicated().any()
    # 방송 및 영상 · 영화 → 두 분야
    assert {"방송 및 영상", "영화"} <= set(c.article_sector["sector"])


def test_boilerplate_removed(cfg, raw):
    c = build_corpus(raw, cfg)
    assert not c.articles["summary"].str.contains("재배포 금지").any()
    assert c.articles["summary_raw"].str.contains("재배포 금지").any()


def test_boilerplate_normalizes_numbers_and_names():
    s = pd.Series([f"넥슨 주가는 {i}원에 거래를 마감했다... 신작 {i} 공개" for i in range(25)])
    cleaned, removed = strip_boilerplate(s, 20, ["넥슨"])
    assert removed and "마감" in removed[0][0]
    assert not cleaned.str.contains("마감").any()


def test_compound_merge_uses_cache_offsets():
    lex = Lexicon(compounds=["영업+이익", "숏+폼"])
    text = "영업이익 개선, 숏폼 확대"
    toks = [("영업", 0, 2, "NNG"), ("이익", 2, 4, "NNG"), ("개선", 5, 7, "NNG"),
            ("숏", 9, 10, "NNG"), ("폼", 10, 11, "NNG"), ("확대", 12, 14, "NNG")]
    assert merge_tokens(toks, text, lex, {"형"}) == ["영업이익", "개선", "숏폼", "확대"]
    # 붙어 있지 않으면 합치지 않는다
    toks2 = [("영업", 0, 2, "NNG"), ("이익", 3, 5, "NNG")]
    assert merge_tokens(toks2, "영업 이익", lex, set()) == ["영업", "이익"]
    # 생성+형 접미사 부착
    assert merge_tokens([("생성", 0, 2, "NNG"), ("형", 2, 3, "XSN")], "생성형", Lexicon(), {"형"}) == ["생성형"]


def test_lexicon_normalize():
    lex = Lexicon(stopwords=["기자"], synonyms={"AI": ["인공지능"]})
    assert lex.normalize("인공지능") == "AI"
    assert lex.normalize("기자") is None


def status(result, k):
    return set(result[0]["profiles"][k]["status"])


def test_statuses(result):
    r = result[0]
    assert "Growing" in status(result, "웹툰")
    assert "Event-driven" in status(result, "판호")
    assert "Emerging" in status(result, "로봇")
    assert "Established" in status(result, "플랫폼")
    # 절대 기사 수는 3배가 됐지만 비중이 일정하므로 증가 트렌드가 아니다
    p = r["profiles"]["플랫폼"]
    assert p["m_count"][-2] > 2 * p["m_count"][0]
    assert not ({"Growing", "Emerging"} & status(result, "플랫폼"))
    assert "판호" in r["event"] and "판호" not in r["growing"]


def test_event_evidence(result):
    p = result[0]["profiles"]["판호"]
    assert p["peak"]["month"] == "2025-03" and p["peak"]["spike"]
    assert any("급증" in e for e in p["evidence"])
    assert p["arts"]["spike"], "급증 시점 기사가 연결되어야 한다"


def test_structural_change(result):
    r = result[0]
    st = r["profiles"]["AI"]["structure"]
    early = {w for w, _ in st["early_assoc"][:3]}
    recent = {w for w, _ in st["recent_assoc"][:3]}
    assert early == {"기술", "개발", "자동화"}
    assert recent == {"규제", "정책", "수익"}
    assert st["assoc_overlap"] < 0.5


def test_outputs(result):
    r, out = result
    assert (out / "radar.html").exists() and (out / "radar.json").exists()
    assert (out / "review" / "lexicon_candidates.xlsx").exists()
    assert (out / "tables" / "keyword_metrics.csv").exists()
    html = (out / "radar.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html
    # 모든 트렌드 목록의 키워드에는 근거와 기사 연결이 있다
    for key in ("emerging", "growing", "event", "structural"):
        for k in r[key]:
            assert r["profiles"][k]["evidence"]
            assert r["profiles"][k]["arts"]["latest"]
    # 분야별 결과는 모든 분야에 대해 있다
    assert set(r["sectors"]) == set(r["meta"]["sectors"])


def test_share_normalization(result):
    r = result[0]
    p = r["profiles"]["웹툰"]
    shares = np.array(p["m_share"])
    assert shares[-6:].mean() > 3 * shares[:6].mean()


def test_layers_split_content_and_market():
    from trend_radar.layers import classify

    cfg = load_config(overrides={"layers": {"min_df": 1, "min_evidence": 1, "min_score": 2.0}})
    content = ["신작", "출시", "게임", "흥행", "이용자"]
    market = ["주가", "목표주가", "투자의견", "공시", "영업이익"]
    kw, titles, strong = [], [], []
    for i in range(40):
        kw.append(content + ["쿠키런"]); titles.append("쿠키런 신작 출시"); strong.append(True)
        kw.append(market + ["증권"]); titles.append("목표주가 상향"); strong.append(True)
    # 기업이 스쳐 간 시황 기사: 시장 성향이어도 자본시장 층에 넣지 않는다
    kw.append(market); titles.append("코스닥 마감 시황"); strong.append(False)
    # 기업이 스쳐 간 업계 동향 기사: 콘텐츠 성향이 뚜렷하면 콘텐츠 층
    kw.append(content); titles.append("게임업계 신작 흥행"); strong.append(False)
    arts = pd.DataFrame({"gid": [f"g{i}" for i in range(len(kw))], "title": titles, "strong": strong})
    lab, kwt = classify(arts, kw, cfg)
    w = dict(zip(kwt["keyword"], kwt["weight"]))
    assert w["쿠키런"] > 0 > w["증권"]  # 씨앗에 없는 단어도 층 성향을 얻는다
    assert lab["content"].iloc[0] and not lab["market"].iloc[0]
    assert lab["market"].iloc[1] and not lab["content"].iloc[1]
    assert not lab["market"].iloc[-2]
    assert lab["content"].iloc[-1]


def test_label_model_recovers_classes_from_one_sided_rules():
    from trend_radar.e2 import label_model, otsu

    rng = np.random.default_rng(0)
    y = rng.random(3000) < 0.4
    def lf(cond_rate_pos, cond_rate_neg, pol):
        fire = np.where(y, rng.random(3000) < cond_rate_pos, rng.random(3000) < cond_rate_neg)
        return np.where(fire, pol, 0)
    L = pd.DataFrame({"pos_a": lf(0.6, 0.02, 1), "pos_b": lf(0.5, 0.05, 1),
                      "neg_a": lf(0.03, 0.7, -1), "neg_b": lf(0.1, 0.5, -1), "neg_c": lf(0.05, 0.4, -1)})
    p, st = label_model(L)
    acc = ((p >= 0.5) == y).mean()
    assert acc > 0.85  # 한쪽 방향 규칙만으로도 뒤집히지 않고 클래스를 복원한다
    assert abs(st.attrs["prior"] - 0.4) < 0.1
    assert (st["estimated_precision"] > 0.6).all()
    t = otsu(np.concatenate([rng.normal(0.1, 0.05, 500), rng.normal(0.9, 0.05, 500)]).clip(0, 1))
    assert 0.3 < t < 0.7


def test_include_sectors_drop(raw):
    cfg = load_config(overrides={"cleaning": {"include_sectors": ["게임", "방송 및 영상"], "related_tier": False}})
    c = build_corpus(raw, cfg)
    assert set(c.article_sector["sector"]) == {"게임", "방송 및 영상"}  # 영화·음악 매핑은 빠진다
    assert "하이브" in c.report["companies_out_of_scope"]  # 음악에만 속한 기업
    assert not c.article_company["company"].eq("하이브").any()


def test_include_sectors_related_tier(raw):
    cfg = load_config(overrides={"cleaning": {"include_sectors": ["게임", "방송 및 영상"], "related_tier": True}})
    c = build_corpus(raw, cfg)
    # 대상 밖 기업(하이브)의 기사는 '연관산업' 한 분야로 남는다
    assert set(c.article_sector["sector"]) == {"게임", "방송 및 영상", "연관산업"}
    hybe = c.article_company.loc[c.article_company["company"] == "하이브", "gid"]
    only_hybe = set(hybe) - set(c.article_company.loc[c.article_company["company"] != "하이브", "gid"])
    tiers = c.articles.set_index("gid").loc[sorted(only_hybe), "tier"]
    assert (tiers == "related").all()
    assert (c.articles["tier"] == "core").any()


def test_related_tier_excluded_from_overall(tmp_path, raw):
    cfg = load_config(overrides={"keywords": {"min_df": 5}, "output": {"profile_top_n": 100},
                                 "network": {"top_n": 50, "min_cooccurrence": 3}, "layers": {"enabled": False},
                                 "cleaning": {"include_sectors": ["게임", "방송 및 영상"], "related_tier": True}})
    r = run(cfg, out_dir=str(tmp_path), raw=raw, with_candidates=False)
    n_related = r["meta"]["related_articles"]
    assert n_related > 0
    assert r["meta"]["articles"] == len(build_corpus(raw, cfg).articles) - n_related  # 전체 트렌드는 핵심 기사만
    assert "연관산업" in r["sectors"] and "연관산업" not in r["meta"]["sectors"]  # 분야별 탭에는 있고 확산도 분모에는 없다
    assert any("연관산업" in (p["a"], p["b"]) for p in r["sector_convergence"]["year"]["pairs"])
