"""E2.3c 사건 이슈 계열: 같은 문장 공기, 콘텐츠 분야 기업만, 정치·방송사 출처 기권, REVIEW 하한, 검토 표본 대조."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_radar.e23 import AB, VO
from trend_radar.e23c import (apply_floor, compare, content_company_names, event_hits, gates_v11, review_reference,
                              sentences)


def test_sentences_split_title_and_summary():
    assert sentences("제목", "첫 문장이다. 둘째 ... 셋째") == ["제목", "첫 문장이다", "둘째", "셋째"]


def test_event_hits_needs_content_company_in_same_sentence():
    ac = pd.DataFrame({"gid": ["a", "b", "c", "d"], "company": ["하이브", "하이브", "대신증권", "씨씨에스"],
                       "sectors": [["음악"], ["음악"], ["금융"], ["방송 및 영상"]]})
    names = content_company_names(ac, {"음악", "방송 및 영상"}, {"하이브": ["HYBE"]})
    assert "대신증권" not in sum(names.tolist(), []) and "HYBE" in names["a"]
    arts = pd.DataFrame({"gid": ["a", "b", "c", "d"],
                         "title": ["방시혁 결론 나오나", "하이브 신보 발매", "대신증권 압수수색", "씨씨에스충북방송 투자자"],
                         "summary": ["HYBE 의장의 주가조작 혐의 수사. 날씨 맑음", "경찰이 다른 사건 수사. 하이브 음반 발매",
                                     "", "거래소가 씨씨에스를 불성실공시법인으로 지정"]})
    hit = event_hits(arts, names, ["수사", "혐의", "압수수색", "불성실공시"])
    assert hit.tolist() == [True, False, False, True]      # b: 기업명과 사건어가 다른 문장, c: 콘텐츠 기업 아님


def test_floor_lifts_only_event_non_list_excludes():
    dec = np.array(["EXCLUDE", "EXCLUDE", "EXCLUDE", "INCLUDE"])
    ev = np.array([True, True, False, True])
    titles = pd.Series(["하이브 의장 수사", "3월 6일 개장 전 주요공시", "주가 급등", "신작"])
    assert apply_floor(dec, ev, titles).tolist() == ["REVIEW", "EXCLUDE", "EXCLUDE", "INCLUDE"]


def test_review_reference_reclass_and_compare(tmp_path):
    (tmp_path / "merged").mkdir()
    (tmp_path / "internal").mkdir()
    pd.DataFrame({"article_id": ["a1", "a2", "a3", "a4"], "class::gpt": ["CONTENT", "CONTENT", "MARKET", "OTHER"],
                  "class::opus": ["CONTENT", "CONTENT", "MARKET", "CONTENT"],
                  "sample_set": "representative", "weight": "10"}).to_csv(tmp_path / "merged" / "review_wide.csv", index=False)
    pd.DataFrame({"article_id": ["a2"], "class": ["MARKET"]}).to_csv(tmp_path / "internal" / "v11_reclass.csv", index=False)
    W = review_reference(tmp_path)
    assert W["ref"].tolist() == ["CONTENT", "MARKET", "MARKET", "DISAGREE"] and W["any_content"].tolist() == [True, False, False, True]
    tab, _ = compare(W, {"base": pd.Series(["INCLUDE", "REVIEW", "REVIEW", "EXCLUDE"], index=["a1", "a2", "a3", "a4"]),
                         "new": pd.Series(["REVIEW", "EXCLUDE", "EXCLUDE", "REVIEW"], index=["a1", "a2", "a3", "a4"])})
    t = tab.set_index("candidate")
    assert t.loc["base", "MARKET_in"] == 2 and t.loc["new", "MARKET_in"] == 0 and t.loc["new", "content_disputed_in"] == 1
    g = gates_v11(t.loc["new"].to_dict(), t.loc["base"].to_dict(), {})
    assert g["pass_all"]
    assert not gates_v11({**t.loc["new"].to_dict(), "CONTENT_in": 0}, t.loc["base"].to_dict(), {})["pass_all"]


def test_event_family_abstains_on_politics_and_broadcaster(monkeypatch):
    import trend_radar.e23c as m

    F = pd.DataFrame({"F_politics_society": [AB, VO, AB], "F_broadcaster_source": [AB, AB, VO]})
    monkeypatch.setattr(m, "event_hits", lambda arts, names, words: np.array([True, True, True]))
    monkeypatch.setattr(m, "content_company_names", lambda *a: pd.Series(dtype=object))

    class Corp:
        article_company = pd.DataFrame(columns=["gid", "company", "sectors"])

    p = {"F": F, "arts": pd.DataFrame({"gid": list("abc")}), "corpus": Corp()}
    assert m.event_family(p, {"cleaning": {}, "e2": {}}).tolist() == [True, False, False]


def test_relevance_weight_modes(tmp_path):
    from trend_radar.selection import SelectionError, relevance_weight

    half = np.array([1.0, 0.5, 0.0])
    assert relevance_weight(half, ["a", "b", "c"], {}, tmp_path) is half
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    pd.DataFrame({"gid": ["a", "b", "c"], "p_content_H2_event": [0.2, 0.0, 0.9]}) \
        .to_parquet(tmp_path / "runs" / "r1" / "02c_relevance_variants.parquet")
    cfg = {"relevance_weighting": {"mode": "half_x_pcontent", "variants_run": "r1", "floor": 0.1}}
    assert np.allclose(relevance_weight(half, ["a", "b", "c"], cfg, tmp_path), [0.2, 0.05, 0.0])
    import pytest
    with pytest.raises(SelectionError):
        relevance_weight(half, ["a", "zz"], cfg, tmp_path)
