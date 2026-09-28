"""기사 층 분류: 콘텐츠·사업 층 / 자본시장 층 (설계 3.3 '별도 레이어').

원자료가 '회사명+주가/공시' 검색이라 시세·실적·공시 기사가 대부분이다. 그 위에서 트렌드를 찾으면
시장 어휘가 콘텐츠 흐름을 덮는다. 그래서 기사를 두 층으로 나눠 따로 분석한다.

1. 씨앗 어휘(trend_radar.yaml layers.content_seed / market_seed)로 확실한 기사만 먼저 가른다.
2. 두 씨앗 기사 묶음에서 전체 키워드의 층 성향(로그 오즈)을 계산한다.
   → 씨앗에 없는 작품명·서비스명도 어느 층 기사에 주로 나오는지로 성향을 얻는다.
3. 기사마다 콘텐츠 점수·시장 점수를 합산해 층을 정한다. 한 기사가 두 층에 모두 들 수 있다
   (예: "PUBG 흥행에 2분기 매출 최대").
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def keyword_weights(kw: list[list[str]], content_seed: set[str], market_seed: set[str], cfg: dict) -> pd.DataFrame:
    lc = cfg["layers"]
    sets = [set(ws) for ws in kw]
    c0 = np.array([len(s & content_seed) for s in sets])
    m0 = np.array([len(s & market_seed) for s in sets])
    # 씨앗 기사: 한 층 씨앗 어휘가 seed_min_hits개 이상이고 다른 층 씨앗보다 seed_dominance배 이상 많다
    k = lc["seed_dominance"]
    seed_c = (c0 >= lc["seed_min_hits"]) & (c0 >= k * m0)
    seed_m = (m0 >= lc["seed_min_hits"]) & (m0 >= k * c0)
    cc, mc, df = Counter(), Counter(), Counter()
    for s, a, b in zip(sets, seed_c, seed_m):
        df.update(s)
        if a:
            cc.update(s)
        if b:
            mc.update(s)
    nc, nm = max(seed_c.sum(), 1), max(seed_m.sum(), 1)
    rows = []
    for w, d in df.items():
        if d < lc["min_df"]:
            continue
        # 씨앗 기사에 몇 번 안 나온 단어는 성향을 믿기 어렵다 → 중립
        if cc[w] + mc[w] < lc["min_evidence"] and w not in content_seed and w not in market_seed:
            continue
        # 평활한 로그 오즈: 콘텐츠 씨앗 기사 비율 / 시장 씨앗 기사 비율
        wgt = np.log((cc[w] + 1) / (nc + 2)) - np.log((mc[w] + 1) / (nm + 2))
        if w in content_seed:
            wgt = max(wgt, lc["seed_weight"])
        if w in market_seed:
            wgt = min(wgt, -lc["seed_weight"])
        rows.append((w, d, cc[w], mc[w], float(np.clip(wgt, -4, 4))))
    out = pd.DataFrame(rows, columns=["keyword", "articles", "in_content_seed", "in_market_seed", "weight"])
    out["layer"] = np.select([out["weight"] >= lc["keyword_threshold"], out["weight"] <= -lc["keyword_threshold"]],
                             ["콘텐츠", "시장"], "중립")
    out.attrs["seed_content"] = int(seed_c.sum())
    out.attrs["seed_market"] = int(seed_m.sum())
    return out.sort_values("weight", ascending=False).reset_index(drop=True)


def classify(arts: pd.DataFrame, kw: list[list[str]], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lc = cfg["layers"]
    content_seed, market_seed = set(lc["content_seed"]), set(lc["market_seed"])
    kwt = keyword_weights(kw, content_seed, market_seed, cfg)
    w = dict(zip(kwt["keyword"], kwt["weight"]))
    thr = lc["keyword_threshold"]
    titles = arts["title"].to_numpy()
    cs, ms = np.zeros(len(kw)), np.zeros(len(kw))
    for i, ws in enumerate(kw):
        for x in set(ws):
            v = w.get(x, 0.0)
            # 제목에 나온 단어는 기사 주제를 더 잘 말해 주므로 두 배
            k = 2.0 if x in titles[i] else 1.0
            if v >= thr:
                cs[i] += k * v
            elif v <= -thr:
                ms[i] -= k * v
    share = cs / np.maximum(cs + ms, 1e-9)
    lab = pd.DataFrame({"gid": arts["gid"].to_numpy(), "content_score": cs, "market_score": ms, "content_share": share})
    strong = arts["strong"].to_numpy() if "strong" in arts else np.ones(len(kw), dtype=bool)
    lab["strong"] = strong
    # 기업이 주인공이 아닌 기사는 콘텐츠 성향이 더 뚜렷해야 콘텐츠 층에 든다 (업계 동향 기사). 자본시장 층에는 넣지 않는다
    need = np.where(strong, lc["content_min_share"], lc["weak_content_min_share"])
    seed_hits = np.array([len(set(ws) & content_seed) for ws in kw])
    lab["content_seed_hits"] = seed_hits
    lab["content"] = (cs >= lc["min_score"]) & (share >= need) & (strong | (seed_hits >= lc["weak_min_seed_hits"]))
    lab["market"] = (ms >= lc["min_score"]) & (share <= 1 - lc["market_min_share"]) & strong
    return lab, kwt
