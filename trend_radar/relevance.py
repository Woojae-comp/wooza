"""콘텐츠 관련성 분류기: LLM 라벨(정답지) → 로지스틱 회귀 → 전체 기사 판정.

8.3만 건 전부를 LLM으로 판정하지 않고, 층화 표본 약 2천 건만 LLM으로 라벨링한 뒤
가벼운 분류기를 학습해 전체에 적용한다. 떼어 둔 라벨로 성능을 재고, 가중치로 판단 근거를 보여 준다.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .llm_extract import PRICES, client

NOISE_TYPES = ["콘텐츠 흐름", "시세·증시", "실적·공시", "기업 경영·지배구조", "다른 업종", "기타"]

SYSTEM = """너는 한국 콘텐츠산업(게임, 음악, 방송·영상, 만화·웹툰, 애니메이션, 캐릭터와 이를 둘러싼 플랫폼·광고·출판·영화) 뉴스를 분류한다.
기사 제목과 요약만 보고, 이 기사가 콘텐츠산업의 흐름을 보여 주는지 판정한다.

relevant = true: 작품·IP, 신작·흥행, 서비스·플랫폼, 이용자, 제작·유통, 기술(AI 등)의 콘텐츠 적용, 해외 진출, 공연·팬덤,
콘텐츠 관련 정책·규제·분쟁, 콘텐츠 사업 전략에 관한 기사. 주가 기사라도 그 이유가 콘텐츠 사업(신작 흥행 등)이면 true.
relevant = false: 주가 등락·시황·종목 나열이 중심인 기사, 실적 수치만 전하는 공시 기사, 지분·지배구조·인사만 다루는 기사,
기업명이 우연히 걸린 다른 업종 기사(금융, 반도체, 여행 등).

type은 기사의 주된 성격: 콘텐츠 흐름, 시세·증시, 실적·공시, 기업 경영·지배구조, 다른 업종, 기타.
JSON만 출력한다."""

SCHEMA = {
    "type": "object",
    "properties": {"articles": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "relevant": {"type": "boolean"},
                       "type": {"type": "string", "enum": NOISE_TYPES}},
        "required": ["id", "relevant", "type"], "additionalProperties": False}}},
    "required": ["articles"], "additionalProperties": False,
}


def stratified_sample(frame: pd.DataFrame, by: list[str], n: int, seed: int = 11, exclude: set | None = None) -> pd.DataFrame:
    f = frame[~frame["gid"].isin(exclude or set())]
    per = max(1, n // f.groupby(by).ngroups)
    rng = np.random.default_rng(seed)
    s = f.assign(_r=rng.random(len(f))).sort_values("_r").groupby(by).head(per).drop(columns="_r")
    if len(s) < n:
        s = pd.concat([s, f[~f["gid"].isin(s["gid"])].sample(n - len(s), random_state=seed)])
    return s.head(n).reset_index(drop=True)


def label_llm(sample: pd.DataFrame, model: str, out: Path, chunk: int = 40, workers: int = 4) -> pd.DataFrame:
    cl = client()

    def call(df):
        body = "\n".join(json.dumps({"id": r.gid, "title": r.title, "summary": r.summary_raw}, ensure_ascii=False)
                         for r in df.itertuples())
        msg = cl.messages.create(
            model=model, max_tokens=8000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "다음 기사들을 분류해.\n" + body}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"})
        if msg.stop_reason != "end_turn":
            return [], msg.usage
        text = "".join(b.text for b in msg.content if b.type == "text")
        return json.loads(text)["articles"], msg.usage

    parts = [sample.iloc[i:i + chunk] for i in range(0, len(sample), chunk)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(call, parts))
    rows = [x for items, _ in res for x in items]
    u_in = sum(u.input_tokens for _, u in res)
    u_out = sum(u.output_tokens for _, u in res)
    u_cr = sum((getattr(u, "cache_read_input_tokens", 0) or 0) for _, u in res)
    pin, pout = PRICES.get(model, (0, 0))
    meta = {"model": model, "requested": len(sample), "labeled": len(rows), "seconds": round(time.time() - t0, 1),
            "input_tokens": u_in, "output_tokens": u_out, "cache_read": u_cr,
            "est_cost_usd": round((u_in * pin + u_cr * pin * 0.1 + u_out * pout) / 1e6, 3)}
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.DataFrame(rows).rename(columns={"id": "gid"})
    lab.to_csv(out / "relevance_labels.csv", index=False, encoding="utf-8-sig")
    (out / "relevance_labels.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return lab


def features(arts: pd.DataFrame, kw: list[list[str]], min_df: int = 5):
    """기사 키워드(본문·제목 구분) TF-IDF + 수치 특징. 반환: 행렬, 특징 이름."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    titles = arts["title"].to_numpy()
    docs = []
    for ws, t in zip(kw, titles):
        s = set(ws)
        docs.append(" ".join(list(s) + ["T_" + w for w in s if w in t]))
    vec = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, token_pattern=None,
                          min_df=min_df, sublinear_tf=True)
    X = vec.fit_transform(docs)
    num = pd.DataFrame({
        "기업이 주인공": arts["strong"].astype(float).to_numpy(),
        "연관산업 기사": (arts["tier"] == "related").astype(float).to_numpy(),
        "제목 태그 있음": arts["tag"].notna().astype(float).to_numpy(),
        "콘텐츠 점수(log)": np.log1p(arts["content_score"].to_numpy()),
        "시장 점수(log)": np.log1p(arts["market_score"].to_numpy()),
        "콘텐츠 비율": arts["content_share"].to_numpy(),
    })
    names = list(vec.get_feature_names_out()) + list(num.columns)
    return sparse.hstack([X, sparse.csr_matrix(num.to_numpy())]).tocsr(), names


def train_eval(X, y: np.ndarray, names: list[str], seed: int = 0) -> tuple[object, dict, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    best = None
    for C in (0.3, 1.0, 3.0, 10.0):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        p = cross_val_predict(m, X[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        f = f1_score(y[tr], p >= 0.5)
        if best is None or f > best[1]:
            best = (C, f)
    m = LogisticRegression(C=best[0], class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    yh = p >= 0.5
    rep = {"C": best[0], "cv_f1": round(best[1], 3), "test_n": int(len(te)), "train_n": int(len(tr)),
           "accuracy": round(accuracy_score(y[te], yh), 3), "precision": round(precision_score(y[te], yh), 3),
           "recall": round(recall_score(y[te], yh), 3), "f1": round(f1_score(y[te], yh), 3),
           "auc": round(roc_auc_score(y[te], p), 3), "positive_rate_labels": round(float(y.mean()), 3)}
    final = LogisticRegression(C=best[0], class_weight="balanced", max_iter=3000).fit(X, y)
    coef = pd.DataFrame({"feature": names, "weight": final.coef_[0]}).sort_values("weight")
    return final, rep, coef
