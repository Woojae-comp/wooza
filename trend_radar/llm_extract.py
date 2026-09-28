"""LLM 구조화 추출: 기사마다 {주체, 이슈, 유형}을 뽑는다 (이슈 단위 트렌드 분석용).

- 표본: 콘텐츠 층 기사에서 분야×연도로 층화 추출 (run_sample)
- 전체: Message Batches API로 비동기 실행 (submit_batch / collect_batch, 50% 할인)
- 비교: 두 모델의 결과를 같은 기사에서 나란히 비교 (compare)

키는 UZA_ANTHROPIC_API_KEY 또는 ANTHROPIC_API_KEY 환경 변수에서 읽는다 (저장소에 두지 않는다).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["신작·흥행", "해외 진출", "IP 사업", "기술·AI", "플랫폼·유통", "팬덤·공연", "규제·분쟁", "조직·경영", "투자·실적", "기타"]
SUBJECT_TYPES = ["작품·IP", "기업", "플랫폼·서비스", "인물", "기관·정책", "기타"]

SYSTEM = f"""너는 한국 콘텐츠산업(게임, 음악, 방송·영상, 만화·웹툰, 애니메이션, 캐릭터) 뉴스를 분석하는 연구 보조다.
기사 제목과 요약을 보고 트렌드 분석에 쓸 정보를 뽑는다. 기사에 없는 내용은 만들지 않는다.

기사마다 다음을 뽑는다.
- subjects: 기사가 다루는 주체. 작품·IP(게임, 드라마, 영화, 웹툰, 캐릭터, 앨범 등 고유 작품명), 기업, 플랫폼·서비스, 인물, 기관·정책.
  기사에 실제로 나온 표기를 쓰되 따옴표·괄호는 뺀다. 최대 5개. 단순히 나열만 된 종목명은 넣지 않는다.
- issues: 이 기사가 보여 주는 산업 이슈를 짧은 명사구 1~3개로. 여러 기사에서 같은 말로 묶일 수 있게
  일반화된 표현을 쓴다 (예: "글로벌 진출", "IP 확장", "신작 출시", "숏폼 드라마", "생성형 AI", "월드투어",
  "OTT 스포츠 중계", "확률형 아이템 규제", "팬덤 플랫폼", "수익화"). 작품명·기업명은 이슈에 넣지 않는다.
  주가 등락·목표주가·실적 수치 자체는 이슈가 아니다. 다만 기사의 중심이 실적·투자면 "실적 개선", "M&A"처럼 쓴다.
- category: 기사의 중심 유형 하나. {", ".join(CATEGORIES)}
- content_relevant: 콘텐츠산업의 흐름(작품, 서비스, 사업, 기술, 이용자, 정책)에 관한 기사면 true.
  주가·시황·종목 나열·다른 업종 기사면 false.

JSON만 출력한다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "subjects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "type": {"type": "string", "enum": SUBJECT_TYPES}},
                            "required": ["name", "type"],
                            "additionalProperties": False,
                        },
                    },
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "content_relevant": {"type": "boolean"},
                },
                "required": ["id", "subjects", "issues", "category", "content_relevant"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["articles"],
    "additionalProperties": False,
}

# 가격 ($/1M 토큰, 2026-06 기준 표). 비용 보고용 추정치
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}


def client():
    import anthropic

    key = os.environ.get("UZA_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("UZA_ANTHROPIC_API_KEY 또는 ANTHROPIC_API_KEY 환경 변수가 필요하다")
    # Claude Code 세션 안에서는 ANTHROPIC_BASE_URL이 세션 프록시를 가리킬 수 있으므로 API로 직접 보낸다
    base = os.environ.get("TREND_RADAR_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    return anthropic.Anthropic(api_key=key, base_url=base, max_retries=4)


def sample_articles(arts: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    """분야(첫 분야) × 연도 층화 표본. 칸마다 고르게, 부족하면 남는 칸에서 채운다."""
    a = arts.copy()
    a["stratum"] = a["sectors"].map(lambda s: s[0] if s else "기타") + "|" + a["date"].dt.year.astype(str)
    per = max(1, n // a["stratum"].nunique())
    rng = np.random.default_rng(seed)
    picked = a.assign(_r=rng.random(len(a))).sort_values("_r").groupby("stratum").head(per).drop(columns="_r")
    if len(picked) < n:
        rest = a[~a["gid"].isin(picked["gid"])].sample(n - len(picked), random_state=seed)
        picked = pd.concat([picked, rest])
    return picked.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def _user_content(chunk: pd.DataFrame) -> str:
    lines = []
    for r in chunk.itertuples():
        lines.append(json.dumps({"id": r.gid, "date": str(r.date.date()), "title": r.title,
                                 "summary": r.summary_raw if hasattr(r, "summary_raw") else r.summary},
                                ensure_ascii=False))
    return "다음 기사들을 분석해.\n" + "\n".join(lines)


def _params(model: str, chunk: pd.DataFrame, effort: str) -> dict:
    return dict(
        model=model,
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _user_content(chunk)}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": effort},
    )


def _parse(msg) -> tuple[list[dict], dict]:
    usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
             "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
             "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
             "stop": msg.stop_reason}
    if msg.stop_reason in ("refusal", "max_tokens"):
        return [], usage
    text = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(text).get("articles", []), usage


def run_sample(arts: pd.DataFrame, model: str, out: Path, chunk_size: int = 20, workers: int = 4,
               effort: str = "low") -> pd.DataFrame:
    """표본을 동기 호출로 추출한다. 결과는 out/{model}.jsonl."""
    cl = client()
    chunks = [arts.iloc[i:i + chunk_size] for i in range(0, len(arts), chunk_size)]

    def call(chunk):
        p = _params(model, chunk, effort)
        if model == "claude-opus-5":
            # 안전 분류기가 거절하면 서버가 기본 대체 모델로 다시 실행한다 (배치 API에서는 쓸 수 없음)
            msg = cl.beta.messages.create(**p, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            msg = cl.messages.create(**p)
        return _parse(msg)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(call, chunks))
    rows, usage = [], []
    for items, u in results:
        rows.extend(items)
        usage.append(u)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{model}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    u = pd.DataFrame(usage)
    pin, pout = PRICES.get(model, (0, 0))
    cost = (u["input"].sum() * pin + u["cache_read"].sum() * pin * 0.1 + u["cache_write"].sum() * pin * 1.25
            + u["output"].sum() * pout) / 1e6
    meta = {"model": model, "articles": len(arts), "extracted": len(rows), "seconds": round(time.time() - t0, 1),
            "input_tokens": int(u["input"].sum()), "output_tokens": int(u["output"].sum()),
            "cache_read": int(u["cache_read"].sum()), "stops": u["stop"].value_counts().to_dict(),
            "est_cost_usd": round(cost, 3)}
    (out / f"{model}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return pd.DataFrame(rows)


def submit_batch(arts: pd.DataFrame, model: str, out: Path, chunk_size: int = 20, effort: str = "low") -> str:
    """전체 추출을 Message Batches API로 제출한다 (비동기, 50% 할인). 배치 ID를 out/batch_{model}.json에 남긴다."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    cl = client()
    reqs = [Request(custom_id=f"c{i // chunk_size}",
                    params=MessageCreateParamsNonStreaming(**_params(model, arts.iloc[i:i + chunk_size], effort)))
            for i in range(0, len(arts), chunk_size)]
    batch = cl.messages.batches.create(requests=reqs)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"batch_{model}.json").write_text(json.dumps({"id": batch.id, "requests": len(reqs)}), encoding="utf-8")
    return batch.id


def collect_batch(batch_id: str, model: str, out: Path) -> pd.DataFrame | None:
    """배치가 끝났으면 결과를 모아 out/{model}_full.jsonl에 쓴다. 아직이면 None."""
    cl = client()
    b = cl.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        return None
    rows = []
    for res in cl.messages.batches.results(batch_id):
        if res.result.type == "succeeded":
            items, _ = _parse(res.result.message)
            rows.extend(items)
    with open(out / f"{model}_full.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return pd.DataFrame(rows)


def load(out: Path, model: str) -> pd.DataFrame:
    p = out / f"{model}.jsonl"
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])


def _norm(s: str) -> str:
    return "".join(str(s).split()).lower()


def compare(arts: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> tuple[pd.DataFrame, dict]:
    """두 모델의 추출을 기사별로 맞대고 일치도를 계산한다."""
    A, B = a.set_index("id"), b.set_index("id")
    rows = []
    for r in arts.itertuples():
        if r.gid not in A.index or r.gid not in B.index:
            continue
        x, y = A.loc[r.gid], B.loc[r.gid]
        ix, iy = {_norm(i) for i in x["issues"]}, {_norm(i) for i in y["issues"]}
        sx, sy = {_norm(s["name"]) for s in x["subjects"]}, {_norm(s["name"]) for s in y["subjects"]}
        rows.append({
            "gid": r.gid, "date": str(r.date.date()), "sectors": ", ".join(r.sectors), "title": r.title,
            f"{name_a}_issues": " | ".join(x["issues"]), f"{name_b}_issues": " | ".join(y["issues"]),
            f"{name_a}_subjects": " | ".join(s["name"] for s in x["subjects"]),
            f"{name_b}_subjects": " | ".join(s["name"] for s in y["subjects"]),
            f"{name_a}_category": x["category"], f"{name_b}_category": y["category"],
            f"{name_a}_relevant": x["content_relevant"], f"{name_b}_relevant": y["content_relevant"],
            "issue_jaccard": len(ix & iy) / len(ix | iy) if ix | iy else 1.0,
            "subject_jaccard": len(sx & sy) / len(sx | sy) if sx | sy else 1.0,
            "category_same": x["category"] == y["category"],
            "relevant_same": bool(x["content_relevant"]) == bool(y["content_relevant"]),
        })
    df = pd.DataFrame(rows)
    summary = {
        "articles": len(df),
        "issue_jaccard_mean": round(float(df["issue_jaccard"].mean()), 3),
        "subject_jaccard_mean": round(float(df["subject_jaccard"].mean()), 3),
        "category_agreement": round(float(df["category_same"].mean()), 3),
        "relevance_agreement": round(float(df["relevant_same"].mean()), 3),
        f"{name_a}_relevant_rate": round(float(df[f"{name_a}_relevant"].mean()), 3),
        f"{name_b}_relevant_rate": round(float(df[f"{name_b}_relevant"].mean()), 3),
        f"{name_a}_distinct_issues": int(len({_norm(i) for l in a["issues"] for i in l})),
        f"{name_b}_distinct_issues": int(len({_norm(i) for l in b["issues"] for i in l})),
    }
    return df, summary
