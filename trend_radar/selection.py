"""운영 모델 포인터 (2026-09-28 사용자 결정).

후속 단계는 레지스트리의 '최신' 실행이 아니라 설정의 선정 모델만 읽는다.
- e2.selected_run_id: 승인된 관련성 모델 (selection_status = approved)
- e2.candidate_run_id: 검증 중인 후보. `--use-candidate`로 명시했을 때만 하류 단계에 연결
- registry/candidate_status.jsonl: 실패한 후보 (candidate_failed = 1) → 입력으로 쓰면 중단
- E3 이후는 선정 E2에서 만들어진 E3만 읽는다 (input_e2_run_id 일치)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class SelectionError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def failed_runs(registry: Path) -> set[str]:
    return {r["run_id"] for r in _jsonl(registry / "candidate_status.jsonl") if r.get("candidate_failed")}


def selected_e2(cfg: dict, registry: Path, out_root: Path, use_candidate: bool | None = None) -> tuple[str, Path, dict]:
    """(run_id, 02_article_relevance.parquet 경로, e2_summary). 실패 후보·미승인 실행이면 SelectionError."""
    s = cfg.get("e2", {})
    use_candidate = cfg.get("_use_candidate", False) if use_candidate is None else use_candidate
    if use_candidate:
        run_id = s.get("candidate_run_id")
        if not run_id or s.get("selection_status") != "candidate":
            raise SelectionError("후보 E2가 지정되지 않았다 (e2.candidate_run_id, selection_status: candidate)")
    else:
        run_id = s.get("selected_run_id")
        if not run_id or s.get("selection_status") not in ("approved", "candidate"):
            raise SelectionError("승인된 E2가 없다 (e2.selected_run_id)")
    if run_id in failed_runs(registry):
        raise SelectionError(f"{run_id}는 실패한 후보(candidate_failed=1)라 하류 입력으로 쓸 수 없다")
    d = out_root / "runs" / run_id
    if not (d / "02_article_relevance.parquet").exists():
        raise SelectionError(f"{run_id}의 02_article_relevance.parquet가 없다")
    return run_id, d / "02_article_relevance.parquet", json.loads((d / "e2_summary.json").read_text(encoding="utf-8"))


def selected_e3(registry: Path, e2_run: str) -> str:
    """선정 E2에서 만든 가장 최근 E3 (keyword_dictionary). 없으면 SelectionError."""
    recs = [r for r in _jsonl(registry / "experiment_registry.jsonl")
            if r.get("task") == "keyword_dictionary" and r.get("input_e2_run_id") == e2_run]
    if not recs:
        raise SelectionError(f"선정 E2({e2_run})로 만든 E3가 없다 — E3를 먼저 실행")
    return recs[-1]["run_id"]


def mark_failed(registry: Path, run_id: str, model_id: str, variants: list[str], reason: str) -> dict:
    from .runlog import append_jsonl, now

    rec = {"run_id": run_id, "model_id": model_id, "variants": variants, "candidate_failed": 1, "reason": reason, "marked_at": now()}
    append_jsonl(registry / "candidate_status.jsonl", rec)
    return rec


def relevance_weight(half: np.ndarray, gids, cfg: dict, out_root: Path) -> np.ndarray:
    """E4·E5 기사 가중치. 기본은 E2 판정 가중(INCLUDE 1 / REVIEW 0.5 / EXCLUDE 0) 그대로.
    relevance_weighting.mode = 'half_x_pcontent' 이면 E2.3c 변형 산출물의 P(CONTENT)를 곱한다
    (유입 여부는 E2 판정 그대로, 유입 기사 안에서 시장 기사 비중만 낮춘다)."""
    rw = cfg.get("relevance_weighting") or {}
    if rw.get("mode", "half") == "half":
        return half
    run = rw.get("variants_run")
    if not run:
        raise SelectionError("relevance_weighting.variants_run 이 비어 있다")
    v = pd.read_parquet(out_root / "runs" / run / "02c_relevance_variants.parquet").set_index("gid")
    p = v[rw.get("column", "p_content_H2_event")].reindex(list(gids)).to_numpy()
    if np.isnan(p).any():
        raise SelectionError("P(CONTENT) 변형 산출물에 없는 기사가 있다 (스냅샷 확인)")
    return half * np.maximum(p, float(rw.get("floor", 0.0)))
