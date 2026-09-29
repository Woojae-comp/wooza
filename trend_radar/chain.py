"""기본 원자료 → 레이더 화면까지 한 번에 (`python -m trend_radar all`).

순서: 형식 확인 → E0 등록·품질 → E2(승인된 모델로 이 스냅샷 재실행) → E3 → E4.2 → E5a → E5b → 화면.
- 모델을 바꾸지 않는다: E2는 승인된 운영 모델(e2.selected_model_id)을 새 스냅샷에 다시 적용할 뿐이다.
  운영 포인터(e2.selected_run_id)는 같은 모델의 새 실행으로 옮기고 decision_log에 REFRESH_SAME_MODEL로 남긴다.
- 같은 스냅샷에서 이미 만든 E2·E3가 있으면 다시 만들지 않는다 (reuse).
- E5b는 이번 실행에서 만든 E4.2·E5a를 명시해 읽는다 (다른 스냅샷·가중 민감도 실행과 섞이지 않게).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .runlog import REGISTRY, append_jsonl, now

CONFIG = Path(__file__).resolve().parent.parent / "trend_radar.yaml"


def _runs() -> list[dict]:
    p = REGISTRY / "run_registry.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def last_run(kind: str, snapshot_id: str) -> str | None:
    ok = [r["run_id"] for r in _runs() if r.get("kind") == kind and r.get("status") == "SUCCESS"
          and r.get("data_snapshot_id") == snapshot_id]
    return ok[-1] if ok else None


def run_snapshot(run_id: str) -> str | None:
    return next((r.get("data_snapshot_id") for r in _runs() if r.get("run_id") == run_id), None)


def set_pointer(run_id: str, path: Path = CONFIG) -> None:
    """trend_radar.yaml의 e2.selected_run_id 한 줄만 바꾼다 (주석·나머지 설정 보존)."""
    s = path.read_text(encoding="utf-8")
    s2, n = re.subn(r"(?m)^(\s*selected_run_id:\s*)\S+", rf"\g<1>{run_id}", s, count=1)
    if n != 1:
        raise RuntimeError("trend_radar.yaml에서 e2.selected_run_id 줄을 찾지 못했다")
    path.write_text(s2, encoding="utf-8")


def run_all(cfg: dict, paths: list[str] | None, out_dir: str, min_df: int = 5, log=print) -> dict:
    from .e0 import check_schema, input_files, run_e0
    from .e2 import run_e2
    from .e3 import run_e3
    from .e42 import run_e42
    from .e5 import run_e5a
    from .e5b import run_e5b
    from .load import read_raw
    from .pipeline import run as run_radar
    from .runlog import snapshot
    from .selection import selected_e3

    out = Path(out_dir)
    files = input_files(paths or cfg["input"]["paths"])
    raw = read_raw(files)
    versions = check_schema(raw, cfg, files)
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last)["data_snapshot_id"]
    cfg["_input_files"] = files
    log(f"[형식] {versions} · 스냅샷 {snap} · 파일 {len(files)}개 · 마지막 기사일 {last}")
    steps = {"data_snapshot_id": snap, "schema_versions": versions}

    log("[E0] 등록·품질 점검")
    run_e0(cfg, files, str(out))
    steps["e0"] = last_run("e0_data_quality", snap)

    e2 = cfg.get("e2", {})
    cur = e2.get("selected_run_id")
    if cur and run_snapshot(cur) == snap:
        log(f"[E2] 같은 스냅샷의 운영 실행 재사용 {cur}")
        e2_run = cur
    else:
        log(f"[E2] 승인 모델 {e2.get('selected_model_id')}를 새 스냅샷에 적용")
        run_e2(cfg, raw, out)
        e2_run = last_run("e2_relevance", snap)
        set_pointer(e2_run)
        cfg["e2"]["selected_run_id"] = e2_run
        append_jsonl(REGISTRY / "decision_log.jsonl", {
            "review_id": f"rev_{e2_run}", "target_type": "pointer", "target_id": "e2.selected_run_id",
            "previous_value": cur, "new_value": e2_run, "reason_code": "REFRESH_SAME_MODEL",
            "reason_text": f"모델 {e2.get('selected_model_id')} 그대로, 새 스냅샷 {snap}에 재실행 (python -m trend_radar all)",
            "reviewer": "pipeline", "reviewed_at": now()})
    steps["e2"] = e2_run

    try:
        e3_run = selected_e3(REGISTRY, e2_run)
        log(f"[E3] 재사용 {e3_run}")
    except Exception:
        log("[E3] 핵심어 사전")
        run_e3(cfg, raw, out, min_df)
        e3_run = selected_e3(REGISTRY, e2_run)
    steps["e3"] = e3_run

    log("[E4.2] 주제 구조")
    run_e42(cfg, raw, out)
    steps["e42"] = last_run("e42_topic_final", snap)
    log("[E5a] 키워드 신호")
    run_e5a(cfg, raw, out)
    steps["e5a"] = last_run("e5a_keyword_signal", snap)
    log("[E5b] 주제 트렌드·계보")
    cfg["e5b"] = {**cfg.get("e5b", {}), "e42_run": steps["e42"], "e5a_run": steps["e5a"]}
    s5b = run_e5b(cfg, raw, out)
    steps["e5b"] = last_run("e5b_topic_trend", snap)

    log("[화면] 키워드 레이더 + 주제 트렌드 탭")
    run_radar(cfg, files, None, str(out), raw=raw)
    steps["html"] = str(out / "radar.html")
    steps["trend_type_counts"] = s5b.get("trend_type_counts")
    append_jsonl(REGISTRY / "chain_runs.jsonl", {"finished_at": now(), **steps})
    return steps
