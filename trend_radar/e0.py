"""E0 데이터 기반: 스냅샷 고정, 실행 기록, 데이터 품질 점검."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from .load import build_corpus, read_raw
from .quality import data_quality, write_quality
from .runlog import Run, snapshot


def input_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) or [p])
    return files


class SchemaError(ValueError):
    pass


def check_schema(raw: pd.DataFrame, cfg: dict, files: list[str] | None = None) -> dict:
    """기본 원자료 형식 확인: 필수 열(input.columns)과 공용 스키마 버전(input.schema_versions).
    어긋나면 SchemaError (분석 전에 멈춘다). 반환: 버전별 행 수."""
    ic = cfg["input"]
    missing = [c for c in ic["columns"].values() if c not in raw.columns]
    if missing:
        raise SchemaError(f"필수 열이 없다: {missing}")
    col, allowed = ic.get("schema_column"), ic.get("schema_versions")
    if not col or not allowed:
        return {}
    if col not in raw.columns:
        raise SchemaError(f"스키마 열 '{col}'이 없다 — 공용 원자료 형식이 아니다")
    counts = raw[col].fillna("(빈 값)").astype(str).value_counts().to_dict()
    bad = {v: n for v, n in counts.items() if v not in allowed}
    if bad:
        raise SchemaError(f"허용되지 않은 스키마 버전 {bad} (허용: {allowed}). 형식이 바뀌었으면 열 대응을 확인하고 "
                          f"input.schema_versions에 추가한다" + (f" · 파일 {files}" if files else ""))
    return counts


def run_e0(cfg: dict, paths: list[str] | None, out_dir: str) -> dict:
    files = input_files(paths or cfg["input"]["paths"])
    out = Path(out_dir)
    raw = read_raw(files)
    versions = check_schema(raw, cfg, files)
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last)
    run = Run("e0_data_quality", cfg, out, snap["data_snapshot_id"])
    with run.stage("load", rows_in=len(raw)) as st:
        corpus = build_corpus(raw, cfg)
        st["rows_out"] = len(corpus.articles)
    with run.stage("quality") as st:
        sheets = data_quality(raw, corpus, cfg, last)
        path = run.dir / "01_data_quality.xlsx"
        write_quality(sheets, path)
        run.artifact(path, "data_quality", rows=len(sheets["점검 요약"]))
        for r in sheets["점검 요약"].query("상태 == '경고'").to_dict("records"):
            run.warn(r["점검 항목"], value=r["값"], note=r["비고"])
        st["rows_out"] = len(sheets["점검 요약"])
    m = run.finish()
    print(f"{run.run_id} · {snap['data_snapshot_id']} · 경고 {len(m['warnings'])}건 → {path}")
    return m
