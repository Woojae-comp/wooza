"""실행 기록과 재현성 (v2 설계 2장, 기획안 11장).

- data_snapshot_id: 입력 파일 해시로 고정한 기사 집합
- run_id: 한 번의 실행. out/logs/{run_id}/run_manifest.json과 events.jsonl을 남긴다
- registry/*.jsonl: 데이터셋·실행·실험·결정 이력 (저장소에 커밋)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import ROOT

REGISTRY = ROOT / "registry"
KST = dt.timezone(dt.timedelta(hours=9))


def now() -> str:
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(cfg: dict) -> str:
    return "sha256:" + sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str))


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def environment() -> dict:
    pkgs = {}
    for name in ("pandas", "numpy", "scipy", "networkx", "kiwipiepy", "sklearn", "anthropic"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "")
        except Exception:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def snapshot(files: list[str | Path], last_date: str, register: bool = True) -> dict:
    """입력 파일 목록으로 data_snapshot_id를 만든다. 같은 파일이면 같은 ID."""
    entries = [{"path": str(Path(f).name), "bytes": Path(f).stat().st_size, "sha256": sha256_file(f)} for f in files]
    joined = "".join(sorted(e["sha256"] for e in entries))
    sid = f"ds_{last_date.replace('-', '')}_{sha256_text(joined)[:8]}"
    rec = {"data_snapshot_id": sid, "created_at": now(), "files": entries, "last_date": last_date}
    if register:
        existing = set()
        p = REGISTRY / "dataset_registry.jsonl"
        if p.exists():
            existing = {json.loads(l)["data_snapshot_id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
        if sid not in existing:
            append_jsonl(p, rec)
    return rec


class Run:
    """한 번의 실행. 단계별 시작·종료·행 수와 산출물 체크섬을 기록한다."""

    def __init__(self, kind: str, cfg: dict, out_root: Path, snapshot_id: str | None = None):
        stamp = dt.datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{stamp}_{uuid.uuid4().hex[:4]}"
        self.kind = kind
        self.dir = out_root / "runs" / self.run_id
        self.log_dir = out_root / "logs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id, "kind": kind, "status": "RUNNING", "started_at": now(), "ended_at": None,
            "data_snapshot_id": snapshot_id, "git": git_info(), "config_hash": config_hash(cfg),
            "random_seed": cfg.get("reproducibility", {}).get("random_seed"),
            "environment": environment(), "stages": [], "warnings": [], "artifacts": [],
        }
        (self.log_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        self._write()

    def _write(self) -> None:
        (self.log_dir / "run_manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def event(self, kind: str, **data) -> None:
        append_jsonl(self.log_dir / "events.jsonl", {"at": now(), "event": kind, **data})

    @contextmanager
    def stage(self, name: str, rows_in: int | None = None):
        st = {"name": name, "status": "RUNNING", "started_at": now(), "rows_in": rows_in}
        self.manifest["stages"].append(st)
        self._write()
        t0 = time.time()
        self.event("stage_start", stage=name)
        try:
            yield st
            st["status"] = "SUCCESS"
        except Exception as e:
            st["status"] = "FAILED"
            st["error"] = repr(e)
            self.manifest["status"] = "FAILED"
            raise
        finally:
            st["ended_at"] = now()
            st["seconds"] = round(time.time() - t0, 1)
            self.event("stage_end", stage=name, status=st["status"])
            self._write()

    def warn(self, message: str, **data) -> None:
        self.manifest["warnings"].append({"message": message, **data})
        self.event("warning", message=message, **data)
        self._write()

    def artifact(self, path: Path, name: str, rows: int | None = None) -> str:
        digest = sha256_file(path)
        aid = f"art_{name}_{digest[:8]}"
        self.manifest["artifacts"].append({"artifact_id": aid, "name": name, "path": str(path), "sha256": digest, "rows": rows})
        self._write()
        return aid

    def finish(self) -> dict:
        if self.manifest["status"] == "RUNNING":
            self.manifest["status"] = "SUCCESS"
        self.manifest["ended_at"] = now()
        self._write()
        summary = {k: self.manifest[k] for k in ("run_id", "kind", "status", "started_at", "ended_at",
                                                  "data_snapshot_id", "config_hash")}
        summary["git_commit"] = self.manifest["git"]["commit"]
        summary["artifacts"] = [a["artifact_id"] for a in self.manifest["artifacts"]]
        summary["warnings"] = len(self.manifest["warnings"])
        append_jsonl(REGISTRY / "run_registry.jsonl", summary)
        return self.manifest


def log_decision(target_type: str, target_id: str, previous_value, new_value, reason_code: str, reason_text: str,
                 reviewer: str, evidence_refs: list | None = None, apply_from: str | None = None) -> dict:
    """사람 결정 기록 (기획안 부록 5). 불용어·병합·임계값·모델 교체·라벨 재분류."""
    rec = {"review_id": f"rev_{dt.datetime.now(KST).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:3]}",
           "target_type": target_type, "target_id": target_id, "previous_value": previous_value,
           "new_value": new_value, "reason_code": reason_code, "reason_text": reason_text, "reviewer": reviewer,
           "reviewed_at": now(), "evidence_refs": evidence_refs or [], "apply_from_version": apply_from}
    append_jsonl(REGISTRY / "decision_log.jsonl", rec)
    return rec
