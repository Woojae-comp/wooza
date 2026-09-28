"""설정 로드. trend_radar.yaml(분석 설정)과 lexicon.yaml(사용자 승인 정제 규칙)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "trend_radar.yaml"
DEFAULT_LEXICON = ROOT / "lexicon.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _merge(cfg, overrides or {})


class Lexicon:
    """사용자가 승인한 불용어·동의어·복합어. 자동으로 채우지 않는다."""

    def __init__(self, stopwords=None, synonyms=None, compounds=None):
        self.stopwords: set[str] = set(stopwords or [])
        self.synonym_map: dict[str, str] = {}
        for canon, variants in (synonyms or {}).items():
            for v in variants or []:
                self.synonym_map[v] = canon
        # "영업+이익" → ("영업", "이익")
        self.compounds: list[tuple[str, ...]] = []
        for c in compounds or []:
            parts = tuple(p.strip() for p in str(c).split("+") if p.strip())
            if len(parts) >= 2:
                self.compounds.append(parts)
        self.compounds.sort(key=len, reverse=True)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Lexicon":
        p = Path(path) if path else DEFAULT_LEXICON
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(d.get("stopwords"), d.get("synonyms"), d.get("compounds"))

    def normalize(self, word: str) -> str | None:
        w = self.synonym_map.get(word, word)
        return None if w in self.stopwords or word in self.stopwords else w
