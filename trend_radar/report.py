"""Trend Radar 화면 (설계 17장). radar.json을 넣은 단일 HTML."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("radar_template.html")


LIST_KEYS = ("cross_sector", "emerging", "growing", "declining", "event", "spreading", "converging", "structural")


def _round(o, nd: int = 4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def _art_ids(o, out: set) -> None:
    """결과 안의 대표 기사 참조(arts)를 모은다."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "arts":
                if isinstance(v, dict):
                    for l in v.values():
                        out.update(str(i) for i in l)
                elif isinstance(v, list):
                    out.update(str(i) for i in v)
            else:
                _art_ids(v, out)
    elif isinstance(o, list):
        for v in o:
            _art_ids(v, out)


def _slim(result: dict, top_profiles: int = 300) -> dict:
    """화면에 필요한 만큼만 남긴다 (전체는 radar.json과 tables/에 있다).
    목록·분야·군집·쌍에 나온 키워드 + 기사 수 상위 키워드의 프로파일만 싣는다."""
    r = dict(result)
    prof = result["profiles"]
    keep = set(result["overall"]["rising"]) | set(result["overall"]["established"])
    for k in LIST_KEYS:
        keep.update(result[k])
    for sec in result["sectors"].values():
        for key in ("rising", "specialized", "established", "declining", "event"):
            keep.update(x["k"] for x in sec[key])
    for pr in result["keyword_pairs"]:
        keep.update((pr["a"], pr["b"]))
    keep.update(b["keyword"] for b in result["bridges"])
    keep.update(sorted(prof, key=lambda k: -prof[k]["articles"])[:top_profiles])
    slim = {}
    for k in keep:
        if k not in prof:
            continue
        p = dict(prof[k])
        if p.get("structure"):
            st = {x: p["structure"].get(x) for x in ("early_df", "recent_df", "assoc_overlap", "profile_similarity",
                                                     "early_assoc", "recent_assoc", "converging_links", "early_cluster", "early_context", "recent_context",
                                                     "recent_cluster", "cluster_moved", "early_degree", "recent_degree")}
            st["converging_links"] = (st["converging_links"] or [])[:6]
            p["structure"] = st
        slim[k] = p
    r["profiles"] = _round(slim)
    ids: set[str] = set()
    _art_ids({k: v for k, v in r.items() if k != "articles"}, ids)
    # 화면용 기사는 요약을 줄인다 (앱 미리보기에서 열리도록 파일 크기를 줄이기 위함)
    r["articles"] = {i: dict(a, s=a["s"][:100]) for i, a in result["articles"].items() if i in ids}
    meta = dict(r["meta"])
    meta.pop("config", None)
    r["meta"] = meta
    nets = dict(r["networks"])
    for k in ("recent", "first"):
        n = dict(nets[k])
        n.pop("edges", None)  # 엣지·노드 전체는 tables/edges_*.csv, centrality_*.csv
        n.pop("nodes", None)
        nets[k] = n
    r["networks"] = nets
    r["sectors"] = _round(r["sectors"])
    r["sector_convergence"] = _round(r["sector_convergence"])
    return r


def write_html(result: dict, path: Path) -> None:
    from .pipeline import _jsonable

    data = json.dumps(_slim(result), ensure_ascii=False, default=_jsonable).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
    Path(path).write_text(html, encoding="utf-8")
