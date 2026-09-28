"""python -m trend_radar run [--input 경로...] [--lexicon lexicon.yaml] [--out out]
python -m trend_radar html [--out out]"""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(prog="trend_radar", description="콘텐츠산업 뉴스 Trend Radar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="분석 실행 → out/radar.html, out/tables/, out/review/")
    r.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    r.add_argument("--config", help="trend_radar.yaml 대신 쓸 설정 (기본값 위에 덮어씀)")
    r.add_argument("--lexicon", help="승인된 정제 규칙 (기본 lexicon.yaml)")
    r.add_argument("--out", help="출력 폴더 (기본 out)")
    r.add_argument("--no-candidates", action="store_true", help="불용어·동의어·복합어 후보 생략")
    h = sub.add_parser("html", help="radar.json에서 radar.html만 다시 만든다")
    h.add_argument("--out", default="out")
    a = ap.parse_args()
    if a.cmd == "html":
        import json
        from pathlib import Path

        from .report import write_html
        out = Path(a.out)
        write_html(json.loads((out / "radar.json").read_text(encoding="utf-8")), out / "radar.html")
        print(out / "radar.html")
        return
    cfg = load_config(a.config)
    if a.cmd == "run":
        run(cfg, a.input, a.lexicon, a.out, with_candidates=not a.no_candidates)


if __name__ == "__main__":
    main()
