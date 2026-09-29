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
    ex = sub.add_parser("export", help="대시보드 평가 묶음: 주제 트렌드·월별 비중·키워드 신호·레이더 목록 TSV + 평가 열 (out/eval/*.zip)")
    ex.add_argument("--out", default="out")
    al = sub.add_parser("all", help="기본 원자료 → 형식 확인·E0·E2(승인 모델 재적용)·E3·E4.2·E5a·E5b·화면을 한 번에")
    al.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    al.add_argument("--out", default="out")
    al.add_argument("--min-df", type=int, default=5)
    e0 = sub.add_parser("e0", help="E0: 스냅샷 등록 + 데이터 품질 점검 (01_data_quality.xlsx)")
    e0.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e0.add_argument("--out", default="out")
    e2 = sub.add_parser("e2", help="E2: 기사 관련성 실험 (약한 지도학습, 앵커 평가) → 02_article_relevance.parquet")
    e2.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e2.add_argument("--out", default="out")
    e2.add_argument("--llm-dir", default="out/llm", help="LLM 도전자 결과 폴더 (relevance_pred.csv)")
    e22 = sub.add_parser("e22", help="E2.2: 관련성 학습 방식 비교 (base/softA/softB × 검색 기업명 처리) → 선정 모델 02_article_relevance.parquet")
    e22.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e22.add_argument("--out", default="out")
    e3 = sub.add_parser("e3", help="E3: 핵심어 사전 실험 (A Strict / B Half / C Soft) → 03_keyword_dictionary.csv")
    e3.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e3.add_argument("--out", default="out")
    e3.add_argument("--min-df", type=int, default=5)
    e4 = sub.add_parser("e4", help="E4: 주제 군집·네트워크 실험 (문서 K-means + NPMI·Leiden 결합) → 07_topic_registry.csv")
    e4.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e4.add_argument("--out", default="out")
    e41 = sub.add_parser("e41", help="E4.1: 문서 군집 보완 (A Leiden 기사 배정 / B LSA K-means / C 분야별 층화 + 계층형 주제)")
    e41.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e41.add_argument("--out", default="out")
    e42 = sub.add_parser("e42", help="E4.2: 주제 구조 확정 (LSA100 K120 + Leiden 해석 + 층화 복구) → 07c_topic_registry.csv")
    e42.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e42.add_argument("--out", default="out")
    e5 = sub.add_parser("e5", help="E5a: 키워드 시계열 신호 실험 (trend 아님) → 06a_keyword_signal.csv")
    e5.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e5.add_argument("--out", default="out")
    e5b = sub.add_parser("e5b", help="E5b: 주제 트렌드 확정 (고정 모델 월별 배정 + 주제 신호 + E4.3 계보) → 06b_topic_trend.csv")
    e5b.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e5b.add_argument("--out", default="out")
    e23 = sub.add_parser("e23", help="E2.3: 3분류(CONTENT/MARKET/OTHER) 라벨 모델 비교 → 통과 시 후보 E2")
    e23.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e23.add_argument("--out", default="out")
    e23b = sub.add_parser("e23b", help="E2.3b: 라벨 모델 보정 진단 (근거 계열 통합, G3 vs H2 × 사전분포 L/M/H)")
    e23b.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e23b.add_argument("--out", default="out")
    e23c = sub.add_parser("e23c", help="E2.3c: 사건 이슈 계열 보강 (H2 + 콘텐츠 기업 사건 근거 + REVIEW 하한), 검토 표본 v1.1 합의와 대조")
    e23c.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    e23c.add_argument("--out", default="out")
    e23c.add_argument("--review-dir", help="out/review/review_YYYYMMDD_HHMMSS (병합된 LLM 판정과 대조)")
    for p_ in (e3, e4, e41, e42, e5):
        p_.add_argument("--use-candidate", action="store_true", help="승인 전 후보 E2(e2.candidate_run_id)로 실행")
    for p_ in (e42, e5):
        p_.add_argument("--weighting", choices=["half", "half_x_pcontent"], help="기사 가중 방식 (relevance_weighting.mode 덮어씀)")
    rv = sub.add_parser("review-sample", help="E2 비교·검토 표본 600건 추출 + 초기 60건 업로드 파일 (LLM API 호출 없음)")
    rv.add_argument("--input", nargs="+", help="원자료 경로 (xlsx/csv/parquet, glob 가능)")
    rv.add_argument("--out", default="out")
    rv.add_argument("--seed", type=int, default=20260929)
    rm = sub.add_parser("review-merge", help="저장된 LLM 응답(model_responses/)을 검사·병합하고 비교 보고서 작성")
    rm.add_argument("--dir", required=True, help="out/review/review_YYYYMMDD_HHMMSS")
    rb = sub.add_parser("review-batches", help="남은 검토 표본으로 다음 입력 묶음 생성 (현재 가이드 버전)")
    rb.add_argument("--dir", required=True)
    rb.add_argument("--split", default="dev", choices=["dev", "holdout"])
    rb.add_argument("--size", type=int, default=30)
    h = sub.add_parser("html", help="radar.json에서 radar.html만 다시 만든다")
    h.add_argument("--out", default="out")
    a = ap.parse_args()
    if a.cmd == "review-sample":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .load import read_raw
        from .review import build_review_sample
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        print(json.dumps(build_review_sample(cfg, read_raw(files), Path(a.out), a.seed), ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "review-batches":
        import json
        from pathlib import Path

        from .review import make_batches
        print(json.dumps(make_batches(Path(a.dir), a.split, a.size), ensure_ascii=False, indent=1))
        return
    if a.cmd == "review-merge":
        import json
        from pathlib import Path

        from .review import merge_responses
        print(json.dumps(merge_responses(Path(a.dir)), ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "export":
        import json
        from pathlib import Path

        from .export import export_pack
        print(json.dumps(export_pack(Path(a.out)), ensure_ascii=False, indent=1))
        return
    if a.cmd == "all":
        import json

        from .chain import run_all
        print(json.dumps(run_all(load_config(), a.input, a.out, a.min_df), ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e5b":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e5b import run_e5b
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        s = run_e5b(cfg, read_raw(files), Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if k != "window_purity"}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e23c":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e23c import run_e23c
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        s = run_e23c(cfg, read_raw(files), Path(a.out), Path(a.review_dir) if a.review_dir else None)
        print(json.dumps({k: v for k, v in s.items() if k != "comparison"}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e23b":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e23b import run_e23b
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        s = run_e23b(cfg, read_raw(files), Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if k not in ("models", "family_removal")}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e23":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e23 import run_e23
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        s = run_e23(cfg, read_raw(files), Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if k not in ("models", "lf_stats")}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e4":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e4 import run_e4
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        s = run_e4(cfg, read_raw(files), Path(a.out))
        print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e41":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e41 import run_e41
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        s = run_e41(cfg, read_raw(files), Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if k not in ("models", "backbones")}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e42":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e42 import run_e42
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        if getattr(a, "weighting", None):
            cfg["relevance_weighting"] = {**(cfg.get("relevance_weighting") or {}), "mode": a.weighting}
        s = run_e42(cfg, read_raw(files), Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if k != "networks"}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e5":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e5 import run_e5a
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        if getattr(a, "weighting", None):
            cfg["relevance_weighting"] = {**(cfg.get("relevance_weighting") or {}), "mode": a.weighting}
        s = run_e5a(cfg, read_raw(files), Path(a.out))
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return
    if a.cmd == "e22":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e22 import run_e22
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        s = run_e22(cfg, read_raw(files), Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if k != "variants"}, ensure_ascii=False, indent=1, default=str))
        return
    if a.cmd == "e3":
        import json
        from pathlib import Path

        from .e0 import input_files
        from .e3 import run_e3
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        s = run_e3(cfg, read_raw(files), Path(a.out), a.min_df)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return
    if a.cmd == "e2":
        from pathlib import Path

        from .e0 import input_files
        from .e2 import run_e2
        from .load import read_raw
        cfg = load_config()
        files = input_files(a.input or cfg["input"]["paths"])
        cfg["_input_files"] = files
        cfg["_use_candidate"] = getattr(a, "use_candidate", False)
        s = run_e2(cfg, read_raw(files), Path(a.out), Path(a.llm_dir))
        import json
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return
    if a.cmd == "e0":
        from .e0 import run_e0
        run_e0(load_config(), a.input, a.out)
        return
    if a.cmd == "html":
        import json
        from pathlib import Path

        from .report import write_html
        out = Path(a.out)
        layers = {k: json.loads((out / k / "radar.json").read_text(encoding="utf-8"))
                  for k in ("content", "market") if (out / k / "radar.json").exists()}
        write_html(layers or json.loads((out / "radar.json").read_text(encoding="utf-8")), out / "radar.html")
        print(out / "radar.html")
        return
    cfg = load_config(a.config)
    if a.cmd == "run":
        run(cfg, a.input, a.lexicon, a.out, with_candidates=not a.no_candidates)


if __name__ == "__main__":
    main()
