"""기본 원자료 형식 확인과 운영 포인터 갱신."""
from __future__ import annotations

import pandas as pd
import pytest

from trend_radar.chain import set_pointer
from trend_radar.e0 import SchemaError, check_schema

CFG = {"input": {"columns": {"article_id": "전역기사ID", "title": "제목"}, "schema_column": "공용원자료스키마",
                 "schema_versions": ["2026-07-30_v9_news_dedup"]}}


def test_check_schema_accepts_v9_and_counts():
    raw = pd.DataFrame({"전역기사ID": ["a", "b"], "제목": ["x", "y"], "공용원자료스키마": ["2026-07-30_v9_news_dedup"] * 2})
    assert check_schema(raw, CFG) == {"2026-07-30_v9_news_dedup": 2}


def test_check_schema_rejects_missing_column_and_unknown_version():
    with pytest.raises(SchemaError, match="필수 열"):
        check_schema(pd.DataFrame({"전역기사ID": ["a"]}), CFG)
    raw = pd.DataFrame({"전역기사ID": ["a"], "제목": ["x"], "공용원자료스키마": ["2027-01-01_v10"]})
    with pytest.raises(SchemaError, match="허용되지 않은 스키마 버전"):
        check_schema(raw, CFG)
    with pytest.raises(SchemaError, match="스키마 열"):
        check_schema(pd.DataFrame({"전역기사ID": ["a"], "제목": ["x"]}), CFG)


def test_set_pointer_changes_only_that_line(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("e2:\n  selected_model_id: e2_1_operational\n  selected_run_id: run_old  # 주석\n  candidate_run_id: null\n", encoding="utf-8")
    set_pointer("run_new", p)
    assert p.read_text(encoding="utf-8") == "e2:\n  selected_model_id: e2_1_operational\n  selected_run_id: run_new  # 주석\n  candidate_run_id: null\n"
