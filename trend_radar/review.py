"""E2 결과 비교·검토 (파일 업로드 방식, 2026-09-29 검토 절차 명세 v1.0).

분석은 기존 방법(E2~E5)으로 수행하고, LLM은 결과를 검토한다.
- 이 모듈은 LLM API를 호출하지 않는다. 사용자가 같은 파일을 여러 LLM 대화창에 직접 업로드하고 응답을 저장한다.
- LLM 응답은 보조 검토 의견이며 정답·학습 라벨로 자동 전환하지 않는다 (학습 함수에 연결하지 않음).
- 운영 포인터(e2.selected_run_id)를 바꾸지 않는다.

build_review_sample: 비교 표본 600 = 대표표본 400 (E2.1 판정 × 섹션 코드 유무 6층, 모집단 비례) + 경계·진단표본 200.
  유사 기사 묶음(중복기사군·정규화 제목) 단위로 개발용 200 / 보관 비교용 400을 나눈다. 초기 60건(30건 × 2묶음)은 개발용에서.
merge_responses: 저장된 응답의 ID 누락·중복·미지 ID, 허용값, 근거 문구 존재 여부를 검사하고 모델 간 합의·불일치를 정리한다.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

GUIDE_VERSION = "1.0"
OUT_COLS = ["article_id", "content_relevance", "market_focus", "review_class", "content_evidence", "exclusion_evidence",
            "reason_short", "needs_more_context"]
ALLOWED = {"content_relevance": {"SUBSTANTIVE", "INCIDENTAL", "NONE", "UNCERTAIN"},
           "market_focus": {"YES", "NO", "UNCERTAIN"},
           "review_class": {"CONTENT", "MARKET", "OTHER", "UNRESOLVED"},
           "needs_more_context": {"YES", "NO"}}
UPLOAD_GUIDE = """# 업로드 순서 (검토 절차 명세 v1.0)

## 1차 — 기존 판정을 가리고 기사 근거 검토
각 LLM의 **새 대화**에 `upload/stage1/`의 파일만 올린다. 다른 LLM 응답은 보여 주지 않는다.
1. `annotation_guide.md` 와 `review_input_batch_001.tsv` 를 첨부
2. `review_prompt.md` 의 본문을 그대로 붙여 넣기 (지시문을 바꿔 원하는 답을 유도하지 않는다)
3. 응답 TSV를 `model_responses/batch_001__{서비스}_{화면 표시 모델명}.tsv` 로 저장 (원문 그대로)
4. `model_responses/responses_log.csv` 에 서비스·화면 표시 모델명·사용 일시·재시도 사유를 기록 (확인할 수 없는 값은 '확인 불가')
5. batch_002도 같은 방식 (새 대화 권장)

`review_output_template_batch_*.tsv` 는 응답 형식 확인용이다 (선택적으로 함께 첨부).

## 병합
`python -m trend_radar review-merge --dir <이 폴더>` → `e2_llm_comparison_report.md`, `merged/priority_review.csv`

## 2차 — 후보 비교 (1차가 끝난 뒤에만)
`upload/stage2/model_comparison_pack.md` 에는 기존 E2 판정이 들어 있다. **1차에서 절대 올리지 않는다.**
`annotation_guide.md` + `model_comparison_pack.md` 첨부, `review_prompt_compare.md` 본문 붙여 넣기. 후보 이름 대응표는 `internal/` (올리지 않음).

## 올리지 않는 파일
`internal/` (추출 사유·앵커·탐침·후보 대응표·원문 링크), `review_manifest.json`
"""

DIAG_CATEGORIES = ["mixed_content_market", "nosid_politics_probe", "content_policy", "broadcaster_source", "model_changed"]


# ---------------------------------------------------------------- 공통

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean_cell(s) -> str:
    return re.sub(r"[\t\r\n]+", " ", "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)).strip()


def norm_text(s: str) -> str:
    """근거 대조용 정규화 (고정): NFKC, 따옴표·말줄임 통일, 공백 하나로."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[‘’´`]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = s.replace("…", "...")
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_title(t: str) -> str:
    """유사 기사 묶음용 제목 정규화: 말머리·괄호·기호·공백 제거."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|【[^】]*】", " ", unicodedata.normalize("NFKC", t or ""))
    return re.sub(r"[^0-9a-z가-힣]", "", t.lower())


def duplicate_groups(gids: pd.Series, titles: pd.Series, dup_group: pd.Series | None) -> pd.Series:
    """묶음 ID: 원자료 중복기사군ID가 있으면 그것, 없으면 정규화 제목. 같은 기업·작품이라는 이유만으로 묶지 않는다."""
    nt = titles.map(norm_title)
    title_key = "t:" + nt.map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:12])
    if dup_group is None:
        return title_key
    dg = gids.map(dup_group)
    g = np.where(dg.notna() & (dg.astype(str) != "") & (dg.astype(str) != "nan"), "d:" + dg.astype(str), title_key)
    # 같은 정규화 제목은 서로 다른 중복기사군이어도 한 묶음 (재전송·유사 기사)
    s = pd.Series(g, index=gids.index)
    first = s.groupby(nt).transform("first")
    return first


def allocate(sizes: dict, n: int, cap: bool = True) -> dict:
    """최대 나머지 비례 배분. cap: 각 층 모집단 크기를 넘지 않게 (가중치만 줄 때는 False)."""
    tot = sum(sizes.values())
    raw = {k: n * v / tot for k, v in sizes.items()}
    out = {k: int(np.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[: n - sum(out.values())]:
        out[k] += 1
    return {k: min(v, sizes[k]) for k, v in out.items()} if cap else out


def draw(pool: pd.DataFrame, n: int, rng, taken_groups: set) -> tuple[pd.DataFrame, int]:
    """묶음이 겹치지 않게 n건 무작위 추출. 반환: 표본, 추출 대상 모집단 크기(이미 뽑힌 묶음 제외)."""
    avail = pool[~pool["group_id"].isin(taken_groups)]
    order = rng.permutation(len(avail))
    picked, seen = [], set()
    for i in order:
        g = avail["group_id"].iat[i]
        if g in seen:
            continue
        seen.add(g)
        picked.append(i)
        if len(picked) >= n:
            break
    return avail.iloc[picked], len(avail)


def split_by_group(sample: pd.DataFrame, n_dev: int, rng) -> pd.Series:
    groups = np.array(list(pd.unique(sample["group_id"])), dtype=object)
    rng.shuffle(groups)
    dev, cnt = set(), 0
    size = sample.groupby("group_id").size()
    for g in groups:
        if cnt >= n_dev:
            break
        dev.add(g)
        cnt += size[g]
    return np.where(sample["group_id"].isin(dev), "dev", "holdout")


# ---------------------------------------------------------------- 표본 추출

def build_review_sample(cfg: dict, raw: pd.DataFrame, out_root: Path, seed: int = 20260929) -> dict:
    from .e2 import PRICE_TITLE
    from .e23 import AB
    from .e23b import candidate_decisions, prepare
    from .runlog import REGISTRY, ROOT, snapshot
    from .selection import selected_e2

    rs = cfg.get("review", {})
    rng = np.random.default_rng(seed)
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    base_run, base_path, _ = selected_e2(cfg, REGISTRY, out_root, use_candidate=False)
    base = pd.read_parquet(base_path).set_index("gid")

    p = prepare(cfg, raw, out_root)
    arts, anc, F = p["arts"].reset_index(drop=True), p["anc"].reset_index(drop=True), p["F"].reset_index(drop=True)
    cand = candidate_decisions(p, "M")
    colid = cfg["input"]["columns"]
    rdd = raw.drop_duplicates(colid["article_id"]).set_index(colid["article_id"])
    dup = rdd["중복기사군ID"] if "중복기사군ID" in rdd else None
    df = pd.DataFrame({
        "gid": arts["gid"], "date": arts["date"].dt.date.astype(str), "press": arts["press"], "title": arts["title"],
        "summary": arts["summary"], "link": arts["link"],
        "e21_decision": arts["gid"].map(base["decision"]).to_numpy(),
        "has_sid": anc["sid"].notna().to_numpy(), "sid": anc["sid"].to_numpy(),
        "anchor_content": anc["anchor_content"].to_numpy().astype(bool),
    })
    df["group_id"] = duplicate_groups(df["gid"], df["title"], dup).to_numpy()
    # 후보 판정 (비교용)
    e22_recs = [json.loads(l) for l in (REGISTRY / "candidate_status.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    e22_run = next((r["run_id"] for r in e22_recs if r.get("model_id") == "e2_2"), None)
    cands = {"E2.1_operational": df["e21_decision"].to_numpy()}
    if e22_run and (out_root / "runs" / e22_run / "02_article_relevance.parquet").exists():
        e22 = pd.read_parquet(out_root / "runs" / e22_run / "02_article_relevance.parquet").set_index("gid")
        cands["E2.2_noco_src_base"] = df["gid"].map(e22["decision"]).to_numpy()
    for k, (dec, _) in cand.items():
        cands[f"E2.3b_{k}_PriorM"] = dec
    for k, v in cands.items():
        df[f"cand::{k}"] = v
    inn = {k: np.isin(v, ["INCLUDE", "REVIEW"]) for k, v in cands.items()}
    changed = np.zeros(len(df), bool)
    for k, v in inn.items():
        if k != "E2.1_operational":
            changed |= v != inn["E2.1_operational"]
    mseed = set(cfg["layers"]["market_seed"])
    df["flag_mixed_content_market"] = df["anchor_content"] & (
        df["title"].fillna("").str.contains(PRICE_TITLE).to_numpy() | (p["lab"]["market_score"].to_numpy() > 0))
    df["flag_nosid_politics_probe"] = ~df["has_sid"] & np.array(
        [any(w in f"{t} {s}" for w in cfg["e2"].get("politics_probe_words", [])) for t, s in zip(df["title"], df["summary"])])
    df["flag_content_policy"] = (F["F_content_policy"] != AB).to_numpy()
    df["flag_broadcaster_source"] = (F["F_broadcaster_source"] != AB).to_numpy()
    df["flag_model_changed"] = changed

    # 대표표본 400: E2.1 판정 × 섹션 코드 유무 6층, 모집단 비례
    n_rep, n_diag = rs.get("representative_n", 400), rs.get("diagnostic_n", 200)
    df["stratum"] = df["e21_decision"].astype(str) + "|" + np.where(df["has_sid"], "sid", "nosid")
    N_h = df["stratum"].value_counts().to_dict()
    alloc = allocate(N_h, n_rep)
    parts, taken = [], set()
    for h, n in alloc.items():
        s, N_avail = draw(df[df["stratum"] == h], n, rng, taken)
        s = s.assign(sample_set="representative", reason=f"stratum:{h}", inclusion_prob=len(s) / N_h[h],
                     stratum_N=N_h[h], stratum_n=len(s))
        taken |= set(s["group_id"])
        parts.append(s)
    rep = pd.concat(parts)
    rep["weight"] = 1 / rep["inclusion_prob"]
    # 경계·진단표본 200: 범주별 균등 할당 (대표표본 묶음 제외)
    quota = allocate({c: 1 for c in DIAG_CATEGORIES}, n_diag, cap=False)
    dparts = []
    for c in DIAG_CATEGORIES:
        pool = df[df[f"flag_{c}"]]
        s, N_avail = draw(pool, quota[c], rng, taken)
        s = s.assign(sample_set="diagnostic", reason=f"diagnostic:{c}", inclusion_prob=len(s) / max(N_avail, 1),
                     stratum_N=N_avail, stratum_n=len(s), weight=np.nan)
        taken |= set(s["group_id"])
        dparts.append(s)
    diag = pd.concat(dparts)
    # 개발용 / 보관 비교용 (묶음 단위)
    rep["split"] = split_by_group(rep, rs.get("representative_dev_n", 100), rng)
    diag["split"] = split_by_group(diag, rs.get("diagnostic_dev_n", 100), rng)
    sample = pd.concat([rep, diag], ignore_index=True)
    assert not (sample.groupby("group_id")["split"].nunique() > 1).any(), "한 묶음이 개발용·보관용에 걸침"
    assert sample["gid"].is_unique
    # 초기 60건: 개발용에서 대표·진단 30건씩, 무작위 순서로 30건 × 2묶음 (추출 사유가 드러나지 않게 섞음)
    n_init = rs.get("initial_n", 60)
    init = pd.concat([sample[(sample["split"] == "dev") & (sample["sample_set"] == ss)].sample(
        n_init // 2, random_state=int(rng.integers(1e9))) for ss in ("representative", "diagnostic")])
    init = init.sample(frac=1, random_state=int(rng.integers(1e9)))
    bsz = rs.get("batch_size", 30)
    sample["batch"] = ""
    for b in range(int(np.ceil(len(init) / bsz))):
        ids = init["gid"].iloc[b * bsz:(b + 1) * bsz]
        sample.loc[sample["gid"].isin(ids), "batch"] = f"batch_{b + 1:03d}"

    # 파일 쓰기
    review_id = f"review_{pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S')}"
    d = out_root / "review" / review_id
    (d / "upload" / "stage1").mkdir(parents=True)
    (d / "upload" / "stage2").mkdir()
    (d / "internal").mkdir()
    (d / "model_responses").mkdir()
    docs = ROOT / "docs" / "review"
    for f, stage in (("annotation_guide.md", "stage1"), ("review_prompt.md", "stage1"), ("annotation_guide.md", "stage2"),
                     ("review_prompt_compare.md", "stage2")):
        shutil.copy(docs / f, d / "upload" / stage / f)
    (d / "UPLOAD_GUIDE.md").write_text(UPLOAD_GUIDE, encoding="utf-8")
    hashes = {}
    for b in sorted(x for x in sample["batch"].unique() if x):
        rows = sample[sample["batch"] == b]
        inp = pd.DataFrame({"article_id": rows["gid"], "title": rows["title"].map(clean_cell), "summary": rows["summary"].map(clean_cell)})
        pi = d / "upload" / "stage1" / f"review_input_{b}.tsv"
        inp.to_csv(pi, sep="\t", index=False)
        tmpl = pd.DataFrame({c: (rows["gid"].to_numpy() if c == "article_id" else "") for c in OUT_COLS})
        pt = d / "upload" / "stage1" / f"review_output_template_{b}.tsv"
        tmpl.to_csv(pt, sep="\t", index=False)
        hashes[pi.name], hashes[pt.name] = sha256_file(pi), sha256_file(pt)
    # 2차 비교 자료 (초기 60건, 후보 익명화)
    names = list(cands)
    perm = rng.permutation(len(names))
    key = {f"후보 {chr(65 + i)}": names[j] for i, j in enumerate(perm)}
    inv = {v: k for k, v in key.items()}
    (d / "internal" / "candidate_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    pack = comparison_pack(df, sample[sample["batch"] != ""], cands, inv)
    (d / "upload" / "stage2" / "model_comparison_pack.md").write_text(pack, encoding="utf-8")
    hashes["model_comparison_pack.md"] = sha256_file(d / "upload" / "stage2" / "model_comparison_pack.md")
    sample.drop(columns=["summary"]).to_csv(d / "internal" / "sample_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["file", "service", "displayed_model", "used_at_kst", "settings_visible", "batch", "rows_returned",
                          "retries", "retry_reason", "manual_edits"]).to_csv(d / "model_responses" / "responses_log.csv", index=False,
                                                                                encoding="utf-8-sig")
    (d / "model_responses" / "README.md").write_text(
        "# 모델 응답 저장\n\n- 파일 이름: `{batch}__{서비스}_{화면 표시 모델명}.tsv` (예: `batch_001__claude_opus.tsv`).\n"
        "- 원래 응답을 그대로 저장한다 (코드 블록·설명문이 섞여도 됨, 병합 코드가 TSV 부분만 읽는다). 수동 수정은 responses_log.csv에 기록.\n"
        "- 화면에서 모델 버전·설정을 확인할 수 없으면 '확인 불가'로 기록하고 추정해 채우지 않는다.\n"
        "- 새 대화에서 진행하고 다른 모델의 응답을 먼저 보여 주지 않는다. 재시도하면 사유와 횟수를 기록한다.\n"
        "- 병합: `python -m trend_radar review-merge --dir <이 폴더의 상위>`\n", encoding="utf-8")
    manifest = {
        "review_id": review_id, "guide_version": GUIDE_VERSION,
        "guide_sha256": sha256_file(docs / "annotation_guide.md"), "prompt_stage1_sha256": sha256_file(docs / "review_prompt.md"),
        "prompt_stage2_sha256": sha256_file(docs / "review_prompt_compare.md"),
        "data_snapshot_id": snap["data_snapshot_id"], "baseline_e2_run_id": base_run,
        "candidates": {"E2.2_noco_src_base": e22_run, "E2.3b": "재계산 (prepare + candidate_decisions, Prior-M)"},
        "seed": seed, "sample": {"representative": int((sample["sample_set"] == "representative").sum()),
                                 "diagnostic": int((sample["sample_set"] == "diagnostic").sum()),
                                 "dev": int((sample["split"] == "dev").sum()), "holdout": int((sample["split"] == "holdout").sum()),
                                 "initial_batches": {b: int((sample["batch"] == b).sum()) for b in sorted(x for x in sample["batch"].unique() if x)}},
        "strata_population": N_h, "representative_allocation": alloc, "diagnostic_quota": quota,
        "diagnostic_pool_sizes": {c: int(df[f"flag_{c}"].sum()) for c in DIAG_CATEGORIES},
        "input_file_sha256": hashes, "llm_api_used": False,
        "notes": ["LLM 응답은 보조 검토 의견이며 정답·학습 라벨이 아니다", "운영 포인터 변경 없음",
                  "보관 비교용(holdout)은 가이드·후보·지표 고정 뒤 사용. 보고 규칙을 바꾸면 개발자료로 취급",
                  "대표·진단 표본 결과를 합쳐 전체 비율로 보고하지 않는다. 대표표본 비율은 weight(1/포함확률)로 가중"],
    }
    (d / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return manifest | {"dir": str(d)}


def comparison_pack(df: pd.DataFrame, rows: pd.DataFrame, cands: dict, inv: dict) -> str:
    """2차 비교 자료: 익명 후보별 전체 집계와 초기 표본 기사별 판정 (제목·요약 포함, 앵커·탐침 등 내부 표시는 넣지 않음)."""
    lines = ["# 후보 비교 자료 (E2 관련성, 익명)", "",
             "후보 이름은 가려져 있다. 전체 집계는 전체 기사(83,211건) 기준이고, 기사별 판정은 초기 검토 표본에 한정한다.",
             "INCLUDE = 관련, REVIEW = 불확실(가중 0.5), EXCLUDE = 무관.", "", "## 1. 전체 집계", "",
             "| 후보 | INCLUDE | REVIEW | EXCLUDE |", "|---|---|---|---|"]
    for name in sorted(inv, key=lambda n: inv[n]):
        v = pd.Series(cands[name])
        lines.append(f"| {inv[name]} | {int((v == 'INCLUDE').sum()):,} | {int((v == 'REVIEW').sum()):,} | {int((v == 'EXCLUDE').sum()):,} |")
    lines += ["", "## 2. 기사별 판정", "", "| article_id | 제목 | 요약 | " + " | ".join(sorted(inv.values())) + " |",
              "|---|---|---|" + "---|" * len(inv)]
    order = sorted(inv, key=lambda n: inv[n])
    for _, r in rows.iterrows():
        cells = [r["gid"], clean_cell(r["title"]).replace("|", "/"), clean_cell(r["summary"]).replace("|", "/")[:220]]
        cells += [str(r[f"cand::{n}"]) for n in order]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 응답 병합·비교

def parse_response(text: str) -> tuple[pd.DataFrame, list[str]]:
    """응답에서 TSV 부분만 읽는다 (코드 블록·설명문 허용). 헤더 행(article_id … review_class) 이후 탭 구분 행."""
    errors, rows, header = [], [], None
    for ln in text.splitlines():
        s = ln.strip().strip("`")
        if not s:
            continue
        cells = [c.strip() for c in ln.rstrip("\n").split("\t")]
        if header is None:
            if "article_id" in cells and "review_class" in cells:
                header = cells
            continue
        if len(cells) == 1:
            continue
        if len(cells) != len(header):
            errors.append(f"열 수 불일치({len(cells)}≠{len(header)}): {ln[:80]}")
            continue
        rows.append(dict(zip(header, cells)))
    if header is None:
        errors.append("헤더 행을 찾지 못함")
        return pd.DataFrame(columns=OUT_COLS), errors
    missing = [c for c in OUT_COLS if c not in header]
    if missing:
        errors.append(f"필수 열 없음: {missing}")
    df = pd.DataFrame(rows, columns=header)
    for c in OUT_COLS:
        if c not in df:
            df[c] = ""
    return df[OUT_COLS], errors


def evidence_found(evidence: str, source: str) -> bool | None:
    """근거 문구가 입력 제목·요약에 있는지. 여러 문구는 / ; | 로 구분. 빈 값은 None."""
    ev = (evidence or "").strip()
    if not ev or ev in ("-", "없음", "N/A"):
        return None
    src = norm_text(source)
    parts = [norm_text(x).strip("'\" ") for x in re.split(r"\s*[/;|]\s*", ev) if x.strip()]
    return all(p in src for p in parts if p)


def validate(df: pd.DataFrame, expected_ids: set, source: dict) -> pd.DataFrame:
    df = df.copy()
    for c, allowed in ALLOWED.items():
        df[c] = df[c].astype(str).str.strip().str.upper()
        df[f"invalid_{c}"] = ~df[c].isin(allowed)
    df["unknown_id"] = ~df["article_id"].isin(expected_ids)
    df["duplicate_id"] = df["article_id"].duplicated(keep=False)
    df["content_evidence_found"] = [evidence_found(e, source.get(i, "")) for e, i in zip(df["content_evidence"], df["article_id"])]
    df["exclusion_evidence_found"] = [evidence_found(e, source.get(i, "")) for e, i in zip(df["exclusion_evidence"], df["article_id"])]
    df["evidence_problem"] = (df["content_evidence_found"] == False) | (df["exclusion_evidence_found"] == False) | \
        ((df["review_class"] == "CONTENT") & df["content_evidence_found"].isna())  # noqa: E712
    return df


def merge_responses(review_dir: Path, spot_check_rate: float = 0.1, seed: int = 0) -> dict:
    """model_responses/{batch}__{model}.tsv|txt 를 읽어 검사·병합하고 비교 보고서를 쓴다."""
    review_dir = Path(review_dir)
    man = pd.read_csv(review_dir / "internal" / "sample_manifest.csv", dtype=str)
    inputs = {p.stem.replace("review_input_", ""): pd.read_csv(p, sep="\t", dtype=str)
              for p in sorted((review_dir / "upload").rglob("review_input_batch_*.tsv"))}
    source = {r.article_id: f"{r.title} {r.summary}" for b in inputs.values() for r in b.itertuples()}
    long, issues = [], []
    for f in sorted((review_dir / "model_responses").glob("*.*")):
        if f.suffix not in (".tsv", ".txt") or "__" not in f.stem:
            continue
        batch, model = f.stem.split("__", 1)
        if batch not in inputs:
            issues.append({"file": f.name, "issue": f"알 수 없는 묶음 {batch}"})
            continue
        df, errs = parse_response(f.read_text(encoding="utf-8"))
        for e in errs:
            issues.append({"file": f.name, "issue": e})
        exp = set(inputs[batch]["article_id"])
        v = validate(df, exp, source)
        for i in sorted(exp - set(v["article_id"])):
            issues.append({"file": f.name, "issue": f"누락 ID {i}"})
        for i in sorted(set(v.loc[v["unknown_id"], "article_id"])):
            issues.append({"file": f.name, "issue": f"알 수 없는 ID {i}"})
        for i in sorted(set(v.loc[v["duplicate_id"], "article_id"])):
            issues.append({"file": f.name, "issue": f"중복 ID {i}"})
        v = v[~v["unknown_id"]].drop_duplicates("article_id")
        v.insert(0, "model", model)
        v.insert(1, "batch", batch)
        long.append(v)
    if not long:
        raise FileNotFoundError(f"{review_dir / 'model_responses'}에 응답 파일이 없다 ({{batch}}__{{model}}.tsv)")
    L = pd.concat(long, ignore_index=True)
    models = sorted(L["model"].unique())
    wide = L.pivot_table(index="article_id", columns="model", values="review_class", aggfunc="first")
    wide.columns = [f"class::{m}" for m in wide.columns]
    ev_bad = L.groupby("article_id")["evidence_problem"].any().rename("any_evidence_problem")
    info = man.set_index("gid")[["sample_set", "reason", "split", "batch", "has_sid", "e21_decision", "weight"] +
                                [c for c in man.columns if c.startswith(("cand::", "flag_"))]]
    W = wide.join(ev_bad).join(info, how="left")
    cls_cols = [c for c in W.columns if c.startswith("class::")]
    W["n_models"] = W[cls_cols].notna().sum(axis=1)
    W["agreement"] = W[cls_cols].nunique(axis=1, dropna=True) == 1
    W["consensus_class"] = np.where(W["agreement"] & (W["n_models"] >= 2), W[cls_cols].bfill(axis=1).iloc[:, 0], "")
    rng = np.random.default_rng(seed)
    agreed = W.index[W["agreement"] & (W["n_models"] >= 2)]
    spot = set(rng.choice(agreed, int(np.ceil(len(agreed) * spot_check_rate)), replace=False)) if len(agreed) else set()
    mixed_flags = W.get("flag_mixed_content_market", pd.Series(False, index=W.index)).astype(str).str.lower() == "true"
    policy_flags = W.get("flag_content_policy", pd.Series(False, index=W.index)).astype(str).str.lower() == "true"
    W["priority"] = np.select(
        [~W["agreement"], W["agreement"] & W["any_evidence_problem"].fillna(False), policy_flags | mixed_flags, W.index.isin(spot)],
        ["1_불일치", "2_합의·근거 없음", "3_정책·혼합 기사", "4_합의 무작위 확인"], "")
    W["researcher_review"] = ""
    W["researcher_note"] = ""
    out = review_dir / "merged"
    out.mkdir(exist_ok=True)
    L.to_csv(out / "responses_long.csv", index=False, encoding="utf-8-sig")
    W.reset_index().to_csv(out / "review_wide.csv", index=False, encoding="utf-8-sig")
    W[W["priority"] != ""].sort_values("priority").reset_index().to_csv(out / "priority_review.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(issues, columns=["file", "issue"]).to_csv(out / "parse_issues.csv", index=False, encoding="utf-8-sig")
    report = comparison_report(L, W, models, issues)
    (review_dir / "e2_llm_comparison_report.md").write_text(report, encoding="utf-8")
    return {"models": models, "articles": int(len(W)), "issues": len(issues), "agreement_rate": round(float(W["agreement"].mean()), 4),
            "priority_counts": W["priority"].value_counts().to_dict(), "report": str(review_dir / "e2_llm_comparison_report.md")}


def md_table(ct: pd.DataFrame) -> str:
    head = "| " + " | ".join([str(ct.index.name or "")] + [str(c) for c in ct.columns]) + " |"
    sep = "|" + "---|" * (len(ct.columns) + 1)
    return "\n".join([head, sep] + ["| " + " | ".join([str(i)] + [str(int(v)) for v in r]) + " |" for i, r in ct.iterrows()])


def kappa(a: pd.Series, b: pd.Series) -> float | None:
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return None
    a, b = a[m], b[m]
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return round(float((po - pe) / (1 - pe)), 4) if pe < 1 else None


def comparison_report(L: pd.DataFrame, W: pd.DataFrame, models: list[str], issues: list[dict]) -> str:
    cls_cols = [f"class::{m}" for m in models]
    out = ["# E2 LLM 비교 검토 보고서 (자동 병합 결과)", "",
           "> LLM 응답은 보조 검토 의견이다. 아래 일치율은 **검토 의견의 일관성**이며 정확도가 아니다. "
           "LLM 판정과 비교한 수치는 모두 'LLM 참고 판정 기준'이다. 연구자 확인 결과는 researcher_review 열에 따로 기록한다.", "",
           "## 1. 응답 형식 점검", "", f"- 모델: {', '.join(models)}", f"- 병합 기사 수: {len(W)}",
           f"- 형식·ID 문제: {len(issues)}건 (merged/parse_issues.csv)"]
    for m in models:
        s = L[L["model"] == m]
        inv = int(s[[c for c in s.columns if c.startswith("invalid_")]].any(axis=1).sum())
        cf = s["content_evidence_found"].dropna()
        out.append(f"- {m}: {len(s)}행, 허용값 위반 {inv}행, 콘텐츠 근거 문구 확인률 "
                   f"{(cf.astype(bool).mean() if len(cf) else float('nan')):.2f} ({len(cf)}건 중), 근거 문제 {int(s['evidence_problem'].sum())}행")
    out += ["", "## 2. 모델별 보조 판정 분포", "", "| 모델 | CONTENT | MARKET | OTHER | UNRESOLVED |", "|---|---|---|---|---|"]
    for m, c in zip(models, cls_cols):
        v = W[c].value_counts()
        out.append(f"| {m} | " + " | ".join(str(int(v.get(k, 0))) for k in ("CONTENT", "MARKET", "OTHER", "UNRESOLVED")) + " |")
    out += ["", "## 3. 모델 간 일치 (검토 의견의 일관성)", "", f"- 전체 일치율: {W['agreement'].mean():.3f}"]
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = W[cls_cols[i]], W[cls_cols[j]]
            m_ = a.notna() & b.notna()
            out.append(f"- {models[i]} vs {models[j]}: 일치 {(a[m_] == b[m_]).mean():.3f}, kappa {kappa(a, b)}")
    for col, title in (("has_sid", "섹션 코드 유무별"), ("sample_set", "표본 구분별"), ("reason", "추출 사유별")):
        if col in W:
            g = W.groupby(col)["agreement"].agg(["size", "mean"]).round(3)
            out += ["", f"### 불일치 — {title}", "", "| 값 | 기사 | 일치율 |", "|---|---|---|"]
            out += [f"| {k} | {int(r['size'])} | {r['mean']:.3f} |" for k, r in g.iterrows()]
    out += ["", "## 4. 기존 판정과의 교차 (LLM 참고 판정 기준)", ""]
    cand_cols = [c for c in W.columns if c.startswith("cand::")]
    cons = W[W["consensus_class"] != ""]
    for c in cand_cols:
        ct = pd.crosstab(cons[c], cons["consensus_class"])
        out += [f"### {c.replace('cand::', '')} × LLM 합의 판정 (합의 {len(cons)}건)", "", md_table(ct) if len(ct) else "(합의 기사 없음)", ""]
    out += ["> 대표표본과 경계·진단표본을 합친 결과이므로 전체 기사 비율로 해석하지 않는다. 대표표본 비율은 weight(1/포함확률)로 가중해야 한다.", "",
            "## 5. 우선 검토 목록", "", "merged/priority_review.csv — 1 불일치 / 2 합의했지만 근거 문구 없음 / 3 정책·혼합 기사 / 4 합의 무작위 확인.",
            "", "| 우선순위 | 기사 |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in W["priority"].value_counts().sort_index().items() if k]
    out += ["", "## 6. 연구자 결정", "", "(researcher_review 열과 decision_log에 기록. 이 보고서는 운영 변경을 지시하지 않는다.)", ""]
    return "\n".join(out)
