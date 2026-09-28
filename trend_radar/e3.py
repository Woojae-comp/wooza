"""E3 핵심어 사전 실험 (v2 설계 3장, 기획안 6장).

입력: E2의 기사별 관련성 확률·판정 (m_rel_ws_lr). REVIEW 처리는 지표 성격에 따라 다르게 한다.

| 처리 항목                          | INCLUDE | REVIEW | EXCLUDE |
|------------------------------------|---------|--------|---------|
| 후보어 추출·문서빈도               | 1.0     | 0.5    | 0       |
| C-value·지속성·분야 엔트로피·연관도 | 1.0     | 0.5    | 0       |
| 기본 로그오즈 (A Strict)           | 관련    | 제외   | 무관    |
| B Half weight                      | 관련 1.0 + REVIEW 0.5 대 EXCLUDE        |
| C Soft label                       | 관련 말뭉치 p_rel, 무관 말뭉치 1-p_rel  |

로그오즈는 사전확률 가중(Monroe 외 2008) z점수. 유의성 1.96을 기준으로 쓴다 (사람이 고른 값이 아님).
등급(CORE / EXTENDED / EMERGING / REVIEW / DROP)은 세 방식의 일치·순위 안정성·성장률로 자동 부여하고,
class_final은 비워 둔다 (사람 확정 또는 이후 규칙 변경은 decision_log).
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from .entities import protected_entities
from .text import HANGUL, JOINERS, _space_free, merged_spans

NOUN_TAGS = {"NNG", "NNP", "SL"}
Z_SIG = 1.96


# ---------------------------------------------------------------- E3 입력

def latest_e2(registry: Path, out_root: Path) -> tuple[str, Path, dict]:
    recs = [json.loads(l) for l in (registry / "experiment_registry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rec = [r for r in recs if r.get("task") == "article_relevance"][-1]
    run_dir = out_root / "runs" / rec["run_id"]
    summary = json.loads((run_dir / "e2_summary.json").read_text(encoding="utf-8"))
    return rec["run_id"], run_dir / "02_article_relevance.parquet", summary


def e3_input(rel: pd.DataFrame, e2_run_id: str, threshold: float) -> pd.DataFrame:
    ws = rel["p_ws_lr"] >= threshold
    seed = rel["seed_layer"].astype(bool)
    lm = rel["p_labelmodel"] >= 0.5
    agree = ws.astype(int) + seed.astype(int) + lm.astype(int)
    reason = np.where(rel["decision"] == "REVIEW",
                      "ws:" + np.where(ws, "+", "-") + " seed:" + np.where(seed, "+", "-") + " lm:" + np.where(lm, "+", "-"), "")
    half = rel["decision"].map({"INCLUDE": 1.0, "REVIEW": 0.5, "EXCLUDE": 0.0})
    return pd.DataFrame({
        "article_id": rel["gid"], "relevance_decision": rel["decision"], "relevance_probability": rel["p_ws_lr"],
        "content_weight_half": half, "content_weight_soft": rel["p_ws_lr"], "general_weight_soft": 1 - rel["p_ws_lr"],
        "model_agreement_count": agree, "review_reason": reason, "e2_run_id": e2_run_id,
    })


# ---------------------------------------------------------------- 후보어 (1~3그램 명사구)

def candidates(tokens, texts, lexicon, attach: set[str], max_n: int = 3,
               restore: dict[str, str] | None = None) -> list[list[str]]:
    """붙어 있거나 공백 하나로 이어진 명사 형태소 연쇄에서 1~3그램. 동의어는 형태소 단위로 통일한다.
    불용어는 여기서 빼지 않는다 (사전 등급에서 판단, 승인된 불용어는 표시만).
    restore: 보호 토큰 → 원래 표기 ('P의거짓' → 'P의 거짓')."""
    restore = restore or {}
    out = []
    for toks, text in zip(tokens, texts):
        spans = merged_spans(toks, text, lexicon, attach)
        runs, cur = [], []
        prev_end = None
        for f, tag, s, e in spans:
            if tag not in NOUN_TAGS:
                if cur:
                    runs.append(cur)
                cur, prev_end = [], None
                continue
            gap = text[prev_end:s] if prev_end is not None else None
            if cur and not (gap in JOINERS or gap == " "):
                runs.append(cur)
                cur = []
            f = restore.get(f, f)
            cur.append(lexicon.synonym_map.get(f, f))
            prev_end = e
        if cur:
            runs.append(cur)
        terms = set()
        for r in runs:
            for n in range(1, max_n + 1):
                for i in range(len(r) - n + 1):
                    g = r[i:i + n]
                    t = " ".join(g)
                    if n == 1 and (len(t) < 2 if HANGUL.search(t) else len(t) < 2 or t.isdigit()):
                        continue
                    if n > 1 and len(t.replace(" ", "")) < 3:
                        continue
                    terms.add(t)
        out.append(sorted(terms))
    return out


def doc_term(doc_terms: list[list[str]], min_df: int) -> tuple[sparse.csr_matrix, list[str]]:
    df = collections.Counter(t for ts in doc_terms for t in ts)
    vocab = sorted((t for t, c in df.items() if c >= min_df), key=lambda t: (-df[t], t))
    idx = {t: i for i, t in enumerate(vocab)}
    rows, cols = [], []
    for r, ts in enumerate(doc_terms):
        for t in ts:
            j = idx.get(t)
            if j is not None:
                rows.append(r)
                cols.append(j)
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(doc_terms), len(vocab)))
    return X, vocab


# ---------------------------------------------------------------- 지표

def weighted_log_odds(X, w_rel: np.ndarray, w_irr: np.ndarray, alpha0: float = 1000.0) -> np.ndarray:
    """사전확률 가중 로그오즈 z (Monroe·Colaresi·Quinn 2008). 사전분포 = 두 말뭉치 합."""
    y_i = np.asarray(X.T @ w_rel).ravel()
    y_j = np.asarray(X.T @ w_irr).ravel()
    n_i, n_j = y_i.sum(), y_j.sum()
    pool = y_i + y_j
    a = alpha0 * pool / max(pool.sum(), 1e-9) + 1e-3
    a0 = a.sum()
    d = np.log((y_i + a) / (n_i + a0 - y_i - a)) - np.log((y_j + a) / (n_j + a0 - y_j - a))
    return d / np.sqrt(1 / (y_i + a) + 1 / (y_j + a))


def c_value(vocab: list[str], f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C-value (Frantzi 외 2000). 1그램도 비교할 수 있게 log2(길이+1)을 쓴다.
    두 번째 값은 '가장 큰 상위 복합어 빈도 / 자기 빈도' (복합어 조각 판정용)."""
    idx = {t: i for i, t in enumerate(vocab)}
    parents = collections.defaultdict(list)
    for t in vocab:
        g = t.split(" ")
        for n in range(1, len(g)):
            for i in range(len(g) - n + 1):
                sub = " ".join(g[i:i + n])
                if sub in idx:
                    parents[idx[sub]].append(idx[t])
    out = np.zeros(len(vocab))
    parent_share = np.zeros(len(vocab))
    for j, t in enumerate(vocab):
        n = t.count(" ") + 1
        nest = parents.get(j)
        base = f[j] - (f[nest].sum() / len(nest) if nest else 0)
        out[j] = np.log2(n + 1) * max(base, 0)
        if nest and f[j] > 0:
            parent_share[j] = f[nest].max() / f[j]
    return out, parent_share


def phrase_quality_fail(term: str, q: dict) -> bool:
    """개념이 아닌 구문: 시간·문서 형식어만으로 된 구문('이날 오전', '공식 홈페이지', '모닝 리포트'),
    직함형('김 부장': 한 글자 성 + 직함)."""
    parts = term.split(" ")
    form = set(q.get("time_words", [])) | set(q.get("form_words", []))
    if all(p in form for p in parts):
        return True
    if len(parts) >= 2 and parts[-1] in form and parts[-1] in set(q.get("form_words", [])):
        return True
    titles = set(q.get("title_words", []))
    if parts[-1] in titles and (len(parts) == 1 or (len(parts[0]) == 1 and HANGUL.search(parts[0]))):
        return True
    return False


def grade(zA, zB, zC, shift_AB, shift_AC, persistence, growth, recent_df, parent_share,
          is_stop, market_phrase, quality_fail, rules: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """등급 부여. 기준값은 통계(유의성 1.96, 순위 변동 분위수)와 설정(rules)에서 온다."""
    z = rules.get("z", Z_SIG)
    sigA, sigB, sigC = zA >= z, zB >= z, zC >= z
    sig_any = sigA | sigB | sigC
    worst = np.maximum(shift_AB, shift_AC)
    med_shift = float(np.median(worst[sig_any])) if sig_any.any() else 0.0
    q75_shift = float(np.quantile(shift_AB[sig_any], 0.75)) if sig_any.any() else 0.0
    fragment = parent_share >= rules.get("fragment_share", 0.5)
    hard_drop = is_stop | quality_fail
    drop = (~sig_any & (np.maximum.reduce([zA, zB, zC]) <= 0)) | hard_drop
    capped = fragment | market_phrase        # 최대 REVIEW
    core = sigA & sigC & (worst <= med_shift) & (persistence >= rules.get("core_persistence", 0.5)) & ~capped
    emerging = (growth >= rules.get("emerging_growth", 2.0)) & (recent_df >= rules.get("emerging_recent_df", 10)) \
        & (zC > 0) & ~capped
    review = sig_any & ((sigA != sigB) | (shift_AB > q75_shift) | capped)
    cls = np.select([drop, core, emerging, review, sig_any], ["DROP", "CORE", "EMERGING", "REVIEW", "EXTENDED"], "DROP")
    reason = np.select(
        [is_stop, quality_fail, fragment & ~drop, market_phrase & ~drop, core, emerging & ~drop, review & ~drop, sig_any],
        ["승인 불용어", "구문 품질 규칙(시간·문서 형식어·직함형)", "상위 복합어의 일부로 주로 쓰임",
         "시장 구문 포함", "A·C 유의, 순위 안정, 지속성 기준 충족", "최근 6개월 비중 급증, 특이도 양(+)",
         "REVIEW 포함 여부에 따라 유의성·순위 변동", "한 가지 이상 방식에서 유의"], "세 방식 모두 특이도 낮음")
    return cls, reason, {"rank_shift_median": med_shift, "rank_shift_AB_q75": q75_shift}


def pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def entropy_rows(M: np.ndarray) -> np.ndarray:
    s = M.sum(1, keepdims=True)
    P = np.divide(M, s, out=np.zeros_like(M), where=s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0), axis=1)
    return H / np.log(M.shape[1])


def compare(zs: dict[str, np.ndarray], vocab: list[str], ks=(50, 100)) -> tuple[pd.DataFrame, dict]:
    names = list(zs)
    rank = {n: pd.Series(-zs[n]).rank(method="first").to_numpy() for n in names}
    top = {n: {k: set(np.argsort(-zs[n])[:k]) for k in ks} for n in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = {"pair": f"{a} vs {b}"}
            for k in ks:
                sa, sb = top[a][k], top[b][k]
                r[f"jaccard_top{k}"] = len(sa & sb) / len(sa | sb)
            u = sorted(top[a][ks[-1]] | top[b][ks[-1]])
            r[f"spearman_union_top{ks[-1]}"] = spearmanr(rank[a][u], rank[b][u]).statistic
            r["spearman_all"] = spearmanr(rank[a], rank[b]).statistic
            rows.append(r)
    uniq = {n: [vocab[j] for j in sorted(top[n][ks[-1]] - set().union(*(top[m][ks[-1]] for m in names if m != n)),
                                          key=lambda j: rank[n][j])] for n in names}
    return pd.DataFrame(rows), uniq


# ---------------------------------------------------------------- 실행

def run_e3(cfg: dict, raw: pd.DataFrame, out_root: Path, min_df: int = 5) -> dict:
    from .config import Lexicon
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .text import prepped_texts, space_joined_names, tokenize_corpus

    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e3_keyword_dictionary", cfg, out_root, snap["data_snapshot_id"])
    from .selection import selected_e2
    e2_run, e2_path, e2_sum = selected_e2(cfg, REGISTRY, out_root)   # 선정(또는 명시 후보) E2만
    lex = Lexicon.load()

    with run.stage("e3_input") as st:
        rel = pd.read_parquet(e2_path)
        inp = e3_input(rel, e2_run, e2_sum["otsu_threshold"])
        p = run.dir / "03_e3_input_articles.parquet"
        inp.to_parquet(p, index=False)
        run.artifact(p, "e3_input", rows=len(inp))
        st["rows_out"] = len(inp)

    with run.stage("candidates") as st:
        corpus = build_corpus(raw, cfg)
        arts = corpus.articles.merge(inp, left_on="gid", right_on="article_id", how="left")
        assert arts["relevance_decision"].notna().all(), "E2 결과와 기사 집합이 다르다 (스냅샷 확인)"
        names = space_joined_names(sorted(corpus.article_company["company"].unique()),
                                   cfg["cleaning"].get("company_aliases") or {})
        e3c = cfg.get("e3", {})
        market_words = set(cfg["layers"]["market_seed"]) | lex.stopwords | set(e3c.get("market_phrase_words", []))
        # 보호 개체: 작품명·정책명을 한 토큰으로 묶어 형태소 분석 → 원래 표기로 복원
        ents = protected_entities(corpus.articles, market_words)
        pe = run.dir / "05_protected_entities.csv"
        ents.to_csv(pe, index=False, encoding="utf-8-sig")
        run.artifact(pe, "protected_entities", rows=len(ents))
        prot = ents[ents["protect"]]["entity"]
        restore = {_space_free(e): e for e in prot}
        names_tok = names + [e for e in prot if e not in set(names)]
        texts = prepped_texts(corpus.articles, names_tok)
        tokens = tokenize_corpus(texts, names_tok, out_root / "cache")
        dts = candidates(tokens, texts, lex, set(cfg["keywords"].get("attach_suffixes", [])), restore=restore)
        X, vocab = doc_term(dts, min_df)
        st["rows_out"] = len(vocab)

    with run.stage("metrics") as st:
        dec = arts["relevance_decision"].to_numpy()
        inc, rev, exc = (dec == "INCLUDE").astype(float), (dec == "REVIEW").astype(float), (dec == "EXCLUDE").astype(float)
        half = arts["content_weight_half"].to_numpy()
        prel = arts["content_weight_soft"].to_numpy()
        zA = weighted_log_odds(X, inc, exc)
        zB = weighted_log_odds(X, half, exc)
        zC = weighted_log_odds(X, prel, 1 - prel)
        df_half = np.asarray(X.T @ half).ravel()
        df_all = np.asarray(X.sum(0)).ravel()
        cval, parent_share = c_value(vocab, df_half)
        from .text import token_form
        co_names = {token_form(n) for n in names} | set(corpus.article_company["company"].unique())
        ent_names = set(ents["entity"])
        entity_type = np.array(["기업" if t.replace(" ", "") in co_names else "작품·개체" if t in ent_names else "개념"
                                for t in vocab])
        # 월별 (가중) 문서빈도 → 지속성·성장
        month = arts["date"].dt.to_period("M").astype(str).to_numpy()
        months = sorted(set(month))
        Mi = {m: i for i, m in enumerate(months)}
        P = sparse.csr_matrix((half, (np.arange(len(month)), [Mi[m] for m in month])), shape=(len(month), len(months)))
        mdf = np.asarray((X.T @ P).todense())
        mtot = np.asarray(P.sum(0)).ravel()
        last12 = slice(len(months) - 12, len(months))
        persistence = (mdf[:, last12] >= 1).mean(1)
        rec6, prv12 = slice(len(months) - 6, len(months)), slice(len(months) - 18, len(months) - 6)
        share_r = (mdf[:, rec6].sum(1) + 1) / mtot[rec6].sum()
        share_p = (mdf[:, prv12].sum(1) + 1) / mtot[prv12].sum()
        growth = share_r / share_p
        first_idx = np.argmax(mdf > 0, axis=1)
        first_month = np.array(months)[first_idx]
        # 분야 엔트로피 (핵심 6개 분야, 가중)
        related = cfg["cleaning"].get("related_label", "연관산업")
        gid_row = pd.Series(np.arange(len(arts)), index=arts["gid"])
        sec = corpus.article_sector[corpus.article_sector["sector"] != related]
        sectors = sorted(sec["sector"].unique())
        S = sparse.csr_matrix((np.ones(len(sec)), (sec["gid"].map(gid_row).to_numpy(), sec["sector"].map({s: i for i, s in enumerate(sectors)}).to_numpy())),
                              shape=(len(arts), len(sectors)))
        sdf = np.asarray((X.T @ sparse.diags(half) @ S).todense())
        sent = entropy_rows(sdf)
        top_sector = np.array(sectors)[sdf.argmax(1)]
        # 콘텐츠 시드와의 연관 (관련 말뭉치 NPMI, 시드 상위 3개 평균)
        seeds = [s for s in cfg["layers"]["content_seed"] if s in set(vocab)]
        sidx = [vocab.index(s) for s in seeds]
        Xr = X.multiply(half[:, None]).tocsc()
        N = half.sum()
        co = np.asarray((Xr.T @ X[:, sidx]).todense())
        pa, pb = df_half / N, df_half[sidx] / N
        pab = co / N
        with np.errstate(divide="ignore", invalid="ignore"):
            npmi = np.where(pab > 0, np.log(pab / np.outer(pa, pb)) / -np.log(pab), -1)
        for k, j in enumerate(sidx):
            npmi[j, k] = -1  # 자기 자신 제외
        net = np.sort(npmi, axis=1)[:, -3:].mean(1)
        # 사전 점수 (백분위 정규화, 기획안 6.2 초기 가중치)
        # 운영 순위는 B(절반 가중) 로그오즈. A·C는 민감도 비교용
        dict_score = 0.35 * pct(zB) + 0.20 * pct(np.log1p(df_half)) + 0.15 * pct(cval) + 0.15 * pct(net) + 0.15 * persistence
        st["rows_out"] = len(vocab)

    with run.stage("compare") as st:
        zs = {"A_strict": zA, "B_half": zB, "C_soft": zC}
        cmp, uniq = compare(zs, vocab)
        rA = pd.Series(-zA).rank(method="first").to_numpy()
        rB = pd.Series(-zB).rank(method="first").to_numpy()
        rC = pd.Series(-zC).rank(method="first").to_numpy()
        shift_AB = np.abs(np.log(rA) - np.log(rB))   # REVIEW 포함 여부에 따른 순위 변동 (로그 순위 차)
        shift_AC = np.abs(np.log(rA) - np.log(rC))
        # 분야별 핵심어 변화: 분야 기사(방식별 가중) 대 EXCLUDE
        sector_rows = []
        for i, s in enumerate(sectors):
            ins = np.asarray(S[:, i].todense()).ravel() > 0
            tops = {}
            for name, (wr, wi) in {"A_strict": (inc * ins, exc), "B_half": (half * ins, exc),
                                   "C_soft": (prel * ins, 1 - prel)}.items():
                z = weighted_log_odds(X, wr, wi)
                tops[name] = [vocab[j] for j in np.argsort(-z)[:20]]
            sector_rows.append({"sector": s, **{f"top20_{k}": " | ".join(v) for k, v in tops.items()},
                                "jaccard_A_C": len(set(tops["A_strict"]) & set(tops["C_soft"])) / len(set(tops["A_strict"]) | set(tops["C_soft"]))})
        sector_cmp = pd.DataFrame(sector_rows)
        approved = {" ".join(c) for c in lex.compounds} | {"".join(c) for c in lex.compounds}
        new_compounds = {n: int(sum(1 for j in np.argsort(-zs[n])[:200] if " " in vocab[j] and vocab[j] not in approved
                                    and vocab[j].replace(" ", "") not in approved)) for n in zs}
        st["rows_out"] = len(cmp)

    with run.stage("grade") as st:
        recent_df = mdf[:, rec6].sum(1)
        is_stop = np.array([" " not in t and t in lex.stopwords for t in vocab])  # 승인된 불용어(1그램)
        # 시장 구문: 시장 씨앗어·승인 불용어·E3.1 시장 구문 단어를 포함 (하이브 주가, 장 초반 강세, 특징주)
        market_phrase = np.array([(any(p in market_words for p in t.split(" ")) or t.replace(" ", "") in market_words)
                                  and not is_stop[j] and entity_type[j] == "개념" for j, t in enumerate(vocab)])
        quality_fail = np.array([entity_type[j] == "개념" and phrase_quality_fail(t, e3c.get("phrase_quality", {}))
                                 for j, t in enumerate(vocab)])
        cls, reason, gth = grade(zA, zB, zC, shift_AB, shift_AC, persistence, growth, recent_df, parent_share,
                                 is_stop, market_phrase, quality_fail, e3c.get("grade", {}))
        med_shift, q75_shift = gth["rank_shift_median"], gth["rank_shift_AB_q75"]
        st["rows_out"] = int((cls != "DROP").sum())

    with run.stage("write") as st:
        titles = arts["title"].to_numpy()
        Xc = X.tocsc()
        order = np.argsort(-prel)
        rank_of = np.empty(len(order), dtype=int)
        rank_of[order] = np.arange(len(order))

        def evidence(j, k=3):
            rows = Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]
            rows = rows[np.argsort(rank_of[rows])][:k]
            return [arts["gid"].iat[r] for r in rows], [titles[r] for r in rows]

        keep = np.flatnonzero((cls != "DROP") | (df_all >= 50))
        domain = pct(zB)
        recs = []
        for j in keep:
            ev_ids, ev_titles = evidence(j)
            recs.append({
                "keyword_id": f"kw_{j:06d}", "keyword": vocab[j], "ngram": vocab[j].count(" ") + 1,
                "entity_type": entity_type[j], "parent_compound_share": round(float(parent_share[j]), 3),
                "doc_freq": int(df_all[j]), "doc_freq_half": round(float(df_half[j]), 1),
                "z_A_strict": round(float(zA[j]), 3), "z_B_half": round(float(zB[j]), 3), "z_C_soft": round(float(zC[j]), 3),
                "rank_A": int(rA[j]), "rank_B": int(rB[j]), "rank_C": int(rC[j]),
                "domain_score": round(float(domain[j]), 4), "termhood_cvalue": round(float(cval[j]), 2),
                "network_score": round(float(net[j]), 4), "persistence": round(float(persistence[j]), 3),
                "sector_entropy": round(float(sent[j]), 3), "top_sector": top_sector[j],
                "growth_6m_vs_prev12m": round(float(growth[j]), 3), "first_month": first_month[j],
                "dictionary_score": round(float(dict_score[j]), 4), "class_auto": cls[j], "class_final": "",
                "review_reason": reason[j], "evidence_refs": " ".join(ev_ids), "evidence_titles": " || ".join(ev_titles),
                "run_id": run.run_id,
            })
        dic = pd.DataFrame(recs).sort_values(["class_auto", "dictionary_score"], ascending=[True, False])
        # E5가 다시 계산하지 않도록 기사×후보 행렬과 순서를 저장
        sparse.save_npz(run.dir / "03_doc_term.npz", X.tocsr())
        (run.dir / "03_doc_term_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        arts[["gid"]].to_csv(run.dir / "03_doc_term_rows.csv", index=False)
        run.artifact(run.dir / "03_doc_term.npz", "doc_term", rows=X.shape[0])
        p3 = run.dir / "03_keyword_dictionary.csv"
        dic.to_csv(p3, index=False, encoding="utf-8-sig")
        run.artifact(p3, "keyword_dictionary", rows=len(dic))
        p4 = run.dir / "04_keyword_candidates_review.csv"
        dic[dic["class_auto"].isin(["REVIEW", "EMERGING"])].to_csv(p4, index=False, encoding="utf-8-sig")
        run.artifact(p4, "keyword_candidates_review")
        rep = run.dir / "e3_keyword_report.xlsx"
        grades = pd.Series(cls).value_counts().rename_axis("등급").reset_index(name="후보 수")
        with pd.ExcelWriter(rep) as xw:
            cmp.round(4).to_excel(xw, sheet_name="A·B·C 비교", index=False)
            pd.DataFrame({k: pd.Series(v[:100]) for k, v in uniq.items()}).to_excel(xw, sheet_name="방식별 고유 후보(상위100)", index=False)
            pd.DataFrame({k: [vocab[j] for j in np.argsort(-z)[:100]] for k, z in zs.items()}).to_excel(xw, sheet_name="방식별 상위100", index=False)
            sector_cmp.to_excel(xw, sheet_name="분야별 핵심어", index=False)
            pd.Series(new_compounds, name="상위200 중 신규 복합어").rename_axis("방식").reset_index().to_excel(xw, sheet_name="신규 복합어 회수", index=False)
            grades.to_excel(xw, sheet_name="등급 분포", index=False)
            for g in ("CORE", "EXTENDED", "EMERGING", "REVIEW"):
                dic[dic["class_auto"] == g].head(300).to_excel(xw, sheet_name=f"{g} 상위", index=False)
        run.artifact(rep, "e3_report")
        st["rows_out"] = len(dic)

    summary = {
        "e2_run_id": e2_run, "candidates": len(vocab), "min_df": min_df,
        "comparison": cmp.round(4).to_dict("records"),
        "unique_top100": {k: v[:15] for k, v in uniq.items()},
        "new_compounds_in_top200": new_compounds,
        "grade_counts": pd.Series(cls).value_counts().to_dict(),
        "operating_rank": "B_half (A·C는 민감도 비교용)",
        "protected_entities": int(len(ents)),
        "thresholds": {"z_significance": Z_SIG, "rank_shift_median": round(med_shift, 4), "rank_shift_AB_q75": round(q75_shift, 4),
                       **e3c.get("grade", {})},
        "top": {g: dic[dic["class_auto"] == g]["keyword"].head(40).tolist() for g in ("CORE", "EXTENDED", "EMERGING", "REVIEW")},
    }
    (run.dir / "e3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_keyword_v2_e3", "run_id": run.run_id, "task": "keyword_dictionary",
        "data_snapshot_id": snap["data_snapshot_id"], "input_e2_run_id": e2_run,
        "models": ["A_strict", "B_half", "C_soft"], "result": summary["comparison"],
        "new_compounds_in_top200": new_compounds, "grade_counts": summary["grade_counts"]})
    run.finish()
    return summary
