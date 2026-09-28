"""E4.1 문서 군집 보완 실험 (2026-09-28 사용자 결정).

역할 (동등한 후보가 아님)
- A 운영 기준선: Leiden 공동체 기반 기사 배정 (개체 확장 네트워크, 기본 백본 / 기본 + 희소 분야 보완 백본)
- B 도전자: LSA(TruncatedSVD 100·200·300) 후 K-means (K 80·120·160)
- C 보완 실험: 분야별 층화 LSA 군집 → 소규모 분야 회복 효과만 평가, 분야 군집끼리 다시 연결
- 예상 최종 구조: Leiden으로 큰 주제를 정하고, 기사량이 크거나 응집도가 낮은 공동체만 LSA로 세분화 (topic_id / subtopic_id)

노드 유형은 사람이 단어를 고르지 않고 다음 순서로 정한다.
1. entity_type(E3): 기업 → 콘텐츠 기업(핵심 6개 분야) / 비콘텐츠 기업(연관산업만), 작품·개체 → 작품·인물·정책명
2. 증권·시세 표현: 시장 씨앗어·시장 구문·승인 불용어 포함, 또는 시장 앵커 대 콘텐츠 앵커 로그오즈 z ≥ 1.96 이면서 B 특이도 z < 1.96
3. 고유명사(Kiwi NNP 포함) 개념어: B 특이도 z ≥ 1.96 → 콘텐츠 고유명사, 아니면 비콘텐츠 고유명사 (LG전자·SK텔레콤 등)
4. 저특이도 일반어: 나머지 개념어 중 B 특이도 하위 50% ∩ 문서빈도 상위 10%
5. 나머지 개념어 (고빈도 콘텐츠 개념 포함)

모델 선정은 평균 순위가 아니라 통과 기준(gate) 우선. 통과 모델끼리만 비교한다.
"""
from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .e4 import (cached_edges, cached_leiden, network_key, centroid_similarity, community_metrics, nodes_hash, npmi_coherence, npmi_edges, pct,
                 sentence_matrix, tfidf)

Z_SIG = 1.96
CONCEPT_KEEP = {"concept"}
EXT_KEEP = {"concept", "content_company", "work_person_policy", "proper_noun_content", "noncontent_company_specific"}
NOISE_TYPES = {"market_expression", "general_low_specificity"}


# ---------------------------------------------------------------- 노드 유형

def has_nnp(term: str, kiwi) -> bool:
    return any(t.tag == "NNP" for t in kiwi.tokenize(term))


def contains_word(term: str, words: set[str]) -> bool:
    """띄어쓴 단어 일치, 또는 붙여 쓴 복합어가 그 단어로 끝남 (교보증권·NH투자증권·목표주가)."""
    t = term.replace(" ", "")
    return bool(set(term.split(" ")) & words) or t in words or any(len(x) >= 2 and t.endswith(x) for x in words)


def log_odds_delta(X, w_rel: np.ndarray, w_irr: np.ndarray, alpha0: float = 1000.0) -> np.ndarray:
    """가중 로그오즈의 효과 크기 (z의 분자). z는 빈도가 클수록 커지므로 '흔하지만 특이하지 않은 말'은 효과 크기로 본다."""
    y_i = np.asarray(X.T @ w_rel).ravel()
    y_j = np.asarray(X.T @ w_irr).ravel()
    n_i, n_j = y_i.sum(), y_j.sum()
    pool = y_i + y_j
    a = alpha0 * pool / max(pool.sum(), 1e-9) + 1e-3
    a0 = a.sum()
    return np.log((y_i + a) / (n_i + a0 - y_i - a)) - np.log((y_j + a) / (n_j + a0 - y_j - a))


def node_types(dic: pd.DataFrame, z_market: np.ndarray, market_words: set[str], company_sectors: dict[str, set[str]],
               core_sectors: set[str], ref: pd.DataFrame, rules: dict, kiwi=None) -> pd.DataFrame:
    """dic: 행렬 어휘 순서의 사전 (keyword, entity_type, z_B_half, doc_freq, 선택: delta_B). ref: 분위수 기준 (네트워크 후보 개념어).
    순서: entity_type → 시장 단어 포함 → 고유명사(NNP) → 시장 앵커 z → 저특이도 일반어 → 개념어.
    - 고유명사: 콘텐츠 특이도 비유의(z_B < 1.96) 이면서 시장 앵커 쪽으로 유의(z_market ≥ 1.96)일 때만 비콘텐츠
    - 증권·시세: 시장 앵커 z ≥ 1.96 이면서 z_market ≥ z_B
    - 저특이도 일반어: 효과 크기(delta_B) 하위 50% ∩ 문서빈도 상위 10% (z는 빈도에 비례해 커지므로 효과 크기로 본다)"""
    from .text import token_form

    zB = dic["z_B_half"].fillna(0).to_numpy()
    eff = dic["delta_B"].fillna(0).to_numpy() if "delta_B" in dic else zB
    c_ref = ref[ref["entity_type"] == "개념"]
    er = np.sort((c_ref["delta_B"] if "delta_B" in c_ref else c_ref["z_B_half"]).fillna(0).to_numpy())
    dr = np.sort(c_ref["doc_freq"].fillna(0).to_numpy())
    ep = np.searchsorted(er, eff, side="right") / max(len(er), 1)
    dp = np.searchsorted(dr, dic["doc_freq"].fillna(0).to_numpy(), side="right") / max(len(dr), 1)
    out, why = [], []
    for k, (t, et) in enumerate(zip(dic["keyword"], dic["entity_type"])):
        if et == "기업":
            secs = company_sectors.get(token_form(t), set())
            if secs & core_sectors:
                out.append("content_company"); why.append("기업(핵심 분야)")
            elif zB[k] >= Z_SIG and z_market[k] < zB[k]:
                out.append("noncontent_company_specific"); why.append("기업(연관산업), 콘텐츠 특이도 유의")
            else:
                out.append("noncontent_company"); why.append("기업(연관산업)")
            continue
        if et == "작품·개체":
            out.append("work_person_policy"); why.append("작품·인물·정책명 (보호 개체)")
            continue
        if contains_word(t, market_words):
            out.append("market_expression"); why.append("시장 단어 포함")
            continue
        if kiwi is not None and has_nnp(t, kiwi):
            if zB[k] < Z_SIG and z_market[k] >= Z_SIG:
                out.append("proper_noun_noncontent"); why.append(f"고유명사, 콘텐츠 비유의·시장 앵커 z={z_market[k]:.1f}")
            else:
                out.append("proper_noun_content"); why.append("고유명사")
            continue
        if z_market[k] >= Z_SIG and z_market[k] >= zB[k]:
            out.append("market_expression"); why.append(f"시장 앵커 z={z_market[k]:.1f} ≥ 콘텐츠 z={zB[k]:.1f}")
            continue
        if ep[k] < rules.get("general_max_z_pct", 0.5) and dp[k] >= rules.get("general_min_df_pct", 0.9):
            out.append("general_low_specificity"); why.append(f"효과 크기 백분위 {ep[k]:.2f}, 문서빈도 백분위 {dp[k]:.2f}")
            continue
        out.append("concept"); why.append("")
    r = pd.DataFrame({"keyword": dic["keyword"], "entity_type": dic["entity_type"], "class_auto": dic["class_auto"],
                      "node_type": out, "reason": why, "z_B_half": zB.round(3), "delta_B": np.round(eff, 4),
                      "z_market": np.round(z_market, 3), "doc_freq": dic["doc_freq"]})
    r["concept_keep"] = r["node_type"].isin(CONCEPT_KEEP)
    r["extended_keep"] = r["node_type"].isin(EXT_KEEP)
    return r


# ---------------------------------------------------------------- A: Leiden 공동체 기반 기사 배정

def community_matrix(memb: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """노드 × 공동체 지시 행렬: 규모 보정(1/√크기), 0/1, 공동체 번호. 고립 노드(-1)는 어느 공동체에도 속하지 않는다."""
    ids = np.array(sorted(set(memb[memb >= 0].tolist())))
    col = {c: i for i, c in enumerate(ids)}
    r = np.where(memb >= 0)[0]
    size = np.bincount([col[m] for m in memb[r]], minlength=len(ids)).astype(float)
    c = np.array([col[m] for m in memb[r]])
    M = sparse.csr_matrix((1 / np.sqrt(size[c]), (r, c)), shape=(len(memb), len(ids)))
    Mb = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(memb), len(ids)))
    return M, Mb, ids


def assign_leiden(V: sparse.csr_matrix, memb: np.ndarray, nkw: np.ndarray, rules: dict) -> pd.DataFrame:
    """V: 기사 × 노드 TF-IDF (l2). community_score = 공동체 소속 핵심어 TF-IDF 합 ÷ √공동체 크기 (= 공동체 지시 벡터와의 코사인).
    keyword_coverage = 공동체에 속한 노드의 TF-IDF 질량 ÷ 기사 전체 노드 질량.
    SPLIT = 1·2위 공동체가 각각 기사 공동체 질량의 split_min_mass 이상 → 두 공동체에 점수 비례 분할.
    LOW = 핵심어 min_keywords 미만, 커버리지 min_coverage 미만, 또는 (분할이 아닌데) 1·2위 점수 차 margin_min 미만."""
    M, Mb, ids = community_matrix(memb)
    S = np.asarray((V @ M).todense())
    mass = np.asarray((V @ Mb).todense())
    tot = np.asarray(V.sum(1)).ravel()
    order = np.argsort(-S, axis=1)
    r = np.arange(len(S))
    t1, t2 = order[:, 0], order[:, 1] if S.shape[1] > 1 else order[:, 0]
    s1, s2 = S[r, t1], S[r, t2]
    cmass = mass.sum(1)
    cov = np.divide(cmass, tot, out=np.zeros_like(cmass), where=tot > 0)
    margin = np.divide(s1 - s2, s1, out=np.zeros_like(s1), where=s1 > 0)
    m1 = np.divide(mass[r, t1], cmass, out=np.zeros_like(cmass), where=cmass > 0)
    m2 = np.divide(mass[r, t2], cmass, out=np.zeros_like(cmass), where=cmass > 0)
    split = (m1 >= rules.get("split_min_mass", 0.25)) & (m2 >= rules.get("split_min_mass", 0.25)) & (s2 > 0)
    low = (nkw < rules.get("min_keywords", 2)) | (cov < rules.get("min_coverage", 0.3)) | \
          (~split & (margin < rules.get("margin_min", 0.1)))
    conf = np.where(s1 <= 0, "UNASSIGNED", np.where(low, "LOW", np.where(split, "SPLIT", "HIGH")))
    w1 = np.where(conf == "SPLIT", s1 / np.maximum(s1 + s2, 1e-12), 1.0)
    return pd.DataFrame({
        "top_community": np.where(s1 > 0, ids[t1], -1), "top_score": s1.round(4),
        "second_community": np.where(s2 > 0, ids[t2], -1), "second_score": s2.round(4),
        "score_margin": margin.round(4), "keyword_coverage": cov.round(4),
        "assignment_confidence": conf, "confidence_score": (cov * margin).round(4),
        "top_weight": w1.round(4), "second_weight": np.where(conf == "SPLIT", 1 - w1, 0).round(4)})


# ---------------------------------------------------------------- B: LSA 후 K-means

def lsa(V: sparse.csr_matrix, dim: int, seed: int = 0):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    dim = int(min(dim, V.shape[1] - 1, V.shape[0] - 1))
    svd = TruncatedSVD(n_components=dim, random_state=seed, algorithm="randomized", n_iter=7)
    Z = normalize(svd.fit_transform(V))
    return svd, Z


def lsa_kmeans(Z: np.ndarray, w: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=150)
    km.fit(Z, sample_weight=w)
    return km


def sector_weights(sec_lists: list[list[str]], power: float = 0.5) -> np.ndarray:
    """분야 기사 수의 제곱근 역수 (평균 1로 정규화). 여러 분야 기사는 분야 가중의 평균."""
    cnt = collections.Counter(s for l in sec_lists for s in set(l))
    w = np.array([np.mean([cnt[s] ** -power for s in set(l)]) if l else np.nan for l in sec_lists])
    w = np.where(np.isnan(w), np.nanmean(w), w)
    return w / w.mean()


def choose_k_silhouette(Z: np.ndarray, w: np.ndarray, ks: list[int], seed: int = 0, sample: int = 3000) -> tuple[int, float]:
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    best = (ks[0], -1.0)
    for k in ks:
        if k >= len(Z):
            break
        lab = lsa_kmeans(Z, w, k, seed).labels_
        if len(set(lab[idx])) < 2:
            continue
        s = silhouette_score(Z[idx], lab[idx], metric="cosine")
        if s > best[1]:
            best = (k, float(s))
    return best


# ---------------------------------------------------------------- 공통 평가

def seed_agreement(labels: list[np.ndarray]) -> tuple[float, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pairs = list(itertools.combinations(range(len(labels)), 2))
    if not pairs:
        return 1.0, 1.0
    return (float(np.mean([adjusted_rand_score(labels[a], labels[b]) for a, b in pairs])),
            float(np.mean([normalized_mutual_info_score(labels[a], labels[b]) for a, b in pairs])))


def sector_capture(lab: np.ndarray, confident: np.ndarray, w: np.ndarray, Ssec: np.ndarray, lift_min: float = 2.0) -> np.ndarray:
    """분야별 가중 기사 중, 그 분야가 lift_min배 이상 몰린 주제에 신뢰 배정된 비율."""
    ok = confident & (lab >= 0)
    labs = np.unique(lab[ok])
    W = w[:, None] * Ssec
    tot_s = W.sum(0)
    overall = tot_s / max(w.sum(), 1e-12)
    cap = np.zeros(Ssec.shape[1])
    for t in labs:
        r = ok & (lab == t)
        share = W[r].sum(0) / max(w[r].sum(), 1e-12)
        lift = np.divide(share, overall, out=np.zeros_like(share), where=overall > 0)
        cap += np.where(lift >= lift_min, W[r].sum(0), 0)
    return np.divide(cap, tot_s, out=np.zeros_like(cap), where=tot_s > 0)


def topic_top_terms(lab: np.ndarray, ok: np.ndarray, Xm: sparse.csr_matrix, w: np.ndarray, n: int = 10) -> dict[int, np.ndarray]:
    out = {}
    for t in np.unique(lab[ok & (lab >= 0)]):
        r = np.where(ok & (lab == t))[0]
        m = np.asarray(Xm[r].T @ w[r]).ravel()
        out[int(t)] = np.argsort(-m)[:n][m[np.argsort(-m)[:n]] > 0]
    return out


def evaluate(name: str, lab: np.ndarray, conf: np.ndarray, seed_labels: list[np.ndarray], ctx: dict, noise: dict) -> dict:
    w = ctx["w"]
    confident = np.isin(conf, ["HIGH", "SPLIT"])
    ari, nmi = seed_agreement(seed_labels)
    sizes = pd.Series(w[lab >= 0]).groupby(lab[lab >= 0]).sum()
    cap = sector_capture(lab, confident, w, ctx["Ssec"])
    small = [ctx["sectors"].index(s) for s in ctx["small_sectors"] if s in ctx["sectors"]]
    top_d = topic_top_terms(lab, confident, ctx["Xd"], w)          # 전체 사전 기준 상위어 (일반어·시장 표현 판정)
    top_k = topic_top_terms(lab, confident, ctx["Xk"], w)          # 보존 노드 기준 상위어 (일관성)
    gen = [np.isin(v, ctx["noise_cols_d"]).sum() >= 5 for v in top_d.values()]
    coh = [npmi_coherence(list(v), ctx["Bk"], w) for v in top_k.values() if len(v) >= 2]
    single = []
    for t in top_d:
        r = np.where(confident & (lab == t))[0]
        cc = collections.Counter(x for i in r for x in ctx["companies"][i])
        single.append(bool(cc) and cc.most_common(1)[0][1] / max(len(r), 1) >= 0.8)
    return {
        "model": name, "topics": int((sizes >= ctx.get("min_topic_weight", 10)).sum()),
        "seed_ari": round(ari, 4), "seed_nmi": round(nmi, 4),
        "largest_share": round(float(sizes.max() / w.sum()) if len(sizes) else 1.0, 4),
        "low_or_unassigned_rate": round(float(w[~confident].sum() / w.sum()), 4),
        "split_rate": round(float(w[conf == "SPLIT"].sum() / w.sum()), 4),
        **{f"capture_{s}": round(float(cap[i]), 4) for i, s in enumerate(ctx["sectors"])},
        "small_sector_capture_min": round(float(min(cap[i] for i in small)) if small else 0.0, 4),
        "general_topic_share": round(float(np.mean(gen)) if gen else 0.0, 4),
        "single_entity_share": round(float(np.mean(single)) if single else 0.0, 4),
        "coherence": round(float(np.mean(coh)) if coh else 0.0, 4),
        **{f"noise_{k}_highconf": round(v, 4) for k, v in noise.items()},
    }


def gates(m: pd.DataFrame, base: pd.Series, rules: dict) -> pd.DataFrame:
    g = rules.get("gates", {})
    m = m.copy()
    m["gate_stability"] = m["seed_ari"] >= base["seed_ari"] + g.get("ari_gain", 0.2)
    m["gate_largest"] = m["largest_share"] <= g.get("largest_share_max", 0.10)
    m["gate_low_conf"] = m["low_or_unassigned_rate"] <= g.get("low_conf_max", 0.30)
    m["gate_small_sector"] = m["small_sector_capture_min"] > base["small_sector_capture_min"]
    m["gate_general"] = m["general_topic_share"] <= base["general_topic_share"]
    m["gate_noise"] = (m["noise_exclude_highconf"] <= base["noise_exclude_highconf"]) & \
                      (m["noise_probe_highconf"] <= base["noise_probe_highconf"])
    gc = [c for c in m.columns if c.startswith("gate_")]
    m["gates_passed"] = m[gc].sum(axis=1)
    m["pass_all"] = m[gc].all(axis=1)
    return m


# ---------------------------------------------------------------- 주제 유형

def topic_type(noise_risk: float, relevance: float, general: bool, policy_hits: int, ent_conc: float,
               core_share: dict[str, float], rules: dict, rel_thr: float) -> tuple[str, list[str]]:
    flags = []
    if noise_risk >= rules.get("noise_risk_max", 0.2) or general or relevance < rel_thr:
        flags.append("NOISE_CANDIDATE")
    if policy_hits >= rules.get("policy_min_hits", 2):
        flags.append("POLICY_EVENT")
    if ent_conc >= rules.get("entity_concentration", 0.7):
        flags.append("ENTITY_EVENT")
    big = [s for s, v in core_share.items() if v >= rules.get("cross_min_share", 0.2)]
    if len(big) >= 2 and max(core_share.values(), default=0) < rules.get("cross_max_share", 0.6):
        flags.append("CROSS_SECTOR")
    primary = next((f for f in ("NOISE_CANDIDATE", "POLICY_EVENT", "ENTITY_EVENT", "CROSS_SECTOR") if f in flags), "INDUSTRY_TOPIC")
    return primary, flags


# ---------------------------------------------------------------- 실행

def run_e41(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    from kiwipiepy import Kiwi
    from sklearn.cluster import MiniBatchKMeans

    from .config import Lexicon
    from .e3 import latest_e2, weighted_log_odds
    from .e4 import best_match_jaccard
    from .e5 import latest_run
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .text import _space_free, prepped_texts, space_joined_names, token_form, tokenize_corpus

    rules = {**cfg.get("e4", {}), **cfg.get("e41", {})}
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e41_topic_structure", cfg, out_root, snap["data_snapshot_id"])
    from .selection import selected_e2, selected_e3
    e2_run, e2_path, e2_sum = selected_e2(cfg, REGISTRY, out_root)
    e3_run = selected_e3(REGISTRY, e2_run)
    e3_dir = out_root / "runs" / e3_run
    rel_thr = e2_sum["otsu_threshold"]
    lex = Lexicon.load()
    cache_dir = out_root / "cache" / "e4"
    base_key = {"data_snapshot_id": snap["data_snapshot_id"], "dictionary": e3_run}
    rng = np.random.default_rng(0)

    # ------------------------------------------------ 입력
    with run.stage("load") as st:
        X = sparse.load_npz(e3_dir / "03_doc_term.npz").tocsr()
        vocab = json.loads((e3_dir / "03_doc_term_vocab.json").read_text(encoding="utf-8"))
        rows = pd.read_csv(e3_dir / "03_doc_term_rows.csv", dtype=str)["gid"]
        dic0 = pd.read_csv(e3_dir / "03_keyword_dictionary.csv")
        dic = dic0.set_index("keyword").reindex(vocab).rename_axis("keyword").reset_index()
        dic["class_auto"] = dic["class_auto"].fillna("DROP")
        dic["entity_type"] = dic["entity_type"].fillna("개념")
        inp = pd.read_parquet(e3_dir / "03_e3_input_articles.parquet").set_index("article_id")
        ents = pd.read_csv(e3_dir / "05_protected_entities.csv")
        rel = pd.read_parquet(e2_path).set_index("gid")
        corpus = build_corpus(raw, cfg)
        arts = corpus.articles.reset_index(drop=True)
        assert arts["gid"].astype(str).tolist() == rows.tolist(), "E3 행렬과 기사 집합이 다르다 (스냅샷 확인)"
        gids = arts["gid"].to_numpy()
        half = inp.loc[gids, "content_weight_half"].to_numpy()
        prel = inp.loc[gids, "content_weight_soft"].to_numpy()
        decision = inp.loc[gids, "relevance_decision"].to_numpy()
        use = half > 0
        st["rows_out"] = int(use.sum())

    # ------------------------------------------------ 노드 유형 (제외 사유 분리)
    with run.stage("node_types") as st:
        related = cfg["cleaning"].get("related_label", "연관산업")
        ac = corpus.article_company
        company_sectors = collections.defaultdict(set)
        for c, secs in zip(ac["company"], ac["sectors"]):
            company_sectors[token_form(c)].update(secs if isinstance(secs, (list, tuple, set, np.ndarray)) else [secs])
        core = set(corpus.article_sector["sector"].unique()) - {related}
        am = rel.loc[gids, "anchor_market"].fillna(False).astype(float).to_numpy()
        acn = rel.loc[gids, "anchor_content"].fillna(False).astype(float).to_numpy()
        z_market = weighted_log_odds(X, am, acn)
        market_words = set(cfg["layers"]["market_seed"]) | set(cfg.get("e3", {}).get("market_phrase_words", [])) | lex.stopwords
        cand = dic["class_auto"].isin(["CORE", "EXTENDED", "EMERGING"]).to_numpy()
        exc_w = (decision == "EXCLUDE").astype(float)
        dic["delta_B"] = log_odds_delta(X, half, exc_w)             # B Half 효과 크기
        ref = dic[cand]                                              # 분위수 기준 = 네트워크 후보 (DROP·REVIEW 제외)
        nt = node_types(dic, z_market, market_words, company_sectors, core, ref, rules, kiwi=None)
        # 형태소 확인(NNP)은 네트워크 후보(CORE·EXTENDED·EMERGING)에만 적용
        kiwi = Kiwi()
        nt_c = node_types(dic[cand].reset_index(drop=True), z_market[cand], market_words, company_sectors, core, ref, rules, kiwi)
        nt.loc[cand, nt_c.columns] = nt_c.to_numpy()
        nt["concept_keep"] = nt["node_type"].isin(CONCEPT_KEEP) & np.isin(nt["class_auto"], rules.get("concept_classes", ["CORE", "EXTENDED"]))
        nt["extended_keep"] = nt["node_type"].isin(EXT_KEEP) & cand
        nt[cand].to_csv(run.dir / "07b_node_types.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07b_node_types.csv", "node_types", rows=int(cand.sum()))
        concept_nodes = np.where(nt["concept_keep"].to_numpy())[0]
        ext_nodes = np.where(nt["extended_keep"].to_numpy())[0]
        d_nodes = np.where(cand)[0]                                   # 전체 사전 (평가용 상위어)
        noise_cols_d = np.where(nt["node_type"].isin(NOISE_TYPES).to_numpy()[d_nodes])[0]
        st["rows_out"] = int(cand.sum())

    # ------------------------------------------------ 문장·제목 행렬 (캐시)
    with run.stage("sentences") as st:
        names = space_joined_names(sorted(corpus.article_company["company"].unique()),
                                   cfg["cleaning"].get("company_aliases") or {})
        prot = ents[ents["protect"]]["entity"]
        restore = {_space_free(e): e for e in prot}
        names_tok = names + [e for e in prot if e not in set(names)]
        texts = prepped_texts(corpus.articles, names_tok)
        tokens = tokenize_corpus(texts, names_tok, out_root / "cache")
        S_all, owner, is_title = sentence_matrix(cache_dir, {**base_key, "kind": "sentences_all"}, tokens, texts, lex,
                                                 set(cfg["keywords"].get("attach_suffixes", [])), restore, vocab)
        tr = np.where(is_title)[0]
        T = (sparse.csr_matrix((np.ones(len(tr)), (owner[tr], tr)), shape=(len(arts), S_all.shape[0])) @ S_all)
        T.data[:] = 1
        st["rows_out"] = S_all.shape[0]

    # 특성: 제목 가중 (제목 용어 title_weight배) + 요약
    tw = rules.get("title_weight", 2.0)
    Fall = (X + (tw - 1) * T).tocsr()
    iu = np.where(use)[0]
    w = half[iu]
    Fk = Fall[iu][:, ext_nodes]
    Vk, idf_k = tfidf(Fk)
    nkw = np.asarray((X[iu][:, ext_nodes] > 0).sum(1)).ravel()
    sec_lists_all = corpus.article_sector.groupby("gid")["sector"].agg(list)
    sectors = sorted(core)
    Ssec = np.array([[s in set(sec_lists_all.get(g, [])) for s in sectors] for g in gids[iu]], float)
    ctx = {"w": w, "Ssec": Ssec, "sectors": sectors, "small_sectors": rules.get("small_sectors", ["만화", "애니메이션", "캐릭터"]),
           "Xd": X[iu][:, d_nodes].tocsr(), "Xk": (X[iu][:, ext_nodes] > 0).astype(np.float32).tocsr(),
           "Bk": (X[iu][:, ext_nodes] > 0).astype(np.float32).tocsc(), "noise_cols_d": noise_cols_d,
           "companies": arts["companies"].to_numpy()[iu]}

    # 잡음 평가용 사후 배정 기사 (학습 입력에 섞지 않음)
    with run.stage("noise_eval_set") as st:
        exc = np.where(decision == "EXCLUDE")[0]
        exc = rng.choice(exc, min(rules.get("eval_exclude_n", 3000), len(exc)), replace=False)
        rv = np.where(decision == "REVIEW")[0]
        rv_low = rv[prel[rv] <= np.quantile(prel[rv], 0.25)]
        rv_low = rng.choice(rv_low, min(rules.get("eval_review_low_n", 2000), len(rv_low)), replace=False)
        vpos = {t: i for i, t in enumerate(vocab)}
        pcols = [vpos[p] for p in rules.get("noise_probes", []) if p in vpos]
        probe = np.unique(X[:, pcols].nonzero()[0]) if pcols else np.array([], int)
        eval_sets = {"exclude": exc, "review_low": rv_low, "probe": probe}
        st["rows_out"] = int(sum(len(v) for v in eval_sets.values()))

    def eval_features(r):
        return tfidf(Fall[r][:, ext_nodes], idf_k)[0], np.asarray((X[r][:, ext_nodes] > 0).sum(1)).ravel()

    ev_feat = {k: eval_features(v) for k, v in eval_sets.items()}

    # ------------------------------------------------ 네트워크: 기본 백본 / 기본 + 희소 분야 보완 백본
    net = rules.get("network", {})
    topk, res = net.get("topk", 15), net.get("resolution", 1.4)
    seeds = list(range(rules.get("leiden_seeds", 5)))
    bb = {"base": (net.get("base_min_co", 10), net.get("base_npmi", 0.20)),
          "rare": (net.get("rare_min_co", 5), net.get("rare_npmi", 0.30))}
    smask = use[owner]
    units = {"article": (X[iu], w), "sentence": (S_all[smask], half[owner[smask]])}
    dfh = dic["doc_freq_half"].fillna(0).to_numpy()
    bb_rows, memberships = [], {}
    with run.stage("backbones") as st:
        for layer, nodes in (("concept", concept_nodes), ("extended", ext_nodes)):
            for unit, (B, uw) in units.items():
                Bl = B[:, nodes].tocsr()
                edges = {}
                for name, (mc, thr) in bb.items():
                    ek = {**network_key(base_key, e2_run, half, nodes_hash(nt["node_type"].tolist(), range(len(nt))), net), "layer": layer, "nodes": nodes_hash(vocab, nodes), "unit": unit, "min_co": mc, "npmi": thr, "topk": topk}
                    edges[name] = (ek, cached_edges(cache_dir, ek, lambda: npmi_edges(Bl, uw, mc, thr, topk)))
                union = pd.concat([edges["base"][1], edges["rare"][1]]).sort_values("npmi", ascending=False) \
                    .drop_duplicates(["i", "j"]).sort_values(["i", "j"]).reset_index(drop=True)
                variants = {"base": (edges["base"][0], edges["base"][1]),
                            "base_rare": ({**edges["base"][0], "union_with": edges["rare"][0]}, union)}
                for vname, (ek, e) in variants.items():
                    runs_ = [cached_leiden(cache_dir, ek, len(nodes), e, res, s) for s in seeds]
                    met = community_metrics([m for m, _ in runs_], [q for _, q in runs_])
                    m0 = runs_[0][0]
                    iso = m0 < 0
                    has_comm = np.asarray((X[iu][:, nodes[~iso]] > 0).sum(1)).ravel() > 0
                    small_nodes = dic.loc[nodes, "top_sector"].isin(ctx["small_sectors"]).to_numpy()
                    bb_rows.append({"layer": layer, "unit": unit, "backbone": vname, "nodes": len(nodes), "edges": len(e),
                                    "rare_only_edges": int(len(e) - len(edges["base"][1])) if vname == "base_rare" else 0,
                                    **met, "isolated_weighted_df_share": round(float(dfh[nodes][iso].sum() / max(dfh[nodes].sum(), 1e-9)), 4),
                                    "articles_without_community_term": round(float(w[~has_comm].sum() / w.sum()), 4),
                                    "small_sector_nodes_isolated_rate": round(float(iso[small_nodes].mean()) if small_nodes.any() else 0.0, 4)})
                    memberships[(layer, unit, vname)] = [m for m, _ in runs_]
        bbd = pd.DataFrame(bb_rows)
        bbd.to_csv(run.dir / "07b_backbone_comparison.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07b_backbone_comparison.csv", "backbone_comparison", rows=len(bbd))
        st["rows_out"] = len(bbd)

    results, models = [], {}

    # ------------------------------------------------ 기준선: E4 희소 TF-IDF K-means (같은 지표로 재평가)
    with run.stage("baseline_sparse_kmeans") as st:
        Xd_use = X[iu][:, d_nodes].tocsr()
        nd = np.asarray((Xd_use > 0).sum(1)).ravel()
        fit = nd >= rules.get("min_keywords", 2)
        Vd, idf_d = tfidf(Xd_use[fit])
        labs = []
        for s in seeds:
            km = MiniBatchKMeans(n_clusters=160, random_state=s, batch_size=4096, n_init=3, max_iter=200).fit(Vd, sample_weight=w[fit])
            lab = np.full(len(iu), -1)
            lab[fit] = km.labels_
            lab[~fit & (nd > 0)] = km.predict(tfidf(Xd_use[~fit & (nd > 0)], idf_d)[0]) if (~fit & (nd > 0)).any() else []
            labs.append(lab)
            if s == 0:
                km0 = km
        conf = np.where(labs[0] < 0, "UNASSIGNED", np.where(fit, "HIGH", "LOW"))
        tr_sim = centroid_similarity(Vd, km0.cluster_centers_, km0.labels_)
        med = pd.Series(tr_sim).groupby(km0.labels_).median()
        noise = {}
        for k, r in eval_sets.items():
            Ve = tfidf(X[r][:, d_nodes], idf_d)[0]
            le = km0.predict(Ve)
            se = centroid_similarity(Ve, km0.cluster_centers_, le)
            noise[k] = float((se >= med.reindex(le).to_numpy()).mean()) if len(r) else 0.0
        results.append(evaluate("E4_baseline_sparse_kmeans_K160", labs[0], conf, labs, ctx, noise))
        st["rows_out"] = 1

    # ------------------------------------------------ A: Leiden 공동체 기반 기사 배정
    with run.stage("A_leiden_assignment") as st:
        for unit in ("article", "sentence"):
            for vname in ("base", "base_rare"):
                ms = memberships[("extended", unit, vname)]
                asg = [assign_leiden(Vk, m, nkw, rules) for m in ms]
                lab = asg[0]["top_community"].to_numpy()
                conf = asg[0]["assignment_confidence"].to_numpy()
                noise = {}
                for k, (Ve, nke) in ev_feat.items():
                    a = assign_leiden(Ve, ms[0], nke, rules)
                    noise[k] = float(a["assignment_confidence"].isin(["HIGH", "SPLIT"]).mean()) if len(a) else 0.0
                name = f"A_leiden_{unit}_{vname}"
                results.append(evaluate(name, lab, conf, [a["top_community"].to_numpy() for a in asg], ctx, noise))
                models[name] = {"kind": "A", "unit": unit, "backbone": vname, "memberships": ms, "assign": asg}
        st["rows_out"] = 4

    # ------------------------------------------------ B: LSA 후 K-means
    with run.stage("B_lsa_kmeans") as st:
        sw = sector_weights([list(set(sec_lists_all.get(g, [])) & core) or list(sec_lists_all.get(g, [])) for g in gids[iu]],
                            rules.get("sector_weight_power", 0.5))
        fitk = nkw >= rules.get("min_keywords", 2)
        for dim in rules.get("svd_dims", [100, 200, 300]):
            svd, Z = lsa(Vk[fitk], dim)
            Zlow = None
            for K in rules.get("lsa_k_grid", [80, 120, 160]):
                labs, kms = [], []
                for s in seeds:
                    km = lsa_kmeans(Z, (w * sw)[fitk], K, s)
                    lab = np.full(len(iu), -1)
                    lab[fitk] = km.labels_
                    low = ~fitk & (nkw > 0)
                    if low.any():
                        from sklearn.preprocessing import normalize
                        lab[low] = km.predict(normalize(svd.transform(Vk[low])))
                    labs.append(lab)
                    kms.append(km)
                conf = np.where(labs[0] < 0, "UNASSIGNED", np.where(fitk, "HIGH", "LOW"))
                km0 = kms[0]
                sim_tr = (Z * km0.cluster_centers_[km0.labels_]).sum(1) / np.maximum(np.linalg.norm(km0.cluster_centers_[km0.labels_], axis=1), 1e-9)
                med = pd.Series(sim_tr).groupby(km0.labels_).median()
                noise = {}
                from sklearn.preprocessing import normalize
                for k, (Ve, nke) in ev_feat.items():
                    if Ve.shape[0] == 0:
                        noise[k] = 0.0
                        continue
                    Ze = normalize(svd.transform(Ve))
                    le = km0.predict(Ze)
                    se = (Ze * km0.cluster_centers_[le]).sum(1) / np.maximum(np.linalg.norm(km0.cluster_centers_[le], axis=1), 1e-9)
                    noise[k] = float(((se >= med.reindex(le).to_numpy()) & (nke >= rules.get("min_keywords", 2))).mean())
                name = f"B_lsa{dim}_K{K}"
                results.append(evaluate(name, labs[0], conf, labs, ctx, noise))
                models[name] = {"kind": "B", "dim": dim, "K": K, "labels": labs}
        st["rows_out"] = len(results)

    with run.stage("gates") as st:
        mc = pd.DataFrame(results)
        is_base = mc["model"] == "E4_baseline_sparse_kmeans_K160"
        base = mc[is_base].iloc[0]
        mc = pd.concat([mc[is_base], gates(mc[~is_base], base, rules)], ignore_index=True)   # 기준선은 통과 기준 대상 아님 (빈칸)
        mc["role"] = mc["model"].map(lambda m: "baseline(E4)" if m.startswith("E4") else "operating_candidate" if m.startswith("A_") else "challenger")
        # 통과 모델끼리만 비교: 통과 기준 수 → 안정성 → 소규모 분야 포착 → 일관성
        rank_cols = ["gates_passed", "seed_ari", "small_sector_capture_min", "coherence"]
        A = mc[mc["role"] == "operating_candidate"].sort_values(rank_cols, ascending=False)
        Bc = mc[mc["role"] == "challenger"].sort_values(rank_cols, ascending=False)
        opA, bestB = A.iloc[0]["model"], Bc.iloc[0]["model"]
        mc.to_csv(run.dir / "07b_model_comparison.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07b_model_comparison.csv", "model_comparison", rows=len(mc))
        st["rows_out"] = len(mc)

    # ------------------------------------------------ C: 분야별 층화 LSA 군집 (보완 실험)
    with run.stage("C_stratified") as st:
        c_rows, c_cent = [], []
        k_total = rules.get("stratified_k_total", 120)
        n_by = Ssec[fitk].sum(0)
        memb_op = models[opA]["memberships"][0]
        _, Mb_op, ids_op = community_matrix(memb_op)
        c_stab = {}
        for si, s in enumerate(sectors):
            r = np.where(fitk & (Ssec[:, si] > 0))[0]
            if len(r) < 50:
                continue
            Ks = int(max(2, round(k_total * n_by[si] / n_by.sum())))
            svd_s, Zs = lsa(Vk[r], rules.get("stratified_dim", 100))
            labs = [lsa_kmeans(Zs, w[r], Ks, sd).labels_ for sd in seeds]
            c_stab[s] = round(seed_agreement(labs)[0], 4)
            for c in range(Ks):
                rr = r[labs[0] == c]
                if not len(rr):
                    continue
                cen = np.asarray(Vk[rr].T @ w[rr]).ravel() / max(w[rr].sum(), 1e-9)
                cm = np.asarray(Mb_op.T @ cen).ravel()
                top = np.argsort(-cen)[:10]
                c_rows.append({"sector": s, "cluster_id": f"{s}_{c:03d}", "articles": len(rr), "weighted": round(float(w[rr].sum()), 1),
                               "top_keywords": ", ".join(vocab[ext_nodes[i]] for i in top),
                               "best_leiden_community": int(ids_op[cm.argmax()]) if cm.sum() > 0 else -1,
                               "leiden_mass_share": round(float(cm.max() / max(cen.sum(), 1e-9)), 4) if cm.sum() > 0 else 0.0})
                c_cent.append(cen / max(np.linalg.norm(cen), 1e-9))
        cdf = pd.DataFrame(c_rows)
        Cm = np.vstack(c_cent) if c_cent else np.zeros((0, len(ext_nodes)))
        sim = Cm @ Cm.T
        link_thr = rules.get("stratified_link_cosine", 0.5)
        links = []
        for a, b in zip(*np.where(np.triu(sim, 1) >= link_thr)):
            if cdf.at[a, "sector"] != cdf.at[b, "sector"]:
                links.append({"a": cdf.at[a, "cluster_id"], "b": cdf.at[b, "cluster_id"], "cosine": round(float(sim[a, b]), 4),
                              "same_leiden": bool(cdf.at[a, "best_leiden_community"] == cdf.at[b, "best_leiden_community"])})
        ldf = pd.DataFrame(links, columns=["a", "b", "cosine", "same_leiden"])
        cdf["cross_sector_links"] = cdf["cluster_id"].map(pd.concat([ldf["a"], ldf["b"]]).value_counts()).fillna(0).astype(int)
        multi = cdf.groupby("best_leiden_community")["sector"].nunique()
        cdf["leiden_shared_by_sectors"] = cdf["best_leiden_community"].map(multi)
        cdf.to_csv(run.dir / "07b_stratified_clusters.csv", index=False, encoding="utf-8-sig")
        ldf.to_csv(run.dir / "07b_stratified_links.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07b_stratified_clusters.csv", "stratified_clusters", rows=len(cdf))
        # 소규모 분야 회복: 그 분야 층화 군집 중 운영 Leiden 공동체와 대응(질량 ≥ 0.3)이 약한 군집 = Leiden이 놓친 주제 후보
        weak = cdf["leiden_mass_share"] < rules.get("stratified_weak_mass", 0.3)
        c_sum = {"stability_ari_by_sector": c_stab, "clusters_by_sector": cdf["sector"].value_counts().to_dict(),
                 "missed_by_leiden_by_sector": cdf[weak]["sector"].value_counts().to_dict(),
                 "cross_sector_links": len(ldf), "cross_links_same_leiden": int(ldf["same_leiden"].sum()) if len(ldf) else 0,
                 "leiden_communities_shared_by_2plus_sectors": int((multi >= 2).sum())}
        st["rows_out"] = len(cdf)

    # ------------------------------------------------ 계층형 주제: 운영 A + 큰·저응집 공동체만 LSA 세분화
    with run.stage("hierarchy") as st:
        mA = models[opA]
        asg = mA["assign"][0].copy()
        asg.insert(0, "gid", gids[iu])
        lab = asg["top_community"].to_numpy()
        confident = asg["assignment_confidence"].isin(["HIGH", "SPLIT"]).to_numpy()
        # 주제별 가중 기사 (분할 가중 반영)
        tw_ = collections.defaultdict(float)
        for t1, t2, w1, w2, wi, ok in zip(asg["top_community"], asg["second_community"], asg["top_weight"], asg["second_weight"], w, confident):
            if ok:
                tw_[t1] += w1 * wi
                if w2 > 0:
                    tw_[t2] += w2 * wi
        topics = sorted(t for t in set(lab[lab >= 0]) if tw_.get(t, 0) > 0)
        top_k = topic_top_terms(lab, confident, ctx["Xk"], w, 10)
        top_d = topic_top_terms(lab, confident, ctx["Xd"], w, 10)
        coh = {t: npmi_coherence(list(top_k.get(t, [])), ctx["Bk"], w) if len(top_k.get(t, [])) >= 2 else 0.0 for t in topics}
        sizes = pd.Series({t: tw_[t] for t in topics})
        big_cut = sizes.quantile(rules.get("refine_size_quantile", 0.9))
        low_cut = pd.Series(coh).quantile(rules.get("refine_coherence_quantile", 0.25))
        sub_rows, sub_of = [], np.array([""] * len(iu), dtype=object)
        from sklearn.preprocessing import normalize
        for t in topics:
            if not (sizes[t] >= big_cut or coh[t] <= low_cut):
                continue
            r = np.where(confident & (lab == t) & fitk)[0]
            if len(r) < rules.get("refine_min_articles", 100):
                continue
            _, Zt = lsa(Vk[r], min(100, len(r) - 1))
            k, sil = choose_k_silhouette(Zt, w[r], list(range(2, 9)))
            labs = [lsa_kmeans(Zt, w[r], k, sd).labels_ for sd in seeds]
            ari_t = seed_agreement(labs)[0]
            for j in range(k):
                rr = r[labs[0] == j]
                m = np.asarray(ctx["Xk"][rr].T @ w[rr]).ravel()
                sub_of[rr] = f"tp_{t:03d}.{j + 1}"
                sub_rows.append({"subtopic_id": f"tp_{t:03d}.{j + 1}", "topic_id": f"tp_{t:03d}", "articles": len(rr),
                                 "weighted_article_count": round(float(w[rr].sum()), 1),
                                 "top_keywords": ", ".join(vocab[ext_nodes[i]] for i in np.argsort(-m)[:12]),
                                 "k_chosen": k, "silhouette": round(sil, 4), "seed_ari": round(ari_t, 4),
                                 "refine_reason": ";".join(x for x, f in (("large", sizes[t] >= big_cut), ("low_coherence", coh[t] <= low_cut)) if f)})
        asg["subtopic_id"] = [s if s else (f"tp_{t:03d}.0" if t >= 0 else "") for s, t in zip(sub_of, lab)]
        asg["topic_id"] = [f"tp_{t:03d}" if t >= 0 else "" for t in lab]
        asg["second_topic_id"] = [f"tp_{t:03d}" if t >= 0 and c == "SPLIT" else "" for t, c in zip(asg["second_community"], asg["assignment_confidence"])]
        asg["weight_half"] = w
        asg["p_rel"] = prel[iu]
        asg["date"] = arts["date"].to_numpy()[iu]
        asg["title"] = arts["title"].to_numpy()[iu]
        bl = models[bestB]["labels"][0]
        asg[f"challenger_{bestB}"] = bl
        asg.to_parquet(run.dir / "07b_article_assignment.parquet", index=False)
        run.artifact(run.dir / "07b_article_assignment.parquet", "article_assignment", rows=len(asg))
        sub = pd.DataFrame(sub_rows)
        sub.to_csv(run.dir / "07b_subtopic_registry.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07b_subtopic_registry.csv", "subtopic_registry", rows=len(sub))
        st["rows_out"] = len(sub)

    # ------------------------------------------------ 주제 등록부 + 잡음 위험 + 유형
    with run.stage("topic_registry") as st:
        ms0 = mA["memberships"][0]
        noise_hits = collections.Counter()
        for k in ("exclude", "probe"):
            Ve, nke = ev_feat[k]
            if Ve.shape[0] == 0:
                continue
            a = assign_leiden(Ve, ms0, nke, rules)
            for t in a.loc[a["assignment_confidence"].isin(["HIGH", "SPLIT"]), "top_community"]:
                noise_hits[t] += 1
        stab = collections.defaultdict(list)
        for other in mA["assign"][1:]:
            j = best_match_jaccard(lab, other["top_community"].to_numpy())
            for t in topics:
                stab[t].append(float(j.get(t, 0)))
        comp = ctx["companies"]
        policy = set(rules.get("policy_words", []))
        reg, pack = [], []
        nt_by_kw = nt.set_index("keyword")["node_type"]
        for t in topics:
            r = np.where(confident & (lab == t))[0]
            tk = [vocab[ext_nodes[i]] for i in top_k.get(t, [])]
            td = [vocab[d_nodes[i]] for i in top_d.get(t, [])]
            ents_ = [x for x in tk if nt_by_kw.get(x) in ("content_company", "work_person_policy", "proper_noun_content", "noncontent_company_specific")]
            cc = collections.Counter(x for i in r for x in comp[i])
            ent_conc = cc.most_common(1)[0][1] / max(len(r), 1) if cc else 0.0
            sw_ = (w[r, None] * Ssec[r]).sum(0)
            core_share = {s: float(v / max(sw_.sum(), 1e-9)) for s, v in zip(sectors, sw_)}
            relv = float(np.average(prel[iu][r], weights=w[r])) if len(r) else 0.0
            general = sum(nt_by_kw.get(x) in NOISE_TYPES for x in td) >= 5
            nr = noise_hits.get(t, 0) / max(len(r) + noise_hits.get(t, 0), 1)
            ph = sum(contains_word(x, policy) for x in td)
            primary, flags = topic_type(nr, relv, general, ph, ent_conc, core_share, rules, rel_thr)
            reps = asg.iloc[r].sort_values("top_score", ascending=False).head(5)
            bnd = asg.iloc[r].sort_values("top_score").head(3)
            cm = sorted(np.where(ms0 == t)[0], key=lambda i: -dfh[ext_nodes[i]])
            nsub = int((sub["topic_id"] == f"tp_{t:03d}").sum()) if len(sub) else 0
            row = {"topic_id": f"tp_{t:03d}", "leiden_community": int(t), "model": opA,
                   "community_nodes": len(cm), "community_keywords": ", ".join(vocab[ext_nodes[i]] for i in cm[:15]),
                   "top_keywords": ", ".join(tk), "top_entities": ", ".join(ents_[:8]),
                   "top_company": cc.most_common(1)[0][0] if cc else "", "entity_concentration": round(ent_conc, 3),
                   "article_count": len(r), "weighted_article_count": round(float(tw_[t]), 1),
                   "low_confidence_articles": int(((lab == t) & ~confident).sum()),
                   "sector_distribution": json.dumps({s: round(v, 3) for s, v in sorted(core_share.items(), key=lambda x: -x[1]) if v > 0}, ensure_ascii=False),
                   "topic_relevance_score": round(relv, 3), "coherence": round(float(coh[t]), 4),
                   "stability": round(float(np.mean(stab[t])) if stab[t] else 1.0, 3),
                   "noise_risk": round(nr, 4), "noise_eval_hits": int(noise_hits.get(t, 0)),
                   "topic_type": primary, "type_flags": ";".join(flags), "policy_hits": ph,
                   "subtopics": nsub,
                   "representative_articles": " || ".join(f"{g} {x}" for g, x in zip(reps["gid"], reps["title"])),
                   "first_month": str(pd.Timestamp(asg.iloc[r]["date"].min()).to_period("M")) if len(r) else None,
                   "last_month": str(pd.Timestamp(asg.iloc[r]["date"].max()).to_period("M")) if len(r) else None,
                   "run_id": run.run_id}
            reg.append(row)
            bdist = pd.Series(bl[r]).value_counts(normalize=True).head(3)
            pack.append({"type": "topic", **{k: row[k] for k in ("topic_id", "topic_type", "type_flags", "top_keywords", "top_entities",
                                                                  "weighted_article_count", "sector_distribution", "entity_concentration",
                                                                  "coherence", "stability", "noise_risk", "subtopics")},
                         "representative_articles": [{"gid": g, "title": x, "date": str(pd.Timestamp(d).date())}
                                                     for g, x, d in zip(reps["gid"], reps["title"], reps["date"])],
                         "boundary_articles": [{"gid": g, "title": x, "score": float(s)} for g, x, s in zip(bnd["gid"], bnd["title"], bnd["top_score"])],
                         "challenger_split": [{"cluster": f"{bestB}_{int(i)}", "share": round(float(v), 3)} for i, v in bdist.items()]})
        treg = pd.DataFrame(reg)
        treg.to_csv(run.dir / "07b_topic_registry.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07b_topic_registry.csv", "topic_registry", rows=len(treg))
        st["rows_out"] = len(treg)

    # ------------------------------------------------ 보고서
    for r in mc.to_dict("records"):
        pack.insert(0, {"type": "model", **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}})
    for r in sub.to_dict("records"):
        pack.append({"type": "subtopic", **r})
    with open(run.dir / "e41_comparison_pack.jsonl", "w", encoding="utf-8") as f:
        for p in pack:
            f.write(json.dumps(p, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    run.artifact(run.dir / "e41_comparison_pack.jsonl", "comparison_pack", rows=len(pack))
    with pd.ExcelWriter(run.dir / "e41_topic_report.xlsx") as xw:
        mc.to_excel(xw, sheet_name="model_gates", index=False)
        bbd.to_excel(xw, sheet_name="backbones", index=False)
        treg.to_excel(xw, sheet_name="topics", index=False)
        sub.to_excel(xw, sheet_name="subtopics", index=False)
        cdf.to_excel(xw, sheet_name="stratified", index=False)
        ldf.to_excel(xw, sheet_name="stratified_links", index=False)
        nt[cand].groupby("node_type").agg(n=("keyword", "size"), examples=("keyword", lambda s: ", ".join(s.head(25)))).reset_index() \
            .to_excel(xw, sheet_name="node_types", index=False)
    run.artifact(run.dir / "e41_topic_report.xlsx", "e41_topic_report")

    summary = {
        "e3_run_id": e3_run, "e2_run_id": e2_run,
        "node_types": nt[cand]["node_type"].value_counts().to_dict(),
        "network_nodes": {"concept": len(concept_nodes), "extended": len(ext_nodes)},
        "backbones": bbd.to_dict("records"),
        "models": mc.drop(columns=[c for c in mc.columns if c.startswith("capture_")]).to_dict("records"),
        "operating_A": opA, "best_challenger_B": bestB,
        "operating_pass_all": bool(mc.set_index("model").loc[opA, "pass_all"]),
        "challenger_pass_all": bool(mc.set_index("model").loc[bestB, "pass_all"]),
        "stratified": c_sum,
        "topics": len(treg), "subtopics": len(sub), "refined_topics": int(sub["topic_id"].nunique()) if len(sub) else 0,
        "topic_types": treg["topic_type"].value_counts().to_dict(),
        "assignment_confidence": asg["assignment_confidence"].value_counts().to_dict(),
        "selection_rule": "통과 기준 우선 → 통과 수·안정성·소규모 분야 포착·일관성 순. A는 운영 기준선, B는 도전자, C는 보완 실험",
    }
    (run.dir / "e41_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_topic_v2_e41", "run_id": run.run_id, "task": "topic_structure",
        "data_snapshot_id": snap["data_snapshot_id"], "input_e3_run_id": e3_run,
        "models": list(mc["model"]),
        "result": {"operating_A": opA, "best_challenger_B": bestB, "topics": len(treg), "subtopics": len(sub)}})
    run.finish()
    return summary
