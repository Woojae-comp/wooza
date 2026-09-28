"""E4.2 주제 구조 확정 (2026-09-28 사용자 결정, decision_log 기록).

역할 분담
| 구성 | 역할 |
|---|---|
| LSA 100차원 + K-means K=120 | 주제의 기본 단위와 기사 배정 (base_model = LSA100_K120) |
| Leiden 개념 네트워크 (기사 단위, 기본 + 희소 분야 보완 백본) | 주제 핵심 개념과 하위 구조 해석 |
| Leiden 개체 확장 네트워크 (문장 단위, 기본 백본) | 기업·작품·인물·정책 연결과 교차 분야 분석 |
| 분야별 층화 군집 | 기본 모델이 놓친 소규모 분야 주제 복구 (조건 통과 시 sector_rescue_topic=1) |
| LLM | 주제명 생성과 모델 차이 설명 (비교 패키지만 생성) |

- GENERIC_BRIDGE (AI·플랫폼·방송): 낮은 특이도지만 정책·산업 해석에 필요한 교차 분야 연결어.
  Leiden 공동체 형성에서 제외, 사전 유지, 주제명·설명과 분야 확산·교차 분야 지표에는 사용.
  연결어를 포함한 복합어(생성형 AI·AI 에이전트·팬덤 플랫폼·방송 제작)는 일반 개념 노드.
- 배정 거부: 군집마다 내부 기사의 중심 유사도 분포 하위 reject_quantile을 경계로, 경계 밖은 REJECTED
  (평가 기사는 UNASSIGNED_NOISE). 1·2위 중심 유사도 차가 학습 기사 하위 margin_quantile 미만이면 LOW.
  잡음 흡수 = 경계 안에 HIGH로 들어온 평가 기사만.
- 잡음 점수: 저특이도 일반어 비중 + 시장 앵커 기사 비중 + 외부 기업·종목명(나열) 비중 + 증권 구문 비중
  − 콘텐츠 개념어 비중 − 콘텐츠 앵커 기사 비중. 사유 코드는 따로 저장.
- 하위 주제: 최대 실루엣과 차이가 tolerance 이내인 가장 작은 k → 시드 ARI, 최소 크기·비중, 일관성 개선을 모두 통과해야 분할.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .e4 import cached_edges, cached_leiden, community_metrics, nodes_hash, npmi_coherence, npmi_edges, tfidf
from .e41 import (CONCEPT_KEEP, EXT_KEEP, community_matrix, contains_word, log_odds_delta, lsa, lsa_kmeans, node_types,
                  sector_capture, sector_weights, seed_agreement, topic_top_terms)

BRIDGE = "generic_bridge"
EXTERNAL = {"proper_noun_noncontent", "noncontent_company"}
ENTITY_ANY = {"proper_noun_content", "proper_noun_noncontent", "content_company", "noncontent_company",
              "noncontent_company_specific", "work_person_policy"}
CONTENT_TYPES = {"concept", "content_company", "work_person_policy", "proper_noun_content", "noncontent_company_specific", BRIDGE}
REASONS = ["GENERIC_WORD_DOMINANT", "MARKET_ARTICLE_DOMINANT", "EXTERNAL_ENTITY_LIST", "LOW_CONTENT_RELEVANCE",
           "LOW_COHERENCE", "NOISE_PROBE_ABSORPTION"]


# ---------------------------------------------------------------- GENERIC_BRIDGE

def apply_bridge(nt: pd.DataFrame, bridge: set[str]) -> pd.DataFrame:
    """단일어 연결어 → generic_bridge. 연결어를 포함한 복합어가 저특이도 일반어로 분류됐으면 개념 노드로 되돌린다."""
    nt = nt.copy()
    single = nt["keyword"].isin(bridge)
    nt.loc[single, "node_type"] = BRIDGE
    nt.loc[single, "reason"] = "교차 분야 연결어 (사용자 결정): Leiden 제외, 사전·주제명·확산 지표 사용"
    comp = ~single & nt["keyword"].str.contains(" ") & nt["keyword"].map(lambda t: bool(set(t.split(" ")) & bridge)) \
        & (nt["node_type"] == "general_low_specificity")
    nt.loc[comp, "node_type"] = "concept"
    nt.loc[comp, "reason"] = "연결어 포함 복합어 → 개념 노드"
    return nt


# ---------------------------------------------------------------- 배정·거부

def top2_similarity(Z: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """정규화된 LSA 벡터와 군집 중심의 코사인. 반환: 1위 군집, 1위 유사도, 2위 군집, 2위 유사도."""
    C = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    S = Z @ C.T
    o = np.argsort(-S, axis=1)[:, :2]
    r = np.arange(len(S))
    return o[:, 0], S[r, o[:, 0]], o[:, 1], S[r, o[:, 1]]


def rejection_bounds(sim: np.ndarray, lab: np.ndarray, q: float) -> pd.Series:
    """군집별 내부 기사 중심 유사도의 하위 q 분위 (배정 거부 경계)."""
    return pd.Series(sim).groupby(lab).quantile(q)


def classify_assignment(lab, sim, margin, bounds: pd.Series, margin_thr: float) -> np.ndarray:
    inside = sim >= bounds.reindex(lab).fillna(np.inf).to_numpy()
    return np.where(~inside, "REJECTED", np.where(margin < margin_thr, "LOW", "HIGH"))


# ---------------------------------------------------------------- 잡음 점수

def noise_components(r: np.ndarray, w: np.ndarray, Xd: sparse.csr_matrix, types_d: np.ndarray, market_anchor: np.ndarray,
                     content_anchor: np.ndarray, list_flag: np.ndarray) -> dict:
    """주제 기사 r의 가중 구성 비중. 용어 비중은 전체 사전(CORE·EXTENDED·EMERGING) 가중 질량 대비."""
    ww = w[r]
    tot_w = max(ww.sum(), 1e-12)
    mass = np.asarray(Xd[r].T @ ww).ravel()
    tm = max(mass.sum(), 1e-12)

    def share(types):
        return float(mass[np.isin(types_d, list(types))].sum() / tm)
    return {"generic_share": share({"general_low_specificity"}), "market_phrase_share": share({"market_expression"}),
            "external_entity_share": share(EXTERNAL), "content_concept_share": share({"concept"}),
            "content_share": share(CONTENT_TYPES),              # 콘텐츠 개념어 + 콘텐츠 기업·작품·인물 + 연결어
            "bridge_share": share({BRIDGE}),
            "market_article_share": float((ww * market_anchor[r]).sum() / tot_w),
            "content_anchor_share": float((ww * content_anchor[r]).sum() / tot_w),
            "entity_list_share": float((ww * list_flag[r]).sum() / tot_w)}


def noise_score(c: dict) -> float:
    return (c["generic_share"] + c["market_article_share"] + max(c["external_entity_share"], c["entity_list_share"])
            + c["market_phrase_share"] - c["content_share"] - c["content_anchor_share"])


def noise_reasons(c: dict, relevance: float, coherence: float, absorption: float, rules: dict, rel_thr: float,
                  coh_cut: float) -> list[str]:
    out = []
    if c["generic_share"] >= c["content_share"]:
        out.append("GENERIC_WORD_DOMINANT")
    if c["market_article_share"] >= rules.get("market_article_dominant", 0.5):
        out.append("MARKET_ARTICLE_DOMINANT")
    if c["entity_list_share"] >= rules.get("entity_list_dominant", 0.5) or \
            c["external_entity_share"] >= max(c["content_share"], 1e-9) * rules.get("external_vs_concept", 0.5):
        out.append("EXTERNAL_ENTITY_LIST")
    if relevance < rel_thr:
        out.append("LOW_CONTENT_RELEVANCE")
    if coherence < coh_cut:
        out.append("LOW_COHERENCE")
    if absorption >= rules.get("probe_absorption_max", 0.1):
        out.append("NOISE_PROBE_ABSORPTION")
    return out


def is_noise(score: float, reasons: list[str], rules: dict) -> bool:
    """잡음 점수가 양수(잡음 성분이 콘텐츠 성분보다 큼)이거나, 구성 사유(일반어·시장·종목 나열)가 하나라도 있으면."""
    structural = {"GENERIC_WORD_DOMINANT", "MARKET_ARTICLE_DOMINANT", "EXTERNAL_ENTITY_LIST"}
    return score > rules.get("noise_score_max", 0.0) or bool(structural & set(reasons)) or len(reasons) >= 2


# ---------------------------------------------------------------- 하위 주제

def pick_k(sils: dict[int, float], tolerance: float) -> int:
    """최대 실루엣과 차이가 tolerance 이내인 가장 작은 k."""
    best = max(sils.values())
    return min(k for k, s in sils.items() if s >= best - tolerance)


def subtopic_gate(ari: float, sizes: np.ndarray, coh_gain: float, rules: dict) -> tuple[bool, str]:
    total = max(sizes.sum(), 1e-12)
    if ari < rules.get("sub_seed_ari", 0.60):
        return False, f"시드 ARI {ari:.2f} < {rules.get('sub_seed_ari', 0.60)}"
    if sizes.min() < rules.get("sub_min_docs", 30):
        return False, f"최소 하위 주제 {int(sizes.min())}건 < {rules.get('sub_min_docs', 30)}"
    if sizes.min() / total < rules.get("sub_min_share", 0.03):
        return False, f"최소 비중 {sizes.min() / total:.3f} < {rules.get('sub_min_share', 0.03)}"
    if coh_gain < rules.get("sub_coherence_gain", 0.05):
        return False, f"일관성 개선 {coh_gain:+.3f} < {rules.get('sub_coherence_gain', 0.05)}"
    return True, f"통과 (ARI {ari:.2f}, 일관성 {coh_gain:+.3f})"


# ---------------------------------------------------------------- 층화 복구

def rescue_flags(max_sim: np.ndarray, docs: np.ndarray, coherence: np.ndarray, rules: dict, coh_cut: float,
                 noise: np.ndarray | None = None) -> np.ndarray:
    """K=120 주제와 유사도 낮음 ∩ 최소 규모 ∩ 일관성 ∩ 잡음 아님 (잡음 점수·사유는 기본 주제와 같은 규칙)."""
    ok = (max_sim < rules.get("rescue_max_cosine", 0.35)) & (docs >= rules.get("rescue_min_docs", 30)) & (coherence >= coh_cut)
    return ok & ~noise if noise is not None else ok


# ---------------------------------------------------------------- 실행

def run_e42(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    from kiwipiepy import Kiwi
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    from .config import Lexicon
    from .e3 import latest_e2, weighted_log_odds
    from .e4 import best_match_jaccard, sentence_matrix
    from .e5 import latest_run
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .text import _space_free, prepped_texts, space_joined_names, token_form, tokenize_corpus

    rules = {**cfg.get("e4", {}), **cfg.get("e41", {}), **cfg.get("e42", {})}
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e42_topic_final", cfg, out_root, snap["data_snapshot_id"])
    e3_run = latest_run(REGISTRY, "keyword_dictionary")
    e3_dir = out_root / "runs" / e3_run
    e2_run, e2_path, e2_sum = latest_e2(REGISTRY, out_root)
    rel_thr = e2_sum["otsu_threshold"]
    lex = Lexicon.load()
    cache_dir = out_root / "cache" / "e4"
    base_key = {"data_snapshot_id": snap["data_snapshot_id"], "dictionary": e3_run}
    rng = np.random.default_rng(0)
    seeds = list(range(rules.get("kmeans_seeds", 5)))
    DIM, K = rules.get("base_dim", 100), rules.get("base_k", 120)
    base_model = f"LSA{DIM}_K{K}"

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

    # ------------------------------------------------ 노드 유형 + GENERIC_BRIDGE
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
        dic["delta_B"] = log_odds_delta(X, half, (decision == "EXCLUDE").astype(float))
        ref = dic[cand]
        nt = node_types(dic, z_market, market_words, company_sectors, core, ref, rules, kiwi=None)
        nt_c = node_types(dic[cand].reset_index(drop=True), z_market[cand], market_words, company_sectors, core, ref, rules, Kiwi())
        nt.loc[cand, nt_c.columns] = nt_c.to_numpy()
        bridge = set(rules.get("generic_bridge", ["AI", "플랫폼", "방송"]))
        nt = apply_bridge(nt, bridge)
        nt["concept_keep"] = nt["node_type"].isin(CONCEPT_KEEP) & nt["class_auto"].isin(rules.get("concept_classes", ["CORE", "EXTENDED"])).to_numpy()
        nt["extended_keep"] = nt["node_type"].isin(EXT_KEEP) & cand
        nt[cand].to_csv(run.dir / "07c_node_types.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07c_node_types.csv", "node_types", rows=int(cand.sum()))
        concept_nodes = np.where(nt["concept_keep"].to_numpy())[0]
        ext_nodes = np.where(nt["extended_keep"].to_numpy())[0]
        d_nodes = np.where(cand)[0]
        types_d = nt["node_type"].to_numpy()[d_nodes]
        bridge_cols = np.where(nt["node_type"].to_numpy() == BRIDGE)[0]
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
        T = sparse.csr_matrix((np.ones(len(tr)), (owner[tr], tr)), shape=(len(arts), S_all.shape[0])) @ S_all
        T.data[:] = 1
        st["rows_out"] = S_all.shape[0]

    tw = rules.get("title_weight", 2.0)
    Fall = (X + (tw - 1) * T).tocsr()
    iu = np.where(use)[0]
    w = half[iu]
    Vk, idf_k = tfidf(Fall[iu][:, ext_nodes])
    nkw = np.asarray((X[iu][:, ext_nodes] > 0).sum(1)).ravel()
    fitk = nkw >= rules.get("min_keywords", 2)
    sec_lists_all = corpus.article_sector.groupby("gid")["sector"].agg(list)
    sectors = sorted(core)
    Ssec = np.array([[s in set(sec_lists_all.get(g, [])) for s in sectors] for g in gids[iu]], float)
    Xd = X[iu][:, d_nodes].tocsr()
    Xk = (X[iu][:, ext_nodes] > 0).astype(np.float32).tocsr()
    Bk = Xk.tocsc()
    companies = arts["companies"].to_numpy()[iu]
    ent_cols = np.where(np.isin(types_d, list(ENTITY_ANY)))[0]
    list_flag = (np.asarray((Xd[:, ent_cols] > 0).sum(1)).ravel() >= rules.get("entity_list_min", 6)).astype(float)
    ctx_small = rules.get("small_sectors", ["만화", "애니메이션", "캐릭터"])

    # ------------------------------------------------ Leiden 두 층 (캐시)
    net = rules.get("network", {})
    topk, res = net.get("topk", 15), net.get("resolution", 1.4)
    smask = use[owner]
    units = {"article": (X[iu], w), "sentence": (S_all[smask], half[owner[smask]])}
    layers_cfg = {"concept": ("article", True), "extended": ("sentence", False)}   # (단위, 희소 보완 백본 사용)
    net_rows, comm_memb, comm_rows = [], {}, []
    dfh = dic["doc_freq_half"].fillna(0).to_numpy()
    with run.stage("networks") as st:
        for layer, (unit, use_rare) in layers_cfg.items():
            nodes = concept_nodes if layer == "concept" else ext_nodes
            B, uw = units[unit]
            Bl = B[:, nodes].tocsr()
            mk = lambda mc, thr: {**base_key, "layer": layer, "nodes": nodes_hash(vocab, nodes), "unit": unit,
                                  "min_co": mc, "npmi": thr, "topk": topk}
            kb = mk(net.get("base_min_co", 10), net.get("base_npmi", 0.20))
            e = cached_edges(cache_dir, kb, lambda: npmi_edges(Bl, uw, kb["min_co"], kb["npmi"], topk))
            key = kb
            if use_rare:
                kr = mk(net.get("rare_min_co", 5), net.get("rare_npmi", 0.30))
                er = cached_edges(cache_dir, kr, lambda: npmi_edges(Bl, uw, kr["min_co"], kr["npmi"], topk))
                e = pd.concat([e, er]).sort_values("npmi", ascending=False).drop_duplicates(["i", "j"]) \
                    .sort_values(["i", "j"]).reset_index(drop=True)
                key = {**kb, "union_with": kr}
            runs_ = [cached_leiden(cache_dir, key, len(nodes), e, res, s) for s in range(rules.get("leiden_seeds", 5))]
            met = community_metrics([m for m, _ in runs_], [q for _, q in runs_])
            m0 = runs_[0][0]
            iso = m0 < 0
            net_rows.append({"layer": layer, "unit": unit, "backbone": "base+rare" if use_rare else "base", "nodes": len(nodes),
                             "edges": len(e), **met,
                             "isolated_weighted_df_share": round(float(dfh[nodes][iso].sum() / max(dfh[nodes].sum(), 1e-9)), 4)})
            comm_memb[layer] = (nodes, m0)
            wdeg = np.bincount(np.r_[e["i"], e["j"]], weights=np.r_[e["npmi"], e["npmi"]], minlength=len(nodes))
            for c in sorted(set(m0[m0 >= 0])):
                mk_ = np.where(m0 == c)[0]
                ordr = mk_[np.argsort(-wdeg[mk_])]
                comm_rows.append({"layer": layer, "community_id": f"{layer[:3]}_{c:03d}", "size": len(mk_),
                                  "keywords": ", ".join(vocab[nodes[k]] for k in ordr[:15])})
        nets = pd.DataFrame(net_rows)
        comms = pd.DataFrame(comm_rows)
        comms.to_csv(run.dir / "07c_network_communities.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07c_network_communities.csv", "network_communities", rows=len(comms))
        st["rows_out"] = len(comms)

    # ------------------------------------------------ 기본 모델: LSA100 + K-means K=120
    with run.stage("base_model") as st:
        sw = sector_weights([list(set(sec_lists_all.get(g, [])) & core) or list(sec_lists_all.get(g, [])) for g in gids[iu]],
                            rules.get("sector_weight_power", 0.5))
        svd, Z = lsa(Vk[fitk], DIM)
        kms = [lsa_kmeans(Z, (w * sw)[fitk], K, s) for s in seeds]
        km = kms[0]
        Zall = normalize(svd.transform(Vk))
        t1, s1, t2, s2 = top2_similarity(Zall, km.cluster_centers_)
        lab = np.where(nkw > 0, t1, -1)
        lab[fitk] = km.labels_                                          # 학습 기사는 K-means 배정 그대로
        sim = (Zall * (km.cluster_centers_[np.maximum(lab, 0)] /
                       np.maximum(np.linalg.norm(km.cluster_centers_[np.maximum(lab, 0)], axis=1, keepdims=True), 1e-12))).sum(1)
        second = np.where(t1 == lab, t2, t1)
        s_second = np.where(t1 == lab, s2, s1)
        margin = sim - s_second
        qs = rules.get("reject_quantile", 0.05)
        bounds = rejection_bounds(sim[fitk], lab[fitk], qs)
        bounds10 = rejection_bounds(sim[fitk], lab[fitk], rules.get("reject_quantile_alt", 0.10))
        margin_thr = float(np.quantile(margin[fitk], rules.get("margin_quantile", 0.10)))
        conf = classify_assignment(lab, sim, margin, bounds, margin_thr)
        conf = np.where(nkw == 0, "UNASSIGNED", np.where(~fitk & (conf == "HIGH"), "LOW", conf))
        seed_labs = []
        for k_ in kms:
            l_ = np.full(len(iu), -1)
            l_[fitk] = k_.labels_
            seed_labs.append(l_)
        ari, nmi = seed_agreement(seed_labs)
        st["rows_out"] = int(len(iu))

    # ------------------------------------------------ 잡음 평가 (군집별 거부 경계 적용)
    with run.stage("noise_eval") as st:
        exc = np.where(decision == "EXCLUDE")[0]
        exc = rng.choice(exc, min(rules.get("eval_exclude_n", 3000), len(exc)), replace=False)
        rv = np.where(decision == "REVIEW")[0]
        rv_low = rv[prel[rv] <= np.quantile(prel[rv], 0.25)]
        rv_low = rng.choice(rv_low, min(rules.get("eval_review_low_n", 2000), len(rv_low)), replace=False)
        vpos = {t: i for i, t in enumerate(vocab)}
        pcols = [vpos[p] for p in rules.get("noise_probes", []) if p in vpos]
        probe = np.unique(X[:, pcols].nonzero()[0]) if pcols else np.array([], int)
        ev_rows, absorbed, ev_assign = [], collections.Counter(), {}
        for name, r in {"exclude": exc, "review_low": rv_low, "probe": probe}.items():
            if not len(r):
                continue
            Ve = tfidf(Fall[r][:, ext_nodes], idf_k)[0]
            nke = np.asarray((X[r][:, ext_nodes] > 0).sum(1)).ravel()
            Ze = normalize(svd.transform(Ve))
            e1, es1, _, es2 = top2_similarity(Ze, km.cluster_centers_)
            ce = classify_assignment(e1, es1, es1 - es2, bounds, margin_thr)
            ce = np.where(nke < rules.get("min_keywords", 2), "LOW", ce)
            ce10 = classify_assignment(e1, es1, es1 - es2, bounds10, margin_thr)
            ce10 = np.where(nke < rules.get("min_keywords", 2), "LOW", ce10)
            nearest = float((nke > 0).mean())
            ev_rows.append({"group": name, "articles": len(r), "nearest_assigned": round(nearest, 4),
                            f"absorbed_q{int(qs * 100):02d}": round(float((ce == "HIGH").mean()), 4),
                            "absorbed_q10": round(float((ce10 == "HIGH").mean()), 4),
                            "unassigned_noise": round(float((ce == "REJECTED").mean()), 4),
                            "low_confidence": round(float((ce == "LOW").mean()), 4)})
            ev_assign[name] = (e1, ce)
            if name in ("exclude", "probe"):
                absorbed.update(e1[ce == "HIGH"].tolist())
        ev = pd.DataFrame(ev_rows)
        ev.to_csv(run.dir / "07c_noise_eval.csv", index=False, encoding="utf-8-sig")
        st["rows_out"] = len(ev)

    # ------------------------------------------------ 주제별 지표·잡음·Leiden 대응
    with run.stage("topics") as st:
        ok = np.isin(conf, ["HIGH", "LOW"])
        top_k = topic_top_terms(lab, ok, Xk, w, 10)
        coh = {t: npmi_coherence(list(v), Bk, w) if len(v) >= 2 else 0.0 for t, v in top_k.items()}
        coh_cut = float(np.quantile(list(coh.values()), rules.get("low_coherence_quantile", 0.10)))
        stab = collections.defaultdict(list)
        for other in seed_labs[1:]:
            j = best_match_jaccard(seed_labs[0][fitk], other[fitk])
            for t in range(K):
                stab[t].append(float(j.get(t, 0)))
        Mcs = {layer: community_matrix(m0) for layer, (nodes, m0) in comm_memb.items()}
        # 기사 특성(ext_nodes) → 각 층 노드 위치 대응표
        proj = {}
        for layer, (nodes, _) in comm_memb.items():
            pos = {n: i for i, n in enumerate(nodes)}
            src = np.array([i for i, n in enumerate(ext_nodes) if n in pos], dtype=int)
            proj[layer] = (src, np.array([pos[ext_nodes[i]] for i in src], dtype=int), len(nodes))
        policy = set(rules.get("policy_words", []))
        nt_by_kw = nt.set_index("keyword")["node_type"]
        cap = sector_capture(lab, ok, w, Ssec)
        reg = []
        for t in range(K):
            r = np.where(ok & (lab == t))[0]
            members = np.where(lab == t)[0]
            if not len(r):
                continue
            comp = noise_components(r, w, Xd, types_d, am[iu], acn[iu], list_flag)
            relv = float(np.average(prel[iu][r], weights=w[r]))
            absorb = absorbed.get(t, 0) / max(len(r), 1)
            reasons = noise_reasons(comp, relv, coh.get(t, 0.0), absorb, rules, rel_thr, coh_cut)
            score = noise_score(comp)
            tk = [vocab[ext_nodes[i]] for i in top_k.get(t, [])]
            # 연결어: 주제 안 비중이 전체 대비 lift ≥ 1
            bm = np.asarray(X[iu][r][:, bridge_cols].T @ w[r]).ravel() / max(w[r].sum(), 1e-12)
            ball = np.asarray(X[iu][:, bridge_cols].T @ w).ravel() / max(w.sum(), 1e-12)
            bkw = [vocab[bridge_cols[i]] for i in np.argsort(-bm) if bm[i] > 0 and ball[i] > 0 and bm[i] / ball[i] >= 1]
            # Leiden 대응 (중심 질량이 어느 공동체에 실리는지)
            cen = np.asarray(Vk[r].T @ w[r]).ravel()
            match = {}
            for layer in comm_memb:
                src, dst, n_ = proj[layer]
                v = np.zeros(n_)
                v[dst] = cen[src]
                _, Mb, ids = Mcs[layer]
                mm = np.asarray(Mb.T @ v).ravel()
                o = np.argsort(-mm)[:3]
                match[layer] = [f"{layer[:3]}_{ids[i]:03d}" for i in o if mm[i] > 0 and mm[i] >= 0.5 * mm[o[0]]]
            ents_ = [x for x in tk if nt_by_kw.get(x) in ENTITY_ANY]
            cc = collections.Counter(x for i in r for x in companies[i])
            ent_dom = cc.most_common(1)[0][1] / len(r) if cc else 0.0
            sw_ = (w[r, None] * Ssec[r]).sum(0)
            core_share = {s: float(v / max(sw_.sum(), 1e-9)) for s, v in zip(sectors, sw_)}
            p = np.array([v for v in core_share.values() if v > 0])
            spread = float(-(p * np.log(p)).sum() / np.log(len(sectors))) if len(p) > 1 else 0.0
            flags = []
            noise = is_noise(score, reasons, rules)
            if noise:
                flags.append("NOISE_CANDIDATE")
            if sum(contains_word(x, policy) for x in tk) >= rules.get("policy_min_hits", 2):
                flags.append("POLICY_EVENT")
            if ent_dom >= rules.get("entity_concentration", 0.7):
                flags.append("ENTITY_EVENT")
            big = [s for s, v in core_share.items() if v >= rules.get("cross_min_share", 0.2)]
            if len(big) >= 2 and max(core_share.values()) < rules.get("cross_max_share", 0.6):
                flags.append("CROSS_SECTOR")
            primary = next((f for f in ("NOISE_CANDIDATE", "POLICY_EVENT", "ENTITY_EVENT", "CROSS_SECTOR") if f in flags), "INDUSTRY_TOPIC")
            reps = r[np.argsort(-sim[r])[:5]]
            bnd = r[np.argsort(sim[r])[:3]]
            reg.append({
                "topic_id": f"tp_{t:03d}", "base_model": base_model, "cluster": t,
                "top_keywords": ", ".join(tk), "generic_bridge_keywords": ", ".join(bkw),
                "concept_communities": ";".join(match["concept"]), "extended_communities": ";".join(match["extended"]),
                "top_entities": ", ".join(ents_[:8]), "top_company": cc.most_common(1)[0][0] if cc else "",
                "entity_dominance": round(ent_dom, 3),
                "article_count": int(len(r)), "weighted_article_count": round(float(w[r].sum()), 1),
                "assignment_rejection_rate": round(float((conf[members] == "REJECTED").mean()), 4),
                "low_confidence_rate": round(float((conf[members] == "LOW").mean()), 4),
                "sector_distribution": json.dumps({s: round(v, 3) for s, v in sorted(core_share.items(), key=lambda x: -x[1]) if v > 0}, ensure_ascii=False),
                "sector_spread": round(spread, 3),
                "topic_relevance_score": round(relv, 3), "coherence": round(float(coh.get(t, 0.0)), 4),
                "stability": round(float(np.mean(stab[t])) if stab[t] else 1.0, 3),
                "noise_score": round(score, 4), **{k: round(v, 4) for k, v in comp.items()},
                "noise_probe_absorption": round(absorb, 4), "noise_reason_codes": ";".join(reasons),
                "topic_type": primary, "type_flags": ";".join(flags),
                "representative_articles": " || ".join(f"{gids[iu][i]} {arts['title'].iat[iu[i]]}" for i in reps),
                "_reps": reps, "_bnd": bnd,
                "first_month": str(pd.Timestamp(arts["date"].to_numpy()[iu][r].min()).to_period("M")),
                "last_month": str(pd.Timestamp(arts["date"].to_numpy()[iu][r].max()).to_period("M")),
                "sector_rescue_flag": 0, "run_id": run.run_id})
        treg = pd.DataFrame(reg)
        noisy = set(treg.loc[treg["topic_type"] == "NOISE_CANDIDATE", "cluster"])
        for i, row in ev.iterrows():
            e1, ce = ev_assign[row["group"]]
            ev.at[i, "absorbed_into_clean_topics"] = round(float(((ce == "HIGH") & ~np.isin(e1, list(noisy))).mean()), 4)
            ev.at[i, "absorbed_into_noise_topics"] = round(float(((ce == "HIGH") & np.isin(e1, list(noisy))).mean()), 4)
        ev.to_csv(run.dir / "07c_noise_eval.csv", index=False, encoding="utf-8-sig")
        st["rows_out"] = len(treg)

    # ------------------------------------------------ 하위 주제 (규칙 변경)
    with run.stage("subtopics") as st:
        big_cut = treg["weighted_article_count"].quantile(rules.get("refine_size_quantile", 0.9))
        low_cut = treg["coherence"].quantile(rules.get("refine_coherence_quantile", 0.25))
        sub_rows, sub_of = [], np.array([""] * len(iu), dtype=object)
        treg["subtopic_count"] = 0
        treg["subtopic_selection_reason"] = "대상 아님 (규모·일관성 기준 미해당)"
        for ti, row in treg.iterrows():
            t = row["cluster"]
            if not (row["weighted_article_count"] >= big_cut or row["coherence"] <= low_cut):
                continue
            r = np.where((conf == "HIGH") & (lab == t) & fitk)[0]
            if len(r) < 2 * rules.get("sub_min_docs", 30):
                treg.at[ti, "subtopic_selection_reason"] = f"기사 {len(r)}건 부족"
                continue
            _, Zt = lsa(Vk[r], min(DIM, len(r) - 1))
            idx = rng.choice(len(r), min(3000, len(r)), replace=False)
            sils = {}
            for k in range(2, 9):
                if k >= len(r):
                    break
                l_ = lsa_kmeans(Zt, w[r], k, 0).labels_
                if len(set(l_[idx])) > 1:
                    sils[k] = float(silhouette_score(Zt[idx], l_[idx], metric="cosine"))
            if not sils:
                continue
            k = pick_k(sils, rules.get("sub_silhouette_tolerance", 0.01))
            labs_ = [lsa_kmeans(Zt, w[r], k, s).labels_ for s in seeds]
            ari_t = seed_agreement(labs_)[0]
            sizes = np.bincount(labs_[0], minlength=k).astype(float)
            sub_coh = []
            for j in range(k):
                rr = r[labs_[0] == j]
                m = np.asarray(Xk[rr].T @ w[rr]).ravel()
                tt = np.argsort(-m)[:10]
                sub_coh.append(npmi_coherence(list(tt[m[tt] > 0]), Bk, w) if (m[tt] > 0).sum() >= 2 else 0.0)
            gain = float(np.average(sub_coh, weights=sizes) - row["coherence"])
            passed, why = subtopic_gate(ari_t, sizes, gain, rules)
            treg.at[ti, "subtopic_selection_reason"] = f"k={k} (실루엣 {sils[k]:.3f}, 최대 {max(sils.values()):.3f}) " + why
            if not passed:
                continue
            treg.at[ti, "subtopic_count"] = k
            for j in range(k):
                rr = r[labs_[0] == j]
                m = np.asarray(Xk[rr].T @ w[rr]).ravel()
                cc = collections.Counter(x for i in rr for x in companies[i])
                dom = cc.most_common(1)[0][1] / len(rr) if cc else 0.0
                sid = f"tp_{t:03d}.{j + 1}"
                sub_of[rr] = sid
                sub_rows.append({"subtopic_id": sid, "topic_id": f"tp_{t:03d}", "articles": len(rr),
                                 "share": round(float(sizes[j] / sizes.sum()), 3),
                                 "top_keywords": ", ".join(vocab[ext_nodes[i]] for i in np.argsort(-m)[:12]),
                                 "coherence": round(sub_coh[j], 4), "top_company": cc.most_common(1)[0][0] if cc else "",
                                 "entity_dominance": round(dom, 3),
                                 "subtopic_type": "ENTITY_EVENT" if dom >= rules.get("entity_concentration", 0.7) else "",
                                 "k_chosen": k, "seed_ari": round(ari_t, 4), "coherence_gain": round(gain, 4)})
        sub = pd.DataFrame(sub_rows, columns=["subtopic_id", "topic_id", "articles", "share", "top_keywords", "coherence",
                                              "top_company", "entity_dominance", "subtopic_type", "k_chosen", "seed_ari", "coherence_gain"])
        # 한 기업 기준으로만 갈라진 분할: 모든 하위 주제가 서로 다른 기업에 편중
        for tid, g in sub.groupby("topic_id"):
            if (g["entity_dominance"] >= rules.get("entity_concentration", 0.7)).all() and g["top_company"].nunique() == len(g):
                sub.loc[g.index, "subtopic_type"] = "ENTITY_EVENT"
                treg.loc[treg["topic_id"] == tid, "subtopic_selection_reason"] += " · 기업별 분할(ENTITY_EVENT)"
        st["rows_out"] = len(sub)

    # ------------------------------------------------ 분야별 층화 군집 → 복구
    with run.stage("sector_rescue") as st:
        tcent = np.zeros((K, Vk.shape[1]))
        for t in range(K):
            r = np.where(ok & (lab == t))[0]
            if len(r):
                tcent[t] = np.asarray(Vk[r].T @ w[r]).ravel()
        tcent = tcent / np.maximum(np.linalg.norm(tcent, axis=1, keepdims=True), 1e-12)
        k_total = rules.get("stratified_k_total", 120)
        n_by = Ssec[fitk].sum(0)
        c_rows, rescue_of = [], np.array([""] * len(iu), dtype=object)
        med_coh = float(treg["coherence"].median())
        for si, s in enumerate(sectors):
            r = np.where(fitk & (Ssec[:, si] > 0))[0]
            if len(r) < 50:
                continue
            Ks = int(max(2, round(k_total * n_by[si] / n_by.sum())))
            _, Zs = lsa(Vk[r], rules.get("stratified_dim", 100))
            ls = lsa_kmeans(Zs, w[r], Ks, 0).labels_
            for c in range(Ks):
                rr = r[ls == c]
                if not len(rr):
                    continue
                cen = np.asarray(Vk[rr].T @ w[rr]).ravel()
                cen = cen / max(np.linalg.norm(cen), 1e-12)
                sims = tcent @ cen
                m = np.asarray(Xk[rr].T @ w[rr]).ravel()
                tt = np.argsort(-m)[:10]
                ch = npmi_coherence(list(tt[m[tt] > 0]), Bk, w) if (m[tt] > 0).sum() >= 2 else 0.0
                comp_c = noise_components(rr, w, Xd, types_d, am[iu], acn[iu], list_flag)
                rel_c = float(np.average(prel[iu][rr], weights=w[rr]))
                rs_c = noise_reasons(comp_c, rel_c, ch, 0.0, rules, rel_thr, coh_cut)
                sc_c = noise_score(comp_c)
                c_rows.append({"sector": s, "cluster_id": f"{s}_{c:03d}", "articles": len(rr), "weighted": round(float(w[rr].sum()), 1),
                               "noise_score": round(sc_c, 4), "noise_reason_codes": ";".join(rs_c),
                               "is_noise": is_noise(sc_c, rs_c, rules), "topic_relevance_score": round(rel_c, 3),
                               "top_keywords": ", ".join(vocab[ext_nodes[i]] for i in tt), "coherence": round(ch, 4),
                               "nearest_topic": f"tp_{int(sims.argmax()):03d}", "nearest_topic_cosine": round(float(sims.max()), 4),
                               "_rows": rr})
        cdf = pd.DataFrame(c_rows)
        cdf["sector_rescue_topic"] = rescue_flags(cdf["nearest_topic_cosine"].to_numpy(), cdf["articles"].to_numpy(),
                                                  cdf["coherence"].to_numpy(), rules, med_coh, cdf["is_noise"].to_numpy(bool)).astype(int)
        resc = []
        for _, rrow in cdf[cdf["sector_rescue_topic"] == 1].iterrows():
            rid = f"tr_{rrow['cluster_id']}"
            rescue_of[rrow["_rows"]] = rid
            resc.append({"topic_id": rid, "base_model": f"stratified_{rrow['sector']}", "top_keywords": rrow["top_keywords"],
                         "article_count": rrow["articles"], "weighted_article_count": rrow["weighted"], "coherence": rrow["coherence"],
                         "nearest_topic": rrow["nearest_topic"], "nearest_topic_cosine": rrow["nearest_topic_cosine"],
                         "sector_distribution": json.dumps({rrow["sector"]: 1.0}, ensure_ascii=False),
                         "topic_type": "INDUSTRY_TOPIC", "sector_rescue_flag": 1, "run_id": run.run_id})
        cdf.drop(columns=["_rows"]).to_csv(run.dir / "07c_stratified_candidates.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07c_stratified_candidates.csv", "stratified_candidates", rows=len(cdf))
        st["rows_out"] = len(resc)

    # ------------------------------------------------ 저장
    with run.stage("write") as st:
        asg = pd.DataFrame({
            "gid": gids[iu], "date": arts["date"].to_numpy()[iu], "title": arts["title"].to_numpy()[iu],
            "topic_id": [f"tp_{t:03d}" if t >= 0 and c != "UNASSIGNED" else "" for t, c in zip(lab, conf)],
            "subtopic_id": [s if s else (f"tp_{t:03d}.0" if t >= 0 and c in ("HIGH", "LOW") else "") for s, t, c in zip(sub_of, lab, conf)],
            "assignment_confidence": conf, "centroid_similarity": sim.round(4), "margin": margin.round(4),
            "second_topic_id": [f"tp_{t:03d}" for t in second], "rescue_topic_id": rescue_of,
            "weight_half": w, "p_rel": prel[iu], "entity_list_article": list_flag.astype(int),
            "anchor_market": am[iu].astype(int), "anchor_content": acn[iu].astype(int)})
        asg.to_parquet(run.dir / "07c_article_assignment.parquet", index=False)
        run.artifact(run.dir / "07c_article_assignment.parquet", "article_assignment", rows=len(asg))
        pack = []
        for _, row in treg.iterrows():
            pack.append({"type": "topic", **{k: row[k] for k in (
                "topic_id", "base_model", "topic_type", "type_flags", "top_keywords", "generic_bridge_keywords", "top_entities",
                "concept_communities", "extended_communities", "weighted_article_count", "sector_distribution", "entity_dominance",
                "coherence", "stability", "noise_score", "noise_reason_codes", "assignment_rejection_rate", "subtopic_count")},
                "representative_articles": [{"gid": gids[iu][i], "title": arts["title"].iat[iu[i]], "similarity": float(sim[i])} for i in row["_reps"]],
                "boundary_articles": [{"gid": gids[iu][i], "title": arts["title"].iat[iu[i]], "similarity": float(sim[i])} for i in row["_bnd"]],
                "subtopics": sub[sub["topic_id"] == row["topic_id"]][["subtopic_id", "top_keywords", "share"]].to_dict("records")})
        for r in resc:
            pack.append({"type": "rescue_topic", **r})
        for r in comms.to_dict("records"):
            pack.append({"type": "community", **r})
        treg_out = pd.concat([treg.drop(columns=["_reps", "_bnd", "cluster"]), pd.DataFrame(resc)], ignore_index=True)
        treg_out.to_csv(run.dir / "07c_topic_registry.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07c_topic_registry.csv", "topic_registry", rows=len(treg_out))
        sub.to_csv(run.dir / "07c_subtopic_registry.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07c_subtopic_registry.csv", "subtopic_registry", rows=len(sub))
        with open(run.dir / "e42_comparison_pack.jsonl", "w", encoding="utf-8") as f:
            for p in pack:
                f.write(json.dumps(p, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        run.artifact(run.dir / "e42_comparison_pack.jsonl", "comparison_pack", rows=len(pack))
        with pd.ExcelWriter(run.dir / "e42_topic_report.xlsx") as xw:
            treg_out.to_excel(xw, sheet_name="topics", index=False)
            sub.to_excel(xw, sheet_name="subtopics", index=False)
            cdf.drop(columns=["_rows"]).to_excel(xw, sheet_name="stratified", index=False)
            ev.to_excel(xw, sheet_name="noise_eval", index=False)
            nets.to_excel(xw, sheet_name="networks", index=False)
            comms.to_excel(xw, sheet_name="communities", index=False)
            nt[cand].groupby("node_type").agg(n=("keyword", "size"), examples=("keyword", lambda s: ", ".join(s.head(25)))) \
                .reset_index().to_excel(xw, sheet_name="node_types", index=False)
        run.artifact(run.dir / "e42_topic_report.xlsx", "e42_topic_report")
        st["rows_out"] = len(treg_out)

    small = [sectors.index(s) for s in ctx_small if s in sectors]
    wsz = pd.Series(w[ok]).groupby(lab[ok]).sum()
    summary = {
        "e3_run_id": e3_run, "base_model": base_model,
        "node_types": nt[cand]["node_type"].value_counts().to_dict(),
        "networks": nets.to_dict("records"),
        "base_metrics": {"seed_ari": round(ari, 4), "seed_nmi": round(nmi, 4),
                         "largest_share": round(float(wsz.max() / w.sum()), 4),
                         **{f"capture_{s}": round(float(cap[i]), 4) for i, s in enumerate(sectors)},
                         "small_sector_capture_min": round(float(min(cap[i] for i in small)), 4),
                         "reject_quantile": qs, "margin_threshold": round(margin_thr, 4)},
        "assignment_confidence": pd.Series(conf).value_counts().to_dict(),
        "noise_eval": ev.to_dict("records"),
        "topics": int(len(treg)), "noise_topics": int((treg["topic_type"] == "NOISE_CANDIDATE").sum()),
        "topic_types": treg["topic_type"].value_counts().to_dict(),
        "noise_reason_counts": collections.Counter(x for s in treg["noise_reason_codes"] for x in s.split(";") if x),
        "subtopic_topics": int((treg["subtopic_count"] > 0).sum()), "subtopics": len(sub),
        "stratified_candidates": int(len(cdf)), "sector_rescue_topics": len(resc),
        "rescue_by_sector": cdf[cdf["sector_rescue_topic"] == 1]["sector"].value_counts().to_dict(),
    }
    (run.dir / "e42_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_topic_v2_e42", "run_id": run.run_id, "task": "topic_final",
        "data_snapshot_id": snap["data_snapshot_id"], "input_e3_run_id": e3_run,
        "models": [base_model, "Leiden concept (article, base+rare)", "Leiden extended (sentence, base)", "stratified rescue"],
        "result": {k: summary[k] for k in ("base_metrics", "topics", "noise_topics", "subtopics", "sector_rescue_topics")}})
    run.finish()
    return summary
