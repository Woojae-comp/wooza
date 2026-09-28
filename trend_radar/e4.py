"""E4 주제 군집·네트워크 실험 (v2 설계 3장, 기획안 7장).

역할 분리 (경쟁 모델로 하나만 고르지 않는다)
- 문서 K-means: 같은 사건·주제를 다룬 기사 묶음 → **주제의 기본 단위**
- NPMI + Leiden: 주제를 구성하는 개념·개체의 관계 → 주제명·구조·교차 분야 해석
- 결합: 문서 군집에 네트워크 공동체를 붙여 topic_registry 구성

네트워크 두 층
- 개념 네트워크: CORE·EXTENDED 개념어. 특이도가 낮고 흔한 일반어(허브)는 뺀다
- 개체 확장 네트워크: 개념어 + 작품·기업·인물·정책명 (CORE·EXTENDED·EMERGING)

동시출현 단위는 기사·문장 두 가지. 두 단위 모두에서 나온 연결은 strong.
가중치: INCLUDE 1 / REVIEW 0.5 / EXCLUDE 0 (EXCLUDE 기사는 입력에서 빠짐).
자동 선택은 지표 순위 평균(Borda)이며, 최종 군집 선택은 정량지표와 연구자 기준으로 한다 (decision_log).
"""
from __future__ import annotations

import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SENT_END = re.compile(r"\n|(?<=[.!?…])\s+")


# ---------------------------------------------------------------- 공통

def pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def borda(df: pd.DataFrame, up: list[str], down: list[str]) -> pd.Series:
    """지표별 순위 평균 (낮을수록 좋음). 한 지표가 결과를 지배하지 않게 한다."""
    r = [df[c].rank(ascending=False) for c in up] + [df[c].rank(ascending=True) for c in down]
    return pd.concat(r, axis=1).mean(axis=1)


def general_words(dic: pd.DataFrame, rules: dict) -> set[str]:
    """허브 일반어: 개념어 중 B 특이도 하위 절반이면서 문서빈도 상위 10% (사업·서비스 같은 말).
    분위수 기준이라 사람이 단어를 고르지 않는다."""
    c = dic[dic["entity_type"] == "개념"]
    zp, dp = pct(c["z_B_half"].to_numpy()), pct(c["doc_freq"].to_numpy())
    m = (zp < rules.get("general_max_z_pct", 0.5)) & (dp >= rules.get("general_min_df_pct", 0.9))
    return set(c.loc[m, "keyword"])


# ---------------------------------------------------------------- 문장 단위 행렬

def sentence_spans(text: str) -> list[tuple[int, int]]:
    out, s = [], 0
    for m in SENT_END.finditer(text):
        if m.start() > s:
            out.append((s, m.start()))
        s = m.end()
    if s < len(text):
        out.append((s, len(text)))
    return out


def sentence_terms(tokens, texts, lexicon, attach, restore) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """문장별 후보어 (E3와 같은 추출 규칙). 반환: 문장별 용어 목록, 문장 → 기사 행 번호, 제목 문장 여부."""
    from .e3 import candidates

    groups, owner, gtexts, title = [], [], [], []
    for r, (toks, text) in enumerate(zip(tokens, texts)):
        spans = sentence_spans(text)
        nl = text.find("\n")
        b = [[] for _ in spans]
        j = 0
        for t in toks:
            while j < len(spans) - 1 and t[1] >= spans[j][1]:
                j += 1
            b[j].append(t)
        for k, g in enumerate(b):
            if g:
                groups.append(g)
                owner.append(r)
                gtexts.append(text)
                title.append(k == 0 and nl > 0 and spans[0][1] == nl)
    return candidates(groups, gtexts, lexicon, attach, restore=restore), np.array(owner, dtype=int), np.array(title, dtype=bool)


def incidence(doc_terms: list[list[str]], vocab: list[str]) -> sparse.csr_matrix:
    idx = {t: i for i, t in enumerate(vocab)}
    rows, cols = [], []
    for r, ts in enumerate(doc_terms):
        for t in ts:
            j = idx.get(t)
            if j is not None:
                rows.append(r)
                cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(doc_terms), len(vocab)))


# ---------------------------------------------------------------- NPMI 네트워크

def npmi_edges(B: sparse.csr_matrix, w: np.ndarray, min_co: int, min_npmi: float, topk: int) -> pd.DataFrame:
    """B: 단위(기사·문장) × 노드 0/1, w: 단위 가중치. 가중 NPMI, 최소 동시출현(가중 전 단위 수), 노드별 상위 k (어느 한쪽 기준)."""
    B = B.tocsr()
    keep_rows = w > 0
    B, w = B[keep_rows], w[keep_rows]
    Bw = sparse.diags(w) @ B
    N = float(w.sum())
    raw = (B.T @ B).tocsr()
    raw.sort_indices()
    cw = (B.T @ Bw).tocsr()
    cw.sort_indices()
    df = np.asarray(Bw.sum(0)).ravel()
    rows = np.repeat(np.arange(raw.shape[0]), np.diff(raw.indptr))
    cols = raw.indices
    m = (rows < cols) & (raw.data >= min_co)
    i, j, co = rows[m], cols[m], raw.data[m]
    same = raw.nnz == cw.nnz and np.array_equal(raw.indptr, cw.indptr) and np.array_equal(raw.indices, cw.indices)
    pij = (cw.data[m] if same else np.asarray(cw[i, j]).ravel()) / N
    ok = pij > 0
    i, j, pij, co = i[ok], j[ok], pij[ok], co[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = np.log(pij / ((df[i] / N) * (df[j] / N))) / -np.log(pij)
    e = pd.DataFrame({"i": i, "j": j, "npmi": npmi, "co_units": co.astype(int)})
    e = e[e["npmi"] >= min_npmi]
    if e.empty:
        return e.reset_index(drop=True)
    both = pd.concat([e.rename(columns={"i": "a", "j": "b"}), e.rename(columns={"j": "a", "i": "b"})])
    both["r"] = both.groupby("a")["npmi"].rank(ascending=False, method="first")
    keep = both[both["r"] <= topk]
    key = pd.DataFrame({"i": np.minimum(keep["a"], keep["b"]), "j": np.maximum(keep["a"], keep["b"])}).drop_duplicates()
    return e.merge(key, on=["i", "j"]).sort_values(["i", "j"]).reset_index(drop=True)


# ---------------------------------------------------------------- 캐시 (네트워크 격자·문장 행렬)
# 키: data_snapshot_id, 사전 버전(E3 run_id), 네트워크 층(노드 집합 해시), 동시출현 단위, 임계값, 상위 k, resolution, 시드

CACHE_VERSION = "e4cache_v2"   # v2: 네트워크 키에 E2 run·기사 가중 해시·노드 유형 버전·네트워크 설정 해시 포함


def cache_key(d: dict) -> str:
    import hashlib

    return hashlib.sha1(json.dumps({**d, "_v": CACHE_VERSION}, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def weight_hash(w: np.ndarray) -> str:
    """기사 가중치(관련성 판정) 해시. E2가 바뀌면 가중 동시출현·NPMI 캐시를 쓰지 않는다."""
    import hashlib

    return hashlib.sha1(np.round(np.asarray(w, dtype=float), 6).tobytes()).hexdigest()[:12]


def network_key(base_key: dict, e2_run: str, w: np.ndarray, node_type_version: str, network_cfg: dict) -> dict:
    """네트워크 캐시 키: data_snapshot_id + 사전 버전 + e2_run_id + 기사 가중 해시 + 노드 유형 버전 + 네트워크 설정 해시."""
    return {**base_key, "e2_run_id": e2_run, "article_weight_hash": weight_hash(w),
            "node_type_version": node_type_version, "network_config_hash": cache_key(network_cfg)}


def nodes_hash(vocab: list[str], nodes) -> str:
    import hashlib

    return hashlib.sha1("\n".join(vocab[i] for i in nodes).encode()).hexdigest()[:12]


def cached_edges(cache_dir: Path, key: dict, build) -> pd.DataFrame:
    p = cache_dir / f"edges_{cache_key(key)}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    e = build()
    cache_dir.mkdir(parents=True, exist_ok=True)
    e.to_parquet(p, index=False)
    p.with_suffix(".json").write_text(json.dumps(key, ensure_ascii=False, default=str), encoding="utf-8")
    return e


def cached_leiden(cache_dir: Path, key: dict, n: int, edges: pd.DataFrame, resolution: float, seed: int):
    k = {**key, "resolution": resolution, "seed": seed}
    p = cache_dir / f"leiden_{cache_key(k)}.npz"
    if p.exists():
        z = np.load(p)
        return z["memb"], float(z["q"])
    m, q = leiden(n, edges, resolution, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(p, memb=m, q=q)
    return m, q


def sentence_matrix(cache_dir: Path, key: dict, tokens, texts, lexicon, attach, restore, vocab):
    """모든 기사의 문장 × 용어 행렬, 문장 → 기사 행, 제목 문장 여부 (캐시)."""
    p = cache_dir / f"sent_{cache_key(key)}.npz"
    if p.exists():
        z = np.load(p)
        S = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return S, z["owner"], z["is_title"]
    sterms, owner, is_title = sentence_terms(tokens, texts, lexicon, attach, restore)
    S = incidence(sterms, vocab)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(p, data=S.data, indices=S.indices, indptr=S.indptr, shape=np.array(S.shape), owner=owner, is_title=is_title)
    return S, owner, is_title


def leiden(n: int, edges: pd.DataFrame, resolution: float, seed: int) -> tuple[np.ndarray, float]:
    """Leiden (RB 배치 모형, NPMI 가중). 반환: 노드별 공동체 번호(고립 노드 -1), 해상도 1 기준 가중 모듈러리티."""
    import igraph as ig
    import leidenalg

    g = ig.Graph(n=n, edges=list(zip(edges["i"].tolist(), edges["j"].tolist())))
    g.es["weight"] = edges["npmi"].tolist()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed, n_iterations=-1)
    memb = np.array(part.membership)
    q = g.modularity(memb.tolist(), weights="weight") if len(edges) else 0.0
    deg = np.asarray(g.degree())
    memb[deg == 0] = -1
    return memb, float(q)


def community_metrics(membs: list[np.ndarray], q: list[float]) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    m0 = membs[0]
    active = m0 >= 0
    sizes = pd.Series(m0[active]).value_counts()
    pairs = list(itertools.combinations(range(len(membs)), 2))
    return {
        "modularity": round(float(np.mean(q)), 4),
        "communities": int((sizes >= 3).sum()),
        "isolated_rate": round(float(1 - active.mean()), 4),
        "largest_share": round(float(sizes.max() / active.sum()) if active.any() else 1.0, 4),
        "seed_nmi": round(float(np.mean([normalized_mutual_info_score(membs[a], membs[b]) for a, b in pairs])), 4),
        "seed_ari": round(float(np.mean([adjusted_rand_score(membs[a], membs[b]) for a, b in pairs])), 4),
    }


# ---------------------------------------------------------------- 문서 군집

def tfidf(X: sparse.csr_matrix, idf: np.ndarray | None = None) -> tuple[sparse.csr_matrix, np.ndarray]:
    from sklearn.preprocessing import normalize

    if idf is None:
        df = np.asarray((X > 0).sum(0)).ravel()
        idf = np.log((1 + X.shape[0]) / (1 + df)) + 1
    return normalize(X @ sparse.diags(idf), norm="l2", axis=1).tocsr(), idf


def centroid_similarity(V: sparse.csr_matrix, centers: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """기사와 소속 군집 중심의 코사인 (대표 기사·경계 기사 선정용)."""
    out = np.zeros(V.shape[0])
    norms = np.maximum(np.linalg.norm(centers, axis=1), 1e-9)
    for c in np.unique(lab):
        r = np.where(lab == c)[0]
        out[r] = (V[r] @ centers[c]) / norms[c]
    return out


def kmeans(V, w: np.ndarray, k: int, seed: int):
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3, max_iter=200)
    km.fit(V, sample_weight=w)
    return km


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else 0.0


def npmi_coherence(terms_idx: list[int], B: sparse.csc_matrix, w: np.ndarray) -> float:
    """상위 핵심어 쌍의 평균 가중 NPMI (기사 단위)."""
    if len(terms_idx) < 2:
        return 0.0
    N = w.sum()
    cols = B[:, terms_idx].toarray() * w[:, None]
    bin_ = B[:, terms_idx].toarray() > 0
    p = cols.sum(0) / N
    vals = []
    for a, b in itertools.combinations(range(len(terms_idx)), 2):
        pab = (w * (bin_[:, a] & bin_[:, b])).sum() / N
        vals.append(-1.0 if pab <= 0 else np.log(pab / (p[a] * p[b])) / -np.log(pab))
    return float(np.mean(vals))


def best_match_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """군집 a 각각에 대해 b에서 가장 겹치는 군집과의 구성원 Jaccard (반복 실행 안정성)."""
    ct = pd.crosstab(a, b)
    sa, sb = ct.sum(axis=1).to_numpy()[:, None], ct.sum(axis=0).to_numpy()[None, :]
    jac = ct.to_numpy() / (sa + sb - ct.to_numpy())
    return pd.Series(jac.max(1), index=ct.index)


# ---------------------------------------------------------------- 실행

def run_e4(cfg: dict, raw: pd.DataFrame, out_root: Path) -> dict:
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    from .config import Lexicon
    from .e5 import latest_run
    from .load import build_corpus
    from .runlog import REGISTRY, Run, append_jsonl, snapshot
    from .text import _space_free, prepped_texts, space_joined_names, tokenize_corpus

    rules = cfg.get("e4", {})
    files = cfg.get("_input_files", [])
    last = str(raw[cfg["input"]["columns"]["date"]].astype(str).str[:10].max())
    snap = snapshot(files, last) if files else {"data_snapshot_id": None}
    run = Run("e4_topics", cfg, out_root, snap["data_snapshot_id"])
    from .selection import selected_e2, selected_e3
    e2_run_id, _, e2_sum = selected_e2(cfg, REGISTRY, out_root)
    e3_run = selected_e3(REGISTRY, e2_run_id)
    e3_dir = out_root / "runs" / e3_run
    e2_sum_thr = e2_sum["otsu_threshold"]   # 저관련 군집 기준 = E2 관련성 임계값
    lex = Lexicon.load()

    # ------------------------------------------------ 입력
    with run.stage("load") as st:
        X = sparse.load_npz(e3_dir / "03_doc_term.npz").tocsr()
        vocab = json.loads((e3_dir / "03_doc_term_vocab.json").read_text(encoding="utf-8"))
        rows = pd.read_csv(e3_dir / "03_doc_term_rows.csv", dtype=str)["gid"]
        dic0 = pd.read_csv(e3_dir / "03_keyword_dictionary.csv")    # DROP 대부분은 사전 파일에 없음 → 행렬 어휘에 맞춰 DROP으로 채움
        dic = dic0.set_index("keyword").reindex(vocab).rename_axis("keyword").reset_index()
        dic["class_auto"] = dic["class_auto"].fillna("DROP")
        dic["entity_type"] = dic["entity_type"].fillna("개념")
        inp = pd.read_parquet(e3_dir / "03_e3_input_articles.parquet").set_index("article_id")
        ents = pd.read_csv(e3_dir / "05_protected_entities.csv")
        corpus = build_corpus(raw, cfg)
        arts = corpus.articles.reset_index(drop=True)
        assert arts["gid"].astype(str).tolist() == rows.tolist(), "E3 행렬과 기사 집합이 다르다 (스냅샷 확인)"
        half = inp.loc[arts["gid"], "content_weight_half"].to_numpy()
        prel = inp.loc[arts["gid"], "content_weight_soft"].to_numpy()
        general = general_words(dic0, rules)
        cls, etype = dic["class_auto"].to_numpy(), dic["entity_type"].to_numpy()
        concept_nodes = np.where(np.isin(cls, rules.get("concept_classes", ["CORE", "EXTENDED"])) & (etype == "개념")
                                 & ~dic["keyword"].isin(general).to_numpy())[0]
        ext_nodes = np.where(np.isin(cls, rules.get("extended_classes", ["CORE", "EXTENDED", "EMERGING"]))
                             & ~dic["keyword"].isin(general).to_numpy())[0]
        use = half > 0
        st["rows_out"] = int(use.sum())

    # ------------------------------------------------ 문장 단위 행렬 (E3와 같은 추출 규칙, 토큰 캐시 재사용)
    with run.stage("sentences") as st:
        names = space_joined_names(sorted(corpus.article_company["company"].unique()),
                                   cfg["cleaning"].get("company_aliases") or {})
        prot = ents[ents["protect"]]["entity"]
        restore = {_space_free(e): e for e in prot}
        names_tok = names + [e for e in prot if e not in set(names)]
        texts = prepped_texts(corpus.articles, names_tok)
        tokens = tokenize_corpus(texts, names_tok, out_root / "cache")
        idx_use = np.where(use)[0]
        cache_dir = out_root / "cache" / "e4"
        base_key = {"data_snapshot_id": snap["data_snapshot_id"], "dictionary": e3_run}
        S_all, owner_all, _ = sentence_matrix(cache_dir, {**base_key, "kind": "sentences_all"}, tokens, texts, lex,
                                              set(cfg["keywords"].get("attach_suffixes", [])), restore, vocab)
        smask = use[owner_all]
        S = S_all[smask]
        s_w = half[owner_all[smask]]
        st["rows_out"] = S.shape[0]

    # ------------------------------------------------ 네트워크 격자
    grid = rules.get("network_grid", {})
    units = {"article": (X[idx_use], half[idx_use]), "sentence": (S, s_w)}
    layers = {"concept": concept_nodes, "extended": ext_nodes}
    seeds = list(range(rules.get("leiden_seeds", 5)))
    net_rows, nets = [], {}
    with run.stage("network_grid") as st:
        for layer, nodes in layers.items():
            for unit, (B, w) in units.items():
                Bl = B[:, nodes].tocsr()
                for min_co in grid.get("min_co", [5, 10]):
                    for thr in grid.get("npmi", [0.10, 0.20]):
                        for topk in grid.get("topk", [15, 20]):
                            ek = {**network_key(base_key, e2_run_id, half, "e4_general_v1", grid), "layer": layer, "nodes": nodes_hash(vocab, nodes), "unit": unit,
                                  "min_co": min_co, "npmi": thr, "topk": topk}
                            e = cached_edges(cache_dir, ek, lambda: npmi_edges(Bl, w, min_co, thr, topk))
                            nets[(layer, unit, min_co, thr, topk)] = e
                            for res in grid.get("resolution", [0.6, 1.0, 1.4]):
                                runs = [cached_leiden(cache_dir, ek, len(nodes), e, res, s) for s in seeds]
                                met = community_metrics([m for m, _ in runs], [q for _, q in runs])
                                net_rows.append({"layer": layer, "unit": unit, "min_co": min_co, "npmi": thr, "topk": topk,
                                                 "resolution": res, "nodes": len(nodes), "edges": len(e), **met,
                                                 "_memb": runs[0][0]})
        ng = pd.DataFrame(net_rows)
        ng["borda"] = np.nan
        for layer in layers:
            m = ng["layer"] == layer
            ng.loc[m, "borda"] = borda(ng[m], ["modularity", "seed_ari", "communities"], ["isolated_rate", "largest_share"])
        st["rows_out"] = len(ng)

    # 강한 연결 (기사·문장 공통) + 선택 설정
    sel_net = {}
    with run.stage("network_select") as st:
        node_rows, edge_rows, comm_rows = [], [], []
        for layer, nodes in layers.items():
            sub = ng[ng["layer"] == layer].sort_values("borda")
            best = sub.iloc[0]
            sel_net[layer] = best
            key = (layer, best["unit"], best["min_co"], best["npmi"], best["topk"])
            other = (layer, "sentence" if best["unit"] == "article" else "article", best["min_co"], best["npmi"], best["topk"])
            e = nets[key].copy()
            oset = set(zip(nets[other]["i"], nets[other]["j"]))
            e["strong"] = [(a, b) in oset for a, b in zip(e["i"], e["j"])]
            memb = best["_memb"]
            for a, b, v, c, s in e[["i", "j", "npmi", "co_units", "strong"]].itertuples(index=False):
                edge_rows.append({"layer": layer, "source": vocab[nodes[a]], "target": vocab[nodes[b]], "npmi": round(float(v), 4),
                                  "co_units": int(c), "unit": best["unit"], "strong": bool(s),
                                  "same_community": bool(memb[a] == memb[b] and memb[a] >= 0)})
            deg = np.bincount(np.r_[e["i"], e["j"]], minlength=len(nodes))
            wdeg = np.bincount(np.r_[e["i"], e["j"]], weights=np.r_[e["npmi"], e["npmi"]], minlength=len(nodes))
            for k, n in enumerate(nodes):
                node_rows.append({"layer": layer, "keyword": vocab[n], "keyword_id": dic.at[n, "keyword_id"],
                                  "entity_type": etype[n], "class_auto": cls[n], "degree": int(deg[k]),
                                  "weighted_degree": round(float(wdeg[k]), 3), "community": f"{layer[:3]}_{memb[k]:03d}" if memb[k] >= 0 else None,
                                  "doc_freq_half": dic.at[n, "doc_freq_half"], "top_sector": dic.at[n, "top_sector"]})
            for c in sorted(set(memb[memb >= 0])):
                mk = np.where(memb == c)[0]
                ordr = mk[np.argsort(-wdeg[mk])]
                kws = [vocab[nodes[k]] for k in ordr]
                sec = dic.loc[nodes[mk], "top_sector"].value_counts(normalize=True)
                comm_rows.append({"layer": layer, "community_id": f"{layer[:3]}_{c:03d}", "size": len(mk),
                                  "top_keywords": ", ".join(kws[:15]),
                                  "entities": ", ".join([vocab[nodes[k]] for k in ordr if etype[nodes[k]] != "개념"][:10]),
                                  "sector_distribution": json.dumps(sec.round(3).to_dict(), ensure_ascii=False),
                                  "cross_sector": bool((sec >= 0.2).sum() >= 2),
                                  "internal_strong_edges": int(e[(memb[e["i"]] == c) & (memb[e["j"]] == c)]["strong"].sum())})
        nodes_df, edges_df, comm_df = pd.DataFrame(node_rows), pd.DataFrame(edge_rows), pd.DataFrame(comm_rows)
        for df_, name in ((nodes_df, "07_keyword_network_nodes.parquet"), (edges_df, "07_keyword_network_edges.parquet")):
            df_.to_parquet(run.dir / name, index=False)
            run.artifact(run.dir / name, name.split(".")[0][3:], rows=len(df_))
        comm_df.to_csv(run.dir / "07_leiden_communities.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07_leiden_communities.csv", "leiden_communities", rows=len(comm_df))
        st["rows_out"] = len(edges_df)

    # ------------------------------------------------ 문서 군집
    with run.stage("document_clusters") as st:
        km_terms = np.where(np.isin(cls, ["CORE", "EXTENDED", "EMERGING"]))[0]
        Xk = X[:, km_terms].tocsr()
        nkw = np.asarray((Xk > 0).sum(1)).ravel()
        min_kw = rules.get("min_keywords", 2)
        cl_rows = np.where(use & (nkw >= min_kw))[0]       # 군집 입력
        low_rows = np.where(use & (nkw >= 1) & (nkw < min_kw))[0]   # 저신뢰: 가장 가까운 중심에만 배정
        V, idf = tfidf(Xk[cl_rows])
        Vlow = tfidf(Xk[low_rows], idf)[0] if len(low_rows) else None
        wv = half[cl_rows]
        Bcsc = (Xk[cl_rows] > 0).astype(np.float32).tocsc()
        rng = np.random.default_rng(0)
        samp = rng.choice(V.shape[0], min(rules.get("silhouette_sample", 8000), V.shape[0]), replace=False)
        sec_all = corpus.article_sector
        related = cfg["cleaning"].get("related_label", "연관산업")
        core_secs = sorted(s for s in sec_all["sector"].unique() if s != related)
        gid_pos = pd.Series(np.arange(len(cl_rows)), index=arts["gid"].to_numpy()[cl_rows])
        sec_cl = sec_all[sec_all["gid"].isin(gid_pos.index)]
        Ssec = sparse.csr_matrix((np.ones(len(sec_cl)), (sec_cl["gid"].map(gid_pos).to_numpy(),
                                                          sec_cl["sector"].map({s: i for i, s in enumerate(sorted(sec_all["sector"].unique()))}).to_numpy())),
                                 shape=(len(cl_rows), sec_all["sector"].nunique()))
        sec_names = sorted(sec_all["sector"].unique())
        km_seeds = list(range(rules.get("kmeans_seeds", 5)))
        gen_idx = set(np.where(np.isin(np.array(vocab)[km_terms], list(general)))[0])
        ent_local = np.array([etype[t] != "개념" for t in km_terms])
        comp = arts["companies"].to_numpy()[cl_rows]
        probes = rules.get("noise_probes", [])
        km_rows, km_models = [], {}
        for K in rules.get("k_grid", [80, 120, 160, 220]):
            labs = []
            for s in km_seeds:
                km = kmeans(V, wv, K, s)
                labs.append(km.labels_)
                if s == 0:
                    km_models[K] = km
            lab = labs[0]
            sizes = np.bincount(lab, weights=wv, minlength=K)
            cent = km_models[K].cluster_centers_
            top = np.argsort(-cent, axis=1)[:, :10]
            coh = [npmi_coherence(list(t), Bcsc, wv) for t in top]
            relv = np.array([np.average(prel[cl_rows][lab == c], weights=wv[lab == c]) if (lab == c).any() else 0 for c in range(K)])
            gen_c = np.array([sum(t in gen_idx for t in tt) >= 5 for tt in top])
            single = []
            for c in range(K):
                cc = collections.Counter(x for l in comp[lab == c] for x in l)
                single.append(cc.most_common(1)[0][1] / max((lab == c).sum(), 1) >= 0.8 if cc else False)
            single = np.array(single)
            # 분야 포착: 분야별로 그 분야가 과반인 군집 수의 최솟값 (소규모 분야 주제가 사라지지 않는지)
            sw = np.asarray((sparse.csr_matrix((wv, (lab, np.arange(len(lab)))), shape=(K, len(lab))) @ Ssec).todense())
            dom = sw / np.maximum(sw.sum(1, keepdims=True), 1e-9)
            sector_cov = {s: int((dom[:, sec_names.index(s)] >= 0.5).sum()) for s in core_secs}
            stab = np.mean([best_match_jaccard(labs[0], labs[b]).mean() for b in range(1, len(labs))])
            # 잡음 탐침: 탐침어 기사가 저관련·일반어 군집으로 가거나 자기 이름이 상위어인 군집에 모이는지
            probe_res = {}
            for p in probes:
                if p not in vocab:
                    continue
                j = np.where(np.array(vocab)[km_terms] == p)[0]
                if not len(j):
                    continue
                r_ = Xk[cl_rows][:, j[0]].nonzero()[0]
                if not len(r_):
                    continue
                cs = pd.Series(lab[r_]).value_counts()
                c0 = cs.index[0]
                probe_res[p] = {"articles": int(len(r_)), "top_cluster_share": round(float(cs.iloc[0] / len(r_)), 3),
                                "top_cluster_low_relevance": bool(relv[c0] < e2_sum_thr),
                                "probe_in_top_keywords": bool(j[0] in top[c0])}
            km_rows.append({
                "K": K, "silhouette": round(float(silhouette_score(V[samp], lab[samp], metric="cosine")), 4),
                "seed_ari": round(float(np.mean([adjusted_rand_score(labs[a], labs[b]) for a, b in itertools.combinations(range(len(labs)), 2)])), 4),
                "stability_jaccard": round(float(stab), 4),
                "coherence": round(float(np.mean(coh)), 4),
                "largest_share": round(float(sizes.max() / sizes.sum()), 4), "size_gini": round(gini(sizes), 4),
                "tiny_clusters": int((np.bincount(lab, minlength=K) < 10).sum()),
                "low_relevance_share": round(float((relv < e2_sum_thr).mean()), 4),
                "general_word_share": round(float(gen_c.mean()), 4), "single_entity_share": round(float(single.mean()), 4),
                "sector_coverage_min": min(sector_cov.values()), "sector_coverage": json.dumps(sector_cov, ensure_ascii=False),
                "probes": json.dumps(probe_res, ensure_ascii=False),
                "probe_separated": round(float(np.mean([v["top_cluster_share"] >= 0.5 and (v["probe_in_top_keywords"] or v["top_cluster_low_relevance"])
                                                          for v in probe_res.values()])) if probe_res else 0.0, 3),
                "_labs": labs, "_coh": coh, "_relv": relv, "_gen": gen_c, "_single": single, "_dom": dom,
            })
        kg = pd.DataFrame(km_rows)
        kg["borda"] = borda(kg, ["silhouette", "seed_ari", "coherence", "sector_coverage_min", "probe_separated"],
                            ["largest_share", "size_gini", "general_word_share"])
        best = kg.sort_values("borda").iloc[0]
        K = int(best["K"])
        km = km_models[K]
        lab = best["_labs"][0]
        sim = centroid_similarity(V, km.cluster_centers_, lab)
        dc = pd.DataFrame({"gid": arts["gid"].to_numpy()[cl_rows], "date": arts["date"].to_numpy()[cl_rows],
                           "title": arts["title"].to_numpy()[cl_rows], "cluster_id": lab, "similarity": sim.round(4),
                           "weight_half": wv, "p_rel": prel[cl_rows], "low_confidence": False,
                           **{f"k{r['K']}": r["_labs"][0] for r in km_rows}})
        if Vlow is not None:
            ll = km.predict(Vlow)
            simlow = centroid_similarity(Vlow, km.cluster_centers_, ll)
            dc = pd.concat([dc, pd.DataFrame({"gid": arts["gid"].to_numpy()[low_rows], "date": arts["date"].to_numpy()[low_rows],
                                              "title": arts["title"].to_numpy()[low_rows], "cluster_id": ll, "similarity": simlow.round(4),
                                              "weight_half": half[low_rows], "p_rel": prel[low_rows], "low_confidence": True})],
                           ignore_index=True)
        dc["document_cluster_id"] = [f"dc{K}_{c:03d}" for c in dc["cluster_id"]]
        dc.drop(columns=["cluster_id"]).to_parquet(run.dir / "07_document_clusters.parquet", index=False)
        run.artifact(run.dir / "07_document_clusters.parquet", "document_clusters", rows=len(dc))
        st["rows_out"] = len(dc)

    # ------------------------------------------------ 결합 (문서 군집 ↔ Leiden 공동체)
    with run.stage("alignment") as st:
        kw_local = np.array(vocab)[km_terms]
        cent = km.cluster_centers_
        comm_sets = {}
        for layer, nodes in layers.items():
            memb = sel_net[layer]["_memb"]
            for c in sorted(set(memb[memb >= 0])):
                comm_sets[(layer, f"{layer[:3]}_{c:03d}")] = set(np.array(vocab)[nodes[memb == c]])
        term_pos = {t: i for i, t in enumerate(kw_local)}
        al_rows = []
        for c in range(K):
            v = cent[c]
            topn = set(kw_local[np.argsort(-v)[:20]])
            for (layer, cid), terms in comm_sets.items():
                ix = [term_pos[t] for t in terms if t in term_pos]
                mass = float(v[ix].sum() / max(v.sum(), 1e-9)) if ix else 0.0
                cos = float(v[ix].sum() / (np.linalg.norm(v) * np.sqrt(len(terms)) + 1e-9)) if ix else 0.0
                jac = len(topn & terms) / len(topn | terms)
                if mass > 0:
                    al_rows.append({"document_cluster_id": f"dc{K}_{c:03d}", "layer": layer, "community_id": cid,
                                    "mass_share": round(mass, 4), "cosine": round(cos, 4), "jaccard_top20": round(jac, 4)})
        al = pd.DataFrame(al_rows)
        al["rank"] = al.groupby(["document_cluster_id", "layer"])["mass_share"].rank(ascending=False, method="first")
        bestm = al[al["rank"] == 1].set_index(["document_cluster_id", "layer"])["mass_share"]
        al["best_share"] = [bestm.get((d, l), 0) for d, l in zip(al["document_cluster_id"], al["layer"])]
        sub_ratio = rules.get("subtopic_ratio", 0.5)
        al["matched"] = (al["rank"] == 1) | (al["mass_share"] >= sub_ratio * al["best_share"])
        al.to_csv(run.dir / "07_topic_alignment.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07_topic_alignment.csv", "topic_alignment", rows=len(al))
        # 대응 약함: 개체 확장 층 최대 질량 점유가 하위 25% (분위수 기준)
        ext_best = al[(al["layer"] == "extended") & (al["rank"] == 1)].set_index("document_cluster_id")["mass_share"]
        weak_cut = float(ext_best.quantile(0.25)) if len(ext_best) else 0.0
        st["rows_out"] = len(al)

    # ------------------------------------------------ 주제 등록부
    with run.stage("topic_registry") as st:
        kgb = kg.set_index("K").loc[K]
        stab_c = np.mean([best_match_jaccard(kgb["_labs"][0], kgb["_labs"][b]).reindex(range(K)).fillna(0).to_numpy()
                          for b in range(1, len(kgb["_labs"]))], axis=0)
        comp_all = arts.set_index("gid")["companies"]
        sec_by_gid = sec_all.groupby("gid")["sector"].agg(list)
        reg, pack = [], []
        other_K = [r["K"] for r in km_rows if r["K"] != K]
        for c in range(K):
            cid = f"dc{K}_{c:03d}"
            m = dc[(dc["document_cluster_id"] == cid) & ~dc["low_confidence"]]
            v = cent[c]
            order = np.argsort(-v)
            top_kw = [kw_local[i] for i in order[:40] if not ent_local[i]][:15]
            top_ent = [kw_local[i] for i in order[:60] if ent_local[i]][:8]
            cc = collections.Counter(x for g in m["gid"] for x in comp_all.get(g, []))
            ent_conc = cc.most_common(1)[0][1] / max(len(m), 1) if cc else 0.0
            sc = collections.Counter(x for g in m["gid"] for x in sec_by_gid.get(g, []))
            tot = sum(sc.values()) or 1
            reps = m.sort_values("similarity", ascending=False).head(5)
            bnd = m.sort_values("similarity").head(3)
            a_ext = al[(al["document_cluster_id"] == cid) & (al["layer"] == "extended") & al["matched"]].sort_values("rank")
            a_con = al[(al["document_cluster_id"] == cid) & (al["layer"] == "concept") & al["matched"]].sort_values("rank")
            relv = float(kgb["_relv"][c])
            noise = [x for x, f in (("low_relevance", relv < e2_sum_thr), ("general_word", bool(kgb["_gen"][c]))) if f]
            best_ext = float(a_ext["mass_share"].iloc[0]) if len(a_ext) else 0.0
            row = {
                "topic_id": f"tp_{c:03d}", "document_cluster_id": cid,
                "matched_leiden_community_ids": ";".join(a_ext["community_id"]),
                "matched_concept_community_ids": ";".join(a_con["community_id"]),
                "subtopic_candidate": int(len(a_ext) > 1),
                "top_keywords": ", ".join(top_kw), "top_entities": ", ".join(top_ent),
                "top_company": cc.most_common(1)[0][0] if cc else "",
                "representative_articles": " || ".join(f"{g} {t}" for g, t in zip(reps["gid"], reps["title"])),
                "article_count": int(len(m)), "low_confidence_articles": int(((dc["document_cluster_id"] == cid) & dc["low_confidence"]).sum()),
                "weighted_article_count": round(float(m["weight_half"].sum()), 1),
                "sector_distribution": json.dumps({k: round(x / tot, 3) for k, x in sc.most_common()}, ensure_ascii=False),
                "entity_concentration": round(ent_conc, 3), "single_entity": bool(kgb["_single"][c]),
                "topic_relevance_score": round(relv, 3), "coherence": round(float(kgb["_coh"][c]), 4),
                "stability": round(float(stab_c[c]), 3), "noise_flag": ";".join(noise),
                "alignment_best_share": round(best_ext, 4), "topic_review_flag": int(best_ext < weak_cut or not len(a_ext)),
                "first_month": str(pd.Timestamp(m["date"].min()).to_period("M")) if len(m) else None,
                "last_month": str(pd.Timestamp(m["date"].max()).to_period("M")) if len(m) else None,
                "run_id": run.run_id,
            }
            reg.append(row)
            # 모델 간 분할: 이 군집 기사가 다른 K에서 어떻게 나뉘는지
            splits = {}
            for k2 in other_K:
                vc = m[f"k{k2}"].value_counts(normalize=True).head(3)
                splits[f"K{k2}"] = [{"cluster": f"dc{k2}_{int(i):03d}", "share": round(float(s), 3)} for i, s in vc.items()]
            pack.append({"type": "cluster", "model": f"kmeans_K{K}", **{k: row[k] for k in (
                "topic_id", "document_cluster_id", "article_count", "weighted_article_count", "top_keywords", "top_entities",
                "sector_distribution", "entity_concentration", "topic_relevance_score", "coherence", "stability", "noise_flag",
                "matched_leiden_community_ids", "subtopic_candidate", "topic_review_flag")},
                "representative_articles": [{"gid": g, "title": t, "date": str(pd.Timestamp(d).date())} for g, t, d in zip(reps["gid"], reps["title"], reps["date"])],
                "boundary_articles": [{"gid": g, "title": t, "similarity": float(s)} for g, t, s in zip(bnd["gid"], bnd["title"], bnd["similarity"])],
                "split_in_other_models": splits})
        tr = pd.DataFrame(reg)
        tr.to_csv(run.dir / "07_topic_registry.csv", index=False, encoding="utf-8-sig")
        run.artifact(run.dir / "07_topic_registry.csv", "topic_registry", rows=len(tr))
        st["rows_out"] = len(tr)

    # ------------------------------------------------ 비교 자료·보고서
    pub = lambda df: df[[c for c in df.columns if not c.startswith("_")]]
    for r in pub(ng).to_dict("records"):
        pack.insert(0, {"type": "model", "model": f"leiden_{r['layer']}_{r['unit']}_co{r['min_co']}_npmi{r['npmi']}_k{r['topk']}_res{r['resolution']}", **r})
    for r in pub(kg).to_dict("records"):
        pack.insert(0, {"type": "model", "model": f"kmeans_K{r['K']}", **r})
    for c in comm_df.to_dict("records"):
        pack.append({"type": "community", **c})
    with open(run.dir / "e4_comparison_pack.jsonl", "w", encoding="utf-8") as f:
        for p in pack:
            f.write(json.dumps(p, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    run.artifact(run.dir / "e4_comparison_pack.jsonl", "comparison_pack", rows=len(pack))
    with pd.ExcelWriter(run.dir / "e4_topic_report.xlsx") as xw:
        tr.to_excel(xw, sheet_name="topics", index=False)
        pub(kg).to_excel(xw, sheet_name="kmeans_grid", index=False)
        pub(ng).sort_values(["layer", "borda"]).to_excel(xw, sheet_name="leiden_grid", index=False)
        comm_df.to_excel(xw, sheet_name="communities", index=False)
        al[al["matched"]].to_excel(xw, sheet_name="alignment", index=False)
        pd.DataFrame({"general_word": sorted(general)}).to_excel(xw, sheet_name="general_words", index=False)
    run.artifact(run.dir / "e4_topic_report.xlsx", "e4_topic_report")

    sel = {l: {k: (v.item() if hasattr(v, "item") else v) for k, v in sel_net[l].items() if not k.startswith("_")} for l in layers}
    summary = {
        "e3_run_id": e3_run, "articles_clustered": int(len(cl_rows)), "articles_low_confidence": int(len(low_rows)),
        "articles_no_keyword": int((use & (nkw == 0)).sum()),
        "general_words": sorted(general),
        "network_nodes": {l: int(len(n)) for l, n in layers.items()},
        "network_selected": sel,
        "network_strong_edge_rate": {l: round(float(edges_df[edges_df["layer"] == l]["strong"].mean()), 4) for l in layers},
        "kmeans_grid": pub(kg).drop(columns=["probes", "sector_coverage"]).to_dict("records"),
        "kmeans_selected_K": K, "probes_selected": json.loads(kgb["probes"]),
        "topics": len(tr), "topic_review_flag": int(tr["topic_review_flag"].sum()),
        "subtopic_candidates": int(tr["subtopic_candidate"].sum()),
        "noise_topics": int((tr["noise_flag"] != "").sum()), "single_entity_topics": int(tr["single_entity"].sum()),
        "selection_rule": "Borda 순위 평균 (자동 제안, 최종 선택은 연구자 기준·decision_log)",
    }
    (run.dir / "e4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    append_jsonl(REGISTRY / "experiment_registry.jsonl", {
        "experiment_id": "exp_topic_v2_e4", "run_id": run.run_id, "task": "topic_clusters",
        "data_snapshot_id": snap["data_snapshot_id"], "input_e3_run_id": e3_run,
        "models": ["NPMI+Leiden (concept/extended × article/sentence grid)", "TF-IDF+MiniBatchKMeans K grid"],
        "result": {"kmeans_selected_K": K, "network_selected": sel, "topics": len(tr)}})
    run.finish()
    return summary
