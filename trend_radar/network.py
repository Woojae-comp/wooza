"""연관어 네트워크, 군집, 거리 (설계 7~9장).

- 엣지: 같은 기사에 함께 등장. 동시등장 기사 수, Jaccard, PMI, NPMI.
- 군집: Louvain (가중치는 Jaccard). 군집 이름은 확정하지 않고 후보만 만든다.
- 거리: 기사 × 키워드 이진 행렬의 코사인 유사도. 화면 배치 거리가 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from .text import DocTerm


@dataclass
class Network:
    label: str
    n_docs: int
    words: list[str]
    df: np.ndarray                 # 구간 내 기사 수
    co: np.ndarray                 # 동시등장 기사 수 (words × words)
    cos: np.ndarray                # 코사인 유사도
    edges: pd.DataFrame
    graph: nx.Graph
    clusters: list[dict] = field(default_factory=list)
    membership: dict[str, int] = field(default_factory=dict)
    centrality: pd.DataFrame | None = None


def cooccurrence(dt: DocTerm, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = dt.X[rows][:, cols]
    co = np.asarray((Xs.T @ Xs).todense())
    df = np.diag(co).copy()
    return df, co


def cosine_from_co(df: np.ndarray, co: np.ndarray) -> np.ndarray:
    """이진 벡터의 코사인 = 동시등장 / sqrt(df_a × df_b)."""
    d = np.sqrt(np.maximum(df, 1))
    return co / np.outer(d, d)


def build_network(dt: DocTerm, rows: np.ndarray, label: str, cfg: dict,
                  words: list[str] | None = None, exclude: set[str] | None = None) -> Network:
    nc = cfg["network"]
    n = len(rows)
    col_df = np.asarray(dt.X[rows].sum(axis=0)).ravel()
    if words is None:
        order = np.argsort(-col_df, kind="stable")
        cand = [j for j in order if col_df[j] >= nc["min_cooccurrence"] and dt.vocab[j] not in (exclude or ())]
        cols = np.array(cand[: nc["top_n"]], dtype=int)
    else:
        cols = np.array([dt.index[w] for w in words if w in dt.index], dtype=int)
    wl = [dt.vocab[j] for j in cols]
    df, co = cooccurrence(dt, rows, cols)
    cos = cosine_from_co(df, co)

    iu, ju = np.triu_indices(len(wl), k=1)
    c = co[iu, ju]
    keep = c >= nc["min_cooccurrence"]
    iu, ju, c = iu[keep], ju[keep], c[keep]
    da, db = df[iu], df[ju]
    jac = c / (da + db - c)
    pmi = np.log((c * n) / (da * db))
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = np.where(c < n, pmi / -np.log(c / n), 1.0)
    edges = pd.DataFrame({"a": [wl[i] for i in iu], "b": [wl[j] for j in ju], "co": c.astype(int),
                          "jaccard": jac, "pmi": pmi, "npmi": npmi, "cosine": cos[iu, ju]})
    edges = edges[edges["pmi"] > 0]
    # 허브 편향을 줄이기 위해 노드마다 Jaccard 상위 k개 엣지만 남긴다
    k = nc["max_edges_per_node"]
    # 각 엣지를 양 끝 노드 입장에서 순위를 매겨 어느 한쪽에서라도 상위 k면 남긴다
    both = pd.concat([edges[["a", "jaccard"]].rename(columns={"a": "n"}).assign(e=edges.index),
                      edges[["b", "jaccard"]].rename(columns={"b": "n"}).assign(e=edges.index)])
    both["r"] = both.groupby("n")["jaccard"].rank(ascending=False, method="first")
    keep_e = set(both.loc[both["r"] <= k, "e"])
    edges = edges.loc[sorted(keep_e)].reset_index(drop=True)

    g = nx.Graph()
    for w, d in zip(wl, df):
        g.add_node(w, df=int(d))
    for r in edges.itertuples():
        g.add_edge(r.a, r.b, weight=r.jaccard, co=r.co, npmi=r.npmi)
    net = Network(label, n, wl, df, co, cos, edges, g)
    _cluster(net, nc)
    _centrality(net)
    return net


def _cluster(net: Network, nc: dict) -> None:
    g = net.graph
    if g.number_of_edges() == 0:
        return
    comms = nx.community.louvain_communities(g, weight="weight", resolution=nc.get("resolution", 1.0),
                                             seed=nc.get("seed", 42))
    dfm = dict(zip(net.words, net.df))
    comms = [c for c in comms if len(c) >= 3]
    comms.sort(key=lambda c: -sum(dfm[w] for w in c))
    for cid, comm in enumerate(comms):
        sub = g.subgraph(comm)
        strength = dict(sub.degree(weight="weight"))
        central = sorted(comm, key=lambda w: (-strength.get(w, 0), -dfm[w]))
        top_edges = sorted(sub.edges(data=True), key=lambda e: -e[2]["weight"])[:8]
        net.clusters.append({
            "id": cid,
            "keywords": sorted(comm, key=lambda w: -dfm[w]),
            "central": central[:5],
            "size": len(comm),
            "name_candidate": " · ".join(central[:3]),
            "top_edges": [(a, b, round(d["weight"], 3), int(d["co"])) for a, b, d in top_edges],
        })
        for w in comm:
            net.membership[w] = cid


def _centrality(net: Network) -> None:
    g = net.graph
    if g.number_of_nodes() == 0:
        return
    deg = nx.degree_centrality(g)
    strength = dict(g.degree(weight="weight"))
    k = min(200, g.number_of_nodes())
    btw = nx.betweenness_centrality(g, k=k, seed=0, weight=None) if g.number_of_edges() else {}
    net.centrality = pd.DataFrame({
        "keyword": net.words,
        "degree": [g.degree(w) for w in net.words],
        "degree_centrality": [deg.get(w, 0.0) for w in net.words],
        "strength": [strength.get(w, 0.0) for w in net.words],
        "betweenness": [btw.get(w, 0.0) for w in net.words],
        "cluster": [net.membership.get(w, -1) for w in net.words],
    })


def cluster_articles(dt: DocTerm, rows: np.ndarray, keywords: list[str], min_hits: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """군집 관련 기사: 군집 키워드가 min_hits개 이상 등장한 기사. (기사 행 번호, 적중 수)"""
    cols = [dt.index[w] for w in keywords if w in dt.index]
    hits = np.asarray(dt.X[rows][:, cols].sum(axis=1)).ravel()
    need = min(min_hits, len(cols))
    sel = hits >= need
    return rows[sel], hits[sel]


def match_clusters(prev: Network, cur: Network, min_jaccard: float = 0.2) -> list[dict]:
    """이전 구간 군집과 현재 구간 군집을 키워드 집합 Jaccard로 대응시킨다."""
    out = []
    used = set()
    for c in cur.clusters:
        s = set(c["keywords"])
        best, bj = None, 0.0
        for p in prev.clusters:
            ps = set(p["keywords"])
            j = len(s & ps) / len(s | ps)
            if j > bj:
                best, bj = p, j
        if best is not None and bj >= min_jaccard:
            used.add(best["id"])
            ps = set(best["keywords"])
            out.append({"cluster": c["id"], "status": "continued", "prev_cluster": best["id"],
                        "similarity": round(bj, 3), "added": sorted(s - ps, key=c["keywords"].index)[:10],
                        "removed": sorted(ps - s, key=best["keywords"].index)[:10],
                        "size_change": c["size"] - best["size"]})
        else:
            out.append({"cluster": c["id"], "status": "new", "prev_cluster": None,
                        "similarity": round(bj, 3), "added": c["keywords"][:10], "removed": [], "size_change": c["size"]})
    for p in prev.clusters:
        if p["id"] not in used:
            out.append({"cluster": None, "status": "disappeared", "prev_cluster": p["id"], "similarity": 0.0,
                        "added": [], "removed": p["keywords"][:10], "size_change": -p["size"]})
    return out
