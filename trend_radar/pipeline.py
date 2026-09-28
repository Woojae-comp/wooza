"""전체 흐름 (설계 2장): 원자료 → 정제 → 키워드 → 지표 → 네트워크·군집 → 거리 → 분야 → 시간 변화 → 트렌드."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import candidates as cand
from .config import Lexicon
from .layers import classify
from .load import Corpus, build_corpus, read_raw
from .metrics import add_periods, keyword_table, long_series
from .network import build_network, cluster_articles, match_clusters
from .text import DocTerm, extract_keywords, majority_tags, prepped_texts, space_joined_names, token_form, tokenize_corpus
from .trends import (STATUS_ORDER, add_half, arrow, bridge_keywords, diffusion, keyword_pair_convergence,
                     make_windows, quarterly_networks, sector_convergence, spread_path, status_frame, structural)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ArticleStore:
    """결과에서 참조하는 대표 기사. 같은 기사를 여러 번 싣지 않도록 행 번호로 참조한다."""

    def __init__(self, arts: pd.DataFrame):
        self.arts = arts
        self.used: dict[int, dict] = {}
        self.dates = arts["date"].to_numpy().astype("datetime64[D]").astype("int64")
        self.titles = arts["title"].astype(object).to_numpy()

    def ref(self, r: int) -> int:
        r = int(r)
        if r not in self.used:
            a = self.arts.iloc[r]
            self.used[r] = {"d": str(a["date"].date()), "t": a["title"], "s": a["summary"][:160],
                            "p": a["press"], "u": a["link"], "c": a["companies"], "f": a["sectors"]}
        return r

    def pick(self, rows, n: int, score: np.ndarray | None = None, recent_first: bool = True) -> list[int]:
        rows = np.asarray(rows)
        if len(rows) == 0:
            return []
        dates = self.dates[rows]
        key = -dates if recent_first else dates
        order = np.lexsort((key, -score)) if score is not None else np.argsort(key, kind="stable")
        out, seen = [], set()
        titles = self.titles
        for i in order:
            t = titles[rows[i]][:18]
            if t in seen:
                continue
            seen.add(t)
            out.append(self.ref(rows[i]))
            if len(out) >= n:
                break
        return out


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def evidence(p: dict) -> list[str]:
    """트렌드 후보 탐지 근거 (설계 19장). 사람이 읽는 문장."""
    ev = []
    ev.append(f"최근 {p['recent_label']} 기사 비중 {_pct(p['recent_share'])}, 직전 {p['base_label']} {_pct(p['base_share'])} "
              f"({p['ratio']:.1f}배)")
    if p["trend6q"] >= 0.6:
        ev.append(f"최근 6분기 비중 증가 추세 (순위상관 {p['trend6q']:.2f})")
    elif p["trend6q"] <= -0.6:
        ev.append(f"최근 6분기 비중 감소 추세 (순위상관 {p['trend6q']:.2f})")
    if p["growth_months"] >= 6:
        ev.append(f"증가 지속기간 {p['growth_months']}개월")
    d = p["diffusion"]
    ev.append(f"{d['n_sectors']}개 분야 중 {d['recent']}개 분야에서 유의미 (첫 12개월 {d['first']}개)")
    up = [s for s, a in d["arrows"].items() if a == "↑↑"]
    if up:
        ev.append(f"{'·'.join(up)} 분야에서 크게 증가 (↑↑)")
    if p.get("peak") and p["peak"]["spike"]:
        pk = p["peak"]
        s = f"{pk['month']} 월 비중이 평소의 {pk['multiple']:.1f}배로 급증 (전후 3개월 기사의 {pk['concentration'] * 100:.0f}%가 이 달)"
        if pk.get("top_company"):
            s += f", 해당 월 기사의 {pk['top_company_share'] * 100:.0f}%가 '{pk['top_company']}' 관련"
        ev.append(s)
    co = p.get("company") or {}
    if co.get("specific"):
        ev.append(f"최근 기사의 {co['share'] * 100:.0f}%가 '{co['top']}' 검색 기사 (기업 한정 이슈 가능성)")
    st = p.get("structure")
    if st:
        if st.get("converging_links"):
            ev.append("새로 가까워진 이슈: " + ", ".join(w for w, *_ in st["converging_links"][:6]))
        if st.get("assoc_overlap") is not None and st["assoc_overlap"] <= 0.35:
            ev.append(f"연관어 TOP20 겹침 {st['assoc_overlap']:.2f} (연결 구조 변화)")
        if st.get("cluster_moved"):
            ev.append(f"군집 이동: [{st['early_cluster']}] → [{st['recent_cluster']}]")
    return ev


def run(cfg: dict, input_paths: list[str] | None = None, lexicon_path: str | None = None,
        out_dir: str | None = None, with_candidates: bool = True, raw: pd.DataFrame | None = None) -> dict:
    out = Path(out_dir or cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    lex = Lexicon.load(lexicon_path)

    log("원자료 로드")
    if raw is None:
        raw = read_raw(input_paths or cfg["input"]["paths"])
    corpus = build_corpus(raw, cfg)
    arts = add_half(add_periods(corpus.articles))
    log(f"기사 {len(arts):,}건 ({corpus.report['date_min']} ~ {corpus.report['date_max']})")

    aliases = cfg["cleaning"].get("company_aliases") or {}
    companies = sorted(corpus.article_company["company"].unique())
    names = space_joined_names(companies, aliases)
    texts = prepped_texts(arts, names)
    log("형태소 분석 (캐시 사용)")
    tokens = tokenize_corpus(texts, names, out / "cache")
    kw = extract_keywords(texts, tokens, lex, cfg)

    lc = cfg.get("layers") or {}
    if not lc.get("enabled"):
        res = analyze(cfg, corpus, arts, texts, tokens, kw, lex, lexicon_path, names, out, with_candidates)
        _finish({"all": res}, out)
        return res

    log("층 분류 (콘텐츠·사업 / 자본시장)")
    lab, kwt = classify(arts, kw, cfg)
    companies = sorted(corpus.article_company["company"].unique())
    co_words = {token_form(n) for n in names} | set(companies)
    # 콘텐츠 층에서는 시장 어휘를 뺀다: 시장 씨앗 + 시장 성향이 강한 키워드 (기업명 제외)
    drop = set(lc["market_seed"]) | {w for w, v in zip(kwt["keyword"], kwt["weight"])
                                     if v <= lc["content_drop_market_weight"] and w not in co_words}
    # 흔한 일반명사(NNG)로 콘텐츠 성향이 아닌 것도 뺀다 (확대·반영·가치·흐름 …).
    # 고유명사(작품·서비스·인물·기업), 영문 용어(AI 등), 드물고 구체적인 단어(수익화)는 남는다.
    tag = majority_tags(tokens)
    content_seed = set(lc["content_seed"])
    generic = {w for w, n, v in zip(kwt["keyword"], kwt["articles"], kwt["weight"])
               if tag.get(w, "NNG") == "NNG" and v < lc["keyword_threshold"] and n >= lc["content_generic_min_df"]
               and w not in content_seed and w not in co_words}
    drop |= generic
    extra = set(lc.get("content_extra_stopwords") or [])
    suffixes = tuple(lc.get("content_drop_suffixes") or ())
    drop |= extra | {w for w in kwt["keyword"] if suffixes and w.endswith(suffixes)}
    review = out / "review"
    review.mkdir(exist_ok=True)
    with pd.ExcelWriter(review / "layers.xlsx") as xw:
        kwt.to_excel(xw, sheet_name="키워드 층 성향", index=False)
        pd.DataFrame({"keyword": sorted(drop), "reason": ["흔한 일반명사" if w in generic else "추가 제외어" if w in extra
                                                  else "시장 어휘" for w in sorted(drop)]}
                     ).to_excel(xw, sheet_name="콘텐츠 층 제외어", index=False)
        lab.assign(title=arts["title"].to_numpy(), date=arts["date"].dt.date.to_numpy()).to_excel(
            xw, sheet_name="기사 층", index=False)
    layer_report = {"articles": int(len(arts)), "content": int(lab["content"].sum()), "market": int(lab["market"].sum()),
                    "both": int((lab["content"] & lab["market"]).sum()),
                    "neither": int((~lab["content"] & ~lab["market"]).sum()),
                    "content_weak": int((lab["content"] & ~lab["strong"]).sum()),
                    "seed_content": kwt.attrs.get("seed_content"), "seed_market": kwt.attrs.get("seed_market"),
                    "content_dropped_keywords": len(drop)}
    log(f"콘텐츠 층 {layer_report['content']:,}건 · 자본시장 층 {layer_report['market']:,}건 · 둘 다 {layer_report['both']:,}건")
    results = {}
    for key, label in (("content", "콘텐츠·사업"), ("market", "자본시장")):
        idx = np.flatnonzero(lab[key].to_numpy())
        sub = arts.iloc[idx].reset_index(drop=True)
        g = set(sub["gid"])
        sub_corpus = Corpus(sub, corpus.article_sector[corpus.article_sector["gid"].isin(g)].reset_index(drop=True),
                            corpus.article_company[corpus.article_company["gid"].isin(g)].reset_index(drop=True),
                            dict(corpus.report))
        sub_kw = [kw[i] for i in idx]
        if key == "content":
            sub_kw = [[w for w in ws if w not in drop] for ws in sub_kw]
        log(f"[{label}] 분석")
        res = analyze(cfg, sub_corpus, sub, [texts[i] for i in idx], [tokens[i] for i in idx], sub_kw, lex,
                      lexicon_path, names, out / key, with_candidates)
        res["meta"]["layer"] = {"key": key, "label": label, **layer_report}
        with open(out / key / "radar.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, default=_jsonable)
        results[key] = res
    _finish(results, out)
    return results


def _finish(results: dict, out: Path) -> None:
    from .report import write_html
    write_html(results, out / "radar.html")
    log(f"완료 → {out / 'radar.html'}")


def analyze(cfg: dict, corpus: Corpus, arts: pd.DataFrame, texts: list[str], tokens: list, kw: list[list[str]],
            lex: Lexicon, lexicon_path: str | None, names: list[str], out: Path, with_candidates: bool) -> dict:
    """한 층(또는 전체)의 기사에 대해 지표·네트워크·트렌드를 계산한다."""
    out.mkdir(parents=True, exist_ok=True)
    companies = sorted(corpus.article_company["company"].unique())
    # 핵심 분야 기사가 앞, 연관산업 기사가 뒤에 오도록 정렬한다 (행 번호가 두 범위에서 같도록).
    # 전체 트렌드는 핵심 분야 기사로만, 연관산업은 분야별·융합에서 한 단계 낮은 분야로만 본다.
    related = cfg["cleaning"].get("related_label", "연관산업")
    if "tier" in arts:
        order = np.argsort((arts["tier"] != "core").to_numpy(), kind="stable")
        arts = arts.iloc[order].reset_index(drop=True)
        kw = [kw[i] for i in order]
        texts = [texts[i] for i in order]
        tokens = [tokens[i] for i in order]
        n_core = int((arts["tier"] == "core").sum())
    else:
        n_core = len(arts)
    arts_all, kw_all = arts, kw
    arts = arts_all.iloc[:n_core]
    dt = DocTerm(kw_all[:n_core], cfg["keywords"]["min_df"])
    dt_all = DocTerm(kw_all, cfg["keywords"]["min_df"], vocab=dt.vocab) if n_core < len(arts_all) else dt
    log(f"기사 {n_core:,}건 (연관산업 {len(arts_all) - n_core:,}건 별도) · 어휘 {len(dt.vocab):,}개")

    ktab, series = keyword_table(dt, arts)
    win = make_windows(arts, cfg)
    gid_row = pd.Series(np.arange(len(arts_all)), index=arts_all["gid"])
    sec_map = corpus.article_sector.assign(row=lambda d: d["gid"].map(gid_row)).dropna()
    sector_rows = {s: np.sort(g["row"].astype(int).to_numpy())
                   for s, g in sorted(sec_map.groupby("sector"), key=lambda x: (x[0] == related, -len(x[1])))}
    core_sector_rows = {k: v[v < n_core] for k, v in sector_rows.items() if k != related}
    sectors = list(core_sector_rows)
    all_rows = np.arange(len(arts))
    ci = {c: i for i, c in enumerate(companies)}
    cr_, cc_ = [], []
    for r, cs in enumerate(arts_all["companies"]):
        for c in cs:
            cr_.append(r)
            cc_.append(ci[c])
    company_X = sparse.csr_matrix((np.ones(len(cr_)), (cr_, cc_)), shape=(len(arts_all), len(companies)))
    # 기업명 키워드: 분야 소속이 기업 매핑에서 오므로 분야 간 거리 계산에서는 뺀다
    co_words = {token_form(n) for n in names} | set(companies)
    co_mask = np.array([w in co_words for w in dt.vocab])

    log("상태 분류 (전체)")
    st = status_frame(dt, arts, all_rows, win, cfg, company_X, companies)
    dif = diffusion(dt, arts, core_sector_rows, win, cfg)
    n_sig_recent = dif["sig_recent"].sum(1)
    n_sig_first = dif["sig_first"].sum(1)
    st["sectors_recent"] = n_sig_recent
    st["sectors_first"] = n_sig_first
    st["Spreading"] = (n_sig_recent - n_sig_first >= cfg["status"]["spreading_min_sector_gain"]) & \
                      (n_sig_recent >= 3) & ~st["Declining"] & (st["ratio"] >= 1.0)

    log("네트워크·군집")
    first_rows = win.rows(arts, "month", win.first12)
    prev_rows = win.rows(arts, "month", win.prev12)
    recent_rows = win.rows(arts, "month", win.recent12)
    net_all = build_network(dt, all_rows, "전체 기간", cfg)
    net_first = build_network(dt, first_rows, f"{win.first12[0]}~{win.first12[-1]}", cfg)
    net_prev = build_network(dt, prev_rows, f"{win.prev12[0]}~{win.prev12[-1]}", cfg)
    net_recent = build_network(dt, recent_rows, f"{win.recent12[0]}~{win.recent12[-1]}", cfg)

    # 네트워크 연결 감소가 함께 있어야 Declining (설계 13장)
    dr = net_recent.centrality.set_index("keyword")["degree"] if net_recent.centrality is not None else pd.Series(dtype=float)
    de = net_first.centrality.set_index("keyword")["degree"] if net_first.centrality is not None else pd.Series(dtype=float)
    deg_drop = st["keyword"].map(lambda w: dr.get(w, 0) < de.get(w, 0) if w in de.index else True)
    st["Declining"] = st["Declining"] & deg_drop

    log("구조 변화·융합")
    top_n = cfg["output"]["profile_top_n"]
    flagged = st.loc[st[STATUS_ORDER].any(axis=1) & (st["articles"] >= 30), "keyword"].tolist()
    prof_words = list(dict.fromkeys(dt.vocab[:top_n] + flagged))
    stru = structural(dt, first_rows, recent_rows, prof_words, net_first, net_recent, cfg).set_index("keyword")
    conv_ok = (stru["n_converging"] >= cfg["status"]["converging_min_new_links"]) & \
              (stru["early_df"] >= cfg["status"].get("converging_min_early_df", 30))
    st = st.set_index("keyword")
    st["Converging"] = False
    st.loc[stru.index[conv_ok], "Converging"] = True
    st["Converging"] &= (st["ratio"] >= 1.0)
    st = st.reset_index()

    pair_words = [dt.vocab[j] for j in np.argsort(-np.asarray(dt.X[recent_rows].sum(axis=0)).ravel())[:1500]]
    pairs = keyword_pair_convergence(dt, first_rows, recent_rows, pair_words, net_first)

    art_cs: list[dict[str, set]] = [dict() for _ in range(len(arts_all))]
    for r in corpus.article_company.itertuples():
        i = gid_row.get(r.gid)
        if i is not None:
            art_cs[i][r.company] = set(r.sectors)
    # 분야 간 거리에는 연관산업도 넣는다 (핵심 분야 × 연관산업 수렴)
    sconv_year = sector_convergence(dt_all, arts_all, sector_rows, art_cs, win.years, "year", exclude=co_mask)
    sconv_half = sector_convergence(dt_all, arts_all, sector_rows, art_cs, win.halves, "half", exclude=co_mask)

    log("분기별 군집 추적")
    qnets = quarterly_networks(dt, arts, win, cfg, exclude=set())

    store = ArticleStore(arts_all)
    n_list = cfg["output"]["articles_per_list"]
    months = win.months
    ms = series["month"]
    mi = {m: i for i, m in enumerate(ms["periods"])}
    comp_by_row = arts["companies"].to_numpy()
    month_arr = arts["month"].to_numpy()
    recent_mask = np.zeros(len(arts), dtype=bool)
    recent_mask[recent_rows] = True
    stx = st.set_index("keyword")
    ktx = ktab.set_index("keyword")

    log("트렌드 프로파일")
    profiles = {}
    for w in prof_words:
        j = dt.index[w]
        s = stx.loc[w]
        rows = dt.docs_with(w)
        statuses = [k for k in STATUS_ORDER if bool(s[k])]
        arrows = {sec: arrow(float(dif["ratio"][j, i]), bool(dif["sig_recent"][j, i]), bool(dif["sig_prev"][j, i]))
                  for i, sec in enumerate(sectors)}
        peak = {"month": s["peak_month"], "multiple": float(s["peak_multiple"]), "concentration": float(s["peak_concentration"]),
                "df": int(s["peak_df"]), "spike": bool(s["event_spike"]), "ongoing": bool(s["event_ongoing"])}
        pr = rows[month_arr[rows] == s["peak_month"]]
        if peak["spike"] and len(pr):
            cc = Counter(c for r in pr for c in comp_by_row[r])
            top_c, n_c = cc.most_common(1)[0]
            peak["top_company"], peak["top_company_share"] = top_c, n_c / len(pr)
        struct = None
        if w in stru.index:
            x = stru.loc[w]
            struct = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in x.items()}
            struct = json.loads(json.dumps(struct, default=_jsonable))
        # 대표 기사: 최근 12개월 중 최근 연관어를 많이 포함한 기사
        assoc = [dt.index[a] for a, _ in (struct or {}).get("recent_assoc", [])[:10] if a in dt.index]
        rr = rows[recent_mask[rows]]
        if len(rr) == 0:
            rr = rows
        score = np.asarray(dt.X[rr][:, assoc].sum(axis=1)).ravel() if assoc else None
        p = {
            "k": w, "articles": int(dt.df[j]), "share": float(ktx.loc[w, "share"]), "occurrences": int(dt.tf[j]),
            "first_seen": str(ktx.loc[w, "first_seen"]), "last_seen": str(ktx.loc[w, "last_seen"]),
            "qoq": _f(ktx.loc[w].get("qoq_share_change")), "yoy": _f(ktx.loc[w].get("yoy_share_change")),
            "status": statuses,
            "company": {"top": s["top_company"], "share": float(s["top_company_share"]), "n": int(s["n_companies"]),
                        "specific": bool(s["company_specific"])},
            "recent_share": float(s["recent_share"]), "base_share": float(s["base_share"]), "ratio": float(s["ratio"]),
            "recent_label": f"{win.recent_q[0]}~{win.recent_q[-1]}", "base_label": f"{win.base_q[0]}~{win.base_q[-1]}",
            "trend6q": float(s["trend6q"]), "growth_months": int(s["growth_months"]), "cv": float(s["cv"]),
            "m_share": [round(float(ms["shares"][j, mi[m]]), 6) if m in mi else 0 for m in months],
            "m_count": [int(ms["counts"][j, mi[m]]) if m in mi else 0 for m in months],
            "peak": peak,
            "diffusion": {"n_sectors": len(sectors), "recent": int(n_sig_recent[j]), "first": int(n_sig_first[j]),
                          "arrows": arrows,
                          "recent_share": {sec: round(float(dif["recent_df"][j, i] / max(dif["recent_tot"][i], 1)), 5)
                                           for i, sec in enumerate(sectors)},
                          "path": spread_path(dif, j)},
            "structure": struct,
            "arts": {"rep": store.pick(rr, n_list, score), "latest": store.pick(rows, 3),
                     "spike": store.pick(pr, 3) if peak["spike"] else []},
        }
        p["evidence"] = evidence(p)
        profiles[w] = p

    log("분야별 트렌드")
    # 분야별은 연관산업까지 전체 행으로 본다 (핵심 행 번호는 그대로)
    core_dt, core_arts = dt, arts
    dt, arts = dt_all, arts_all
    recent_mask = np.zeros(len(arts_all), dtype=bool)
    recent_mask[recent_rows] = True
    recent_mask[n_core:] = arts_all["month"].iloc[n_core:].isin(win.recent12).to_numpy()
    sector_out = {}
    excl_growth = set()
    for sec, rows in sector_rows.items():
        sst = status_frame(dt, arts, rows, win, cfg, company_X, companies).rename(columns={"Event-driven": "Event_driven"})
        srec = rows[recent_mask[rows]]
        sfirst = rows[np.isin(arts["month"].to_numpy()[rows], win.first12)]
        # 분야 특화도: 분야 내 최근 비중 / 전체 최근 비중
        all_recent_share = np.asarray(dt.X[recent_rows].sum(axis=0)).ravel() / max(len(recent_rows), 1)
        lift = sst["recent_share"].to_numpy() / np.maximum(all_recent_share, 1e-9)
        sst["lift"] = lift
        snet = build_network(dt, srec, f"{sec} 최근 12개월", dict(cfg, network=dict(cfg["network"], top_n=250)))
        snet_first = build_network(dt, sfirst, f"{sec} 첫 12개월", dict(cfg, network=dict(cfg["network"], top_n=250))) if len(sfirst) > 50 else None
        matches = {m["cluster"]: m for m in match_clusters(snet_first, snet)} if snet_first else {}
        clusters = []
        for c in snet.clusters[:10]:
            crow, hits = cluster_articles(dt, srec, c["keywords"][:15])
            m = matches.get(c["id"], {})
            clusters.append({"id": c["id"], "name_candidate": c["name_candidate"], "keywords": c["keywords"][:15],
                             "central": c["central"], "size": c["size"], "articles": int(len(crow)),
                             "top_edges": c["top_edges"][:5], "change": m.get("status"), "added": m.get("added", [])[:6],
                             "removed": m.get("removed", [])[:6], "arts": store.pick(crow, 3, hits)})

        def rows_for(mask, sort_col, n=15):
            d = sst[mask & (sst["articles"] >= 20)].sort_values(sort_col, ascending=False).head(n)
            res = []
            for r in d.itertuples():
                kr = dt.docs_with(r.keyword)
                kr = np.intersect1d(kr, srec)
                res.append({"k": r.keyword, "status": [k for k in STATUS_ORDER if getattr(r, k.replace("-", "_"))],
                            "recent_share": float(r.recent_share), "base_share": float(r.base_share), "ratio": float(r.ratio),
                            "trend6q": float(r.trend6q), "lift": float(r.lift), "recent_df": int(r.recent_df),
                            "company_specific": bool(r.company_specific), "top_company": r.top_company,
                            "arts": store.pick(kr, 2)})
            return res

        rising = sst["Emerging"] | sst["Growing"]
        sector_out[sec] = {
            "articles": int(len(rows)), "recent_articles": int(len(srec)),
            "rising": rows_for(rising, "ratio"),
            "specialized": rows_for((sst["lift"] >= 2) & (sst["recent_df"] >= 10), "recent_share"),
            "established": rows_for(sst["Established"], "mean_share", 10),
            "declining": rows_for(sst["Declining"], "base_share", 10),
            "event": rows_for(sst["Event_driven"], "peak_multiple", 10),
            "clusters": clusters,
        }

    dt, arts = core_dt, core_arts
    log("후보·출력")
    result = assemble(cfg, corpus, arts, win, dt, st, stx, profiles, dif, sectors, net_all, net_first, net_prev,
                      net_recent, stru, pairs, sconv_year, sconv_half, qnets, sector_out, store)
    result["meta"]["sector_tabs"] = list(sector_out)
    result["meta"]["related_label"] = related
    result["meta"]["related_articles"] = int(len(arts_all) - n_core)
    result["lexicon"] = {"stopwords": len(lex.stopwords), "synonyms": len(lex.synonym_map), "compounds": len(lex.compounds),
                         "path": str(lexicon_path or "lexicon.yaml")}

    # 표 출력
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    ktab.merge(st.drop(columns=["articles"]), on="keyword").to_csv(tables / "keyword_metrics.csv", index=False, encoding="utf-8-sig")
    long_series(dt, series, "month", top=3000).to_csv(tables / "keyword_monthly.csv", index=False, encoding="utf-8-sig")
    long_series(dt, series, "quarter", top=3000).to_csv(tables / "keyword_quarterly.csv", index=False, encoding="utf-8-sig")
    long_series(dt, series, "year", top=3000).to_csv(tables / "keyword_yearly.csv", index=False, encoding="utf-8-sig")
    for net, name in ((net_all, "all"), (net_first, "first12"), (net_recent, "recent12")):
        net.edges.to_csv(tables / f"edges_{name}.csv", index=False, encoding="utf-8-sig")
        if net.centrality is not None:
            net.centrality.to_csv(tables / f"centrality_{name}.csv", index=False, encoding="utf-8-sig")
    stru.reset_index().to_csv(tables / "structural_change.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(tables / "keyword_pair_convergence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(corpus.report["company_mention"]).to_csv(tables / "company_mention_rate.csv", index=False, encoding="utf-8-sig")

    if with_candidates:
        review = out / "review"
        review.mkdir(exist_ok=True)
        comp_set = set(companies) | {token_form(n) for n in names}
        sw = cand.stopword_candidates(dt, arts, series, sector_rows, net_all.centrality, comp_set)
        syn = cand.synonym_candidates(dt, arts, texts)
        cmp_ = cand.compound_candidates(tokens, texts, arts)
        with pd.ExcelWriter(review / "lexicon_candidates.xlsx") as xw:
            sw.to_excel(xw, sheet_name="불용어", index=False)
            syn.to_excel(xw, sheet_name="동의어", index=False)
            cmp_.to_excel(xw, sheet_name="복합어", index=False)
            pd.DataFrame(corpus.report["company_mention"]).to_excel(xw, sheet_name="기업 등장률", index=False)
        result["candidates"] = {
            "stopwords": sw.head(400).to_dict("records"), "synonyms": syn.head(150).to_dict("records"),
            "compounds": cmp_.head(150).to_dict("records"),
        }

    with open(out / "radar.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=_jsonable)
    return result


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def _net_summary(net, dt, store, recent_rows_all, n_clusters=12) -> dict:
    cl = []
    for c in net.clusters[:n_clusters]:
        rows_c, hits = cluster_articles(dt, recent_rows_all, c["keywords"][:15])
        cl.append({"id": c["id"], "name_candidate": c["name_candidate"], "keywords": c["keywords"][:20], "central": c["central"],
                   "size": c["size"], "articles": int(len(rows_c)), "top_edges": c["top_edges"][:6],
                   "arts": store.pick(rows_c, 3, hits)})
    nodes = []
    if net.centrality is not None:
        for r in net.centrality.itertuples():
            nodes.append([r.keyword, int(net.df[net.words.index(r.keyword)]), int(r.cluster), round(r.degree_centrality, 4),
                          round(r.betweenness, 4)])
    edges = net.edges.sort_values("jaccard", ascending=False)
    return {"label": net.label, "articles": net.n_docs, "clusters": cl, "nodes": nodes,
            "edges": edges[["a", "b", "co", "jaccard", "npmi"]].round(4).values.tolist()}


def assemble(cfg, corpus, arts, win, dt, st, stx, profiles, dif, sectors, net_all, net_first, net_prev, net_recent,
             stru, pairs, sconv_year, sconv_half, qnets, sector_out, store) -> dict:
    prof = set(profiles)

    def pick(mask, sort, n=40, asc=False):
        d = st[mask & st["keyword"].isin(prof)].sort_values(sort, ascending=asc).head(n)
        return d["keyword"].tolist()

    sig3 = st["sectors_recent"] >= 3
    overall = {
        # 확산만으로는 '확대'가 아니다: 비중도 늘어야 한다
        "rising": pick((st["Growing"] | st["Emerging"] | (st["Spreading"] & (st["ratio"] >= 1.2))) & sig3
                       & ~st["Declining"] & ~st["company_specific"],
                       "recent_share"),
        "established": pick(st["Established"], "mean_share", 30),
    }
    cross = pick(sig3 & ~st["company_specific"] & (st["ratio"] >= 1.0) & st["keyword"].map(lambda w: w in profiles and sum(
        a in ("↑", "↑↑") for a in profiles[w]["diffusion"]["arrows"].values()) >= 2), "sectors_recent", 40)
    emerging = pick(st["Emerging"], "ratio", 40)
    growing = pick(st["Growing"], "ratio", 40)
    declining = pick(st["Declining"], "ratio", 30, asc=True)
    event = pick(st["Event-driven"], "peak_multiple", 30)
    spreading = pick(st["Spreading"], "sectors_recent", 30)
    converging = pick(st["Converging"], "ratio", 30)
    # 기사가 많은(산업적으로 중요한) 키워드부터. 겹침 순으로 두면 맥락이 흐린 일반어가 앞에 온다
    s_idx = stru[stru["structural_change"]].sort_values("recent_df", ascending=False)
    structural = [w for w in s_idx.index if w in prof][:60]

    def conv_table(sc):
        S = len(sc["sectors"])
        rows = []
        ps = sc["periods"]
        for a in range(S):
            for b in range(a + 1, S):
                dser = [None if np.isnan(sc["distance"][p][a, b]) else round(float(sc["distance"][p][a, b]), 3) for p in ps]
                sser = [round(float(sc["shared_rate"][p][a, b]), 4) for p in ps]
                valid = [x for x in dser if x is not None]
                if len(valid) < 2:
                    continue
                # 다리 키워드: 두 분야 모두에서 과대표현된 키워드 (마지막 구간)
                pa, pb = sc["profiles"][ps[-1]][a], sc["profiles"][ps[-1]][b]
                pa0, pb0 = sc["profiles"][ps[0]][a], sc["profiles"][ps[0]][b]
                both = np.minimum(pa, pb)
                gain = both - np.minimum(pa0, pb0)
                top = np.argsort(-both)[:8]
                rows.append({"a": sc["sectors"][a], "b": sc["sectors"][b], "distance": dser, "shared_rate": sser,
                             "change": round(valid[-1] - valid[0], 3),
                             "bridge": [dt.vocab[k] for k in top if both[k] > 0],
                             "bridge_new": [dt.vocab[k] for k in np.argsort(-gain)[:8] if gain[k] > 0]})
        return {"periods": ps, "pairs": sorted(rows, key=lambda r: r["change"])}

    # 전체 요약 통계
    mon = arts.groupby("month").size()
    report = dict(corpus.report)
    report["company_mention"] = report["company_mention"][:40]
    return {
        "meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "articles": int(len(arts)), "vocab": len(dt.vocab),
                 "date_min": report["date_min"], "date_max": report["date_max"], "months": win.months,
                 "monthly_articles": [int(mon.get(m, 0)) for m in win.months], "quarters": win.quarters,
                 "recent_q": win.recent_q, "base_q": win.base_q, "recent12": [win.recent12[0], win.recent12[-1]],
                 "prev12": [win.prev12[0], win.prev12[-1]], "first12": [win.first12[0], win.first12[-1]],
                 "sectors": sectors, "sector_articles": {s: int(v) for s, v in report["sectors"].items()},
                 "cleaning": {k: v for k, v in report.items() if k != "sectors"}, "config": cfg},
        "overall": overall, "cross_sector": cross, "emerging": emerging, "growing": growing, "declining": declining,
        "event": event, "spreading": spreading, "converging": converging, "structural": structural,
        "keyword_pairs": pairs.to_dict("records"),
        "sector_convergence": {"year": conv_table(sconv_year), "half": conv_table(sconv_half)},
        "bridges": bridge_keywords(net_recent),
        "networks": {"recent": _net_summary(net_recent, dt, store, win.rows(arts, "month", win.recent12)),
                     "first": _net_summary(net_first, dt, store, win.rows(arts, "month", win.first12)),
                     "cluster_changes": match_clusters(net_first, net_recent)},
        "quarters": qnets,
        "sectors": sector_out,
        "profiles": profiles,
        "articles": {str(k): v for k, v in store.used.items()},
    }
