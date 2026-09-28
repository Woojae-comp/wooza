"""키워드 추출 (설계 4~5장).

형태소 분석 결과는 (형태, 시작, 끝, 품사) 순서열로 캐시한다.
사용자가 lexicon.yaml에서 복합어를 승인하면 붙어 있는 형태소를 캐시에서 바로 합치므로
형태소 분석을 다시 돌리지 않는다.
"""
from __future__ import annotations

import hashlib
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Lexicon

Token = tuple[str, int, int, str]  # form, start, end, tag
KEEP_TAGS = {"NNG", "NNP", "SL", "XSN"}
HANGUL = re.compile(r"[가-힣]")
# 붙어 있는 것으로 보는 간격 (K-POP, 숏-폼)
JOINERS = {"", "-", "·"}


def analysis_text(title: str, summary: str) -> str:
    """기본 분석 텍스트는 제목 + 요약. 검색어는 넣지 않는다."""
    return f"{title}\n{summary}"


def _space_free(name: str) -> str:
    """기업명을 한 형태소로 잡기 위한 표기 ('CJ ENM' → 'CJENM', 'JYP Ent.' → 'JYPEnt', 'SM C&C' → 'SMCC')."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", name)


token_form = _space_free


def space_joined_names(companies: list[str], aliases: dict[str, list[str]] | None) -> list[str]:
    names = set(companies)
    for al in (aliases or {}).values():
        names.update(al or [])
    return sorted(names, key=len, reverse=True)


class Prep:
    """띄어쓰기가 들어간 기업명('CJ ENM')을 붙여 써서 한 형태소로 잡히게 한다.
    토큰 위치는 이 변환 뒤 텍스트 기준이다."""

    def __init__(self, names: list[str]):
        spaced = [n for n in names if _space_free(n) != n and len(_space_free(n)) >= 2]
        self.pat = re.compile("|".join(re.escape(n) for n in spaced)) if spaced else None

    def __call__(self, text: str) -> str:
        return self.pat.sub(lambda m: _space_free(m.group(0)), text) if self.pat else text


def prepped_texts(articles: pd.DataFrame, names: list[str]) -> list[str]:
    prep = Prep(names)
    return [prep(analysis_text(t, s)) for t, s in zip(articles["title"], articles["summary"])]


def tokenize_texts(texts: list[str], names: list[str]) -> list[list[Token]]:
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    for n in names:
        sf = _space_free(n)
        if HANGUL.search(sf) or len(sf) > 2:
            kiwi.add_user_word(sf, "NNP", 0)
    return [[(t.form, t.start, t.start + t.len, t.tag) for t in toks if t.tag in KEEP_TAGS]
            for toks in kiwi.tokenize(texts)]


def tokenize_corpus(texts: list[str], names: list[str], cache_dir: Path) -> list[list[Token]]:
    key = hashlib.sha1("\x00".join(texts).encode() + "\x00".join(names).encode()).hexdigest()[:16]
    cache = cache_dir / f"tokens_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    toks = tokenize_texts(texts, names)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(toks, f, protocol=pickle.HIGHEST_PROTOCOL)
    return toks


def majority_tags(tokens: list[list[Token]]) -> dict[str, str]:
    """형태별 가장 많이 붙은 품사. 복합어로 합쳐진 형태는 없으므로 호출하는 쪽에서 NNG로 본다."""
    cnt: dict[str, Counter] = {}
    for toks in tokens:
        for f, _, _, t in toks:
            cnt.setdefault(f, Counter())[t] += 1
    return {f: c.most_common(1)[0][0] for f, c in cnt.items()}


def contiguous(a: Token, b: Token, text: str) -> bool:
    return b[1] >= a[2] and text[a[2]:b[1]] in JOINERS


def merge_tokens(tokens: list[Token], text: str, lexicon: Lexicon, attach_suffixes: set[str]) -> list[str]:
    """복합어 병합, 접미사 부착. 결과는 품사 제거된 형태 목록 (필터 전)."""
    forms: list[tuple[str, str, int, int]] = []
    i, n = 0, len(tokens)
    while i < n:
        matched = False
        for comp in lexicon.compounds:
            k = len(comp)
            if i + k <= n and tuple(t[0] for t in tokens[i:i + k]) == comp and all(
                    contiguous(tokens[j], tokens[j + 1], text) for j in range(i, i + k - 1)):
                forms.append(("".join(comp), "NNG", tokens[i][1], tokens[i + k - 1][2]))
                i += k
                matched = True
                break
        if matched:
            continue
        f, s, e, tag = tokens[i]
        if tag == "XSN":
            if f in attach_suffixes and forms and forms[-1][3] == s and forms[-1][1] != "XSN":
                pf, ptag, ps, _ = forms[-1]
                forms[-1] = (pf + f, ptag, ps, e)
            else:
                forms.append((f, "XSN", s, e))
        else:
            forms.append((f, tag, s, e))
        i += 1
    return [f for f, tag, _, _ in forms if tag != "XSN"]


def keep_form(w: str, min_hangul: int, min_latin: int) -> bool:
    if HANGUL.search(w):
        return len(w) >= min_hangul
    return len(w) >= min_latin and not w.isdigit()


def extract_keywords(texts: list[str], tokens: list[list[Token]], lexicon: Lexicon, cfg: dict) -> list[list[str]]:
    """기사별 키워드 목록 (중복 포함, 출현 횟수 계산용)."""
    kc = cfg["keywords"]
    attach = set(kc.get("attach_suffixes", []))
    mh, ml = kc.get("min_hangul_length", 2), kc.get("min_latin_length", 2)
    # 로마자 대소문자: 가장 많이 쓰인 표기로 통일
    latin = Counter()
    out = []
    for toks, text in zip(tokens, texts):
        words = merge_tokens(toks, text, lexicon, attach)
        words = [w for w in words if keep_form(w, mh, ml)]
        latin.update(w for w in words if not HANGUL.search(w))
        out.append(words)
    best: dict[str, tuple[int, str]] = {}
    for w, c in latin.items():
        k = w.lower()
        if k not in best or c > best[k][0]:
            best[k] = (c, w)
    case = {w: best[w.lower()][1] for w in latin}
    final = []
    for words in out:
        norm = []
        for w in words:
            w = case.get(w, w)
            w = lexicon.normalize(w)
            if w is not None:
                norm.append(w)
        final.append(norm)
    return final


class DocTerm:
    """기사 × 키워드 이진 행렬 (설계 9장)과 출현 횟수."""

    def __init__(self, doc_words: list[list[str]], min_df: int, vocab: list[str] | None = None):
        df = Counter()
        tf = Counter()
        for ws in doc_words:
            df.update(set(ws))
            tf.update(ws)
        if vocab is None:
            vocab = sorted((w for w, c in df.items() if c >= min_df), key=lambda w: (-df[w], w))
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        rows, cols = [], []
        for r, ws in enumerate(doc_words):
            for w in set(ws):
                j = self.index.get(w)
                if j is not None:
                    rows.append(r)
                    cols.append(j)
        self.X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(doc_words), len(vocab)))
        self.df = np.array([df[w] for w in vocab])
        self.tf = np.array([tf[w] for w in vocab])
        self.Xc = self.X.tocsc()

    def docs_with(self, word: str) -> np.ndarray:
        j = self.index[word]
        return self.Xc.indices[self.Xc.indptr[j]:self.Xc.indptr[j + 1]]
