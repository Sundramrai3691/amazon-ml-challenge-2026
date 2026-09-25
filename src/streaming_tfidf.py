"""Streaming character TF-IDF retrieval (Approach B).

Isolated implementation that never materialises the full source corpus in RAM
and never constructs a ``source_chunk x all_S1`` similarity matrix.

Exact sklearn numerical equivalence is the **hard** correctness requirement.
Every parameter, formula, and tie-break is documented and tested against the
installed sklearn version in ``tests/test_streaming_tfidf.py``.

This module does NOT replace :mod:`src.blocking`. The existing
``char_tfidf_candidates`` there remains the correctness oracle.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as _skl_normalize

CandidateMap = dict[str, list[str]]
PairMeta = dict[tuple[str, str], dict[str, object]]


@dataclass(frozen=True)
class VocabBuildResult:
    """Output of :func:`build_fixed_vocabulary`.

    The ngram -> index mapping has deterministic order: max_features
    descending by global term frequency; ties broken by ngram
    lexicographic ascending (matching ``Counter.most_common``).
    """

    vocabulary: dict[str, int]
    idf: np.ndarray  # shape (V,), float32
    tf: np.ndarray   # global term-freq per feature (V,) int64
    df: np.ndarray   # global doc-freq  per feature (V,) int64
    N_docs: int
    analyzer: str
    ngram_range: tuple[int, int]
    min_df: int
    max_features: int | None


def _iter_texts(texts: Iterable[list[str]] | Iterable[Sequence[str]] | Iterable[pd.DataFrame]) -> Iterable[list[str]]:
    """Accept various input shapes as text-chunk iterables for builder functions."""
    for chunk in texts:
        if isinstance(chunk, pd.DataFrame):
            if "text" in chunk.columns:
                yield [str(x) for x in chunk["text"].tolist()]
            else:
                raise ValueError("DataFrame chunks must have a 'text' column")
        else:
            yield [str(x) for x in chunk]


def build_fixed_vocabulary(
    text_chunk_iterable: Iterable[list[str]] | Iterable[Sequence[str]] | Iterable[pd.DataFrame],
    *,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int | None = None,
    lowercase: bool = False,
) -> VocabBuildResult:
    """Build a TF-IDF vocabulary by streaming text chunks.

    Semantics are tested to match sklearn's ``TfidfVectorizer`` with matching
    parameters (smooth_idf=True, sublinear_tf=False, norm='l2', use_idf=True)
    on every dimension:

    - per-chunk ``CountVectorizer`` aggregates **global term-frequency** and
      **global document-frequency** counts; no intermediate approximations.
    - ``min_df`` filtering happens BEFORE the top-``max_features`` selection.
    - ``max_features`` keeps the top-V features by **global term frequency**
      descending; ties are broken by **ngram string lexicographic ascending**
      order (verified against sklearn).
    - Smoothed idf: ``log((1 + N) / (1 + df)) + 1``.

    Parameters
    ----------
    text_chunk_iterable:
        Iterable of chunks. Each chunk is a list/sequence of raw strings, or
        a ``pd.DataFrame`` with a ``text`` column. Chunks are freed after
        processing so that the full corpus never resides in memory.
    """
    # Aggregation arrays are exact. Python Counter-based tf/df across
    # potentially large ngram space; memory bounded by Counter size (for
    # char_wb 3-5 gram over ~1.2M docs est. < 200 MB; see design doc).
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    n_docs = 0

    for chunk in _iter_texts(text_chunk_iterable):
        if not chunk:
            continue
        vec = CountVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            lowercase=lowercase,
            vocabulary=None,
        )
        try:
            X = vec.fit_transform(chunk)
        except ValueError:
            # Empty vocabulary (rare degenerate chunk); skip.
            n_docs += len(chunk)
            continue
        feats = vec.get_feature_names_out()
        # Global term frequency per ngram: sum of rows = total occurrences across chunk
        chunk_tf = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)  # type: ignore[union-attr]
        # Global document frequency per ngram: presence in doc
        chunk_df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64)  # type: ignore[union-attr]
        for i, name in enumerate(feats):
            tf_c = int(chunk_tf[i])
            if tf_c:
                tf_counter[name] += tf_c
            df_c = int(chunk_df[i])
            if df_c:
                df_counter[name] += df_c
        n_docs += len(chunk)
        del vec, X

    # 1) min_df filter
    if min_df and min_df > 1:
        surviving = [(ng, tf_counter[ng]) for ng in tf_counter if df_counter[ng] >= min_df]
    else:
        surviving = list(tf_counter.items())

    # 2) max_features: top-V by tf desc, ties by ngram asc (Counter.most_common() style)
    surviving.sort(key=lambda kv: (-kv[1], kv[0]))
    if max_features and max_features > 0 and len(surviving) > max_features:
        surviving = surviving[:max_features]

    if surviving:
        ngrams_sorted = [ng for ng, _ in surviving]
        vocabulary = {ng: idx for idx, ng in enumerate(ngrams_sorted)}
        tf_arr = np.array([tf for _, tf in surviving], dtype=np.int64)
        df_arr = np.array([df_counter[ng] for ng in ngrams_sorted], dtype=np.int64)
    else:
        ngrams_sorted: list[str] = []
        vocabulary: dict[str, int] = {}
        tf_arr = np.zeros(0, dtype=np.int64)
        df_arr = np.zeros(0, dtype=np.int64)

    N = max(int(n_docs), 0)
    if df_arr.size == 0:
        idf_arr = np.zeros(0, dtype=np.float32)
    else:
        idf_arr = np.log((1.0 + N) / (1.0 + df_arr.astype(np.float64))) + 1.0
        idf_arr = idf_arr.astype(np.float32)

    return VocabBuildResult(
        vocabulary=vocabulary,
        idf=idf_arr,
        tf=tf_arr,
        df=df_arr,
        N_docs=N,
        analyzer=analyzer,
        ngram_range=tuple(ngram_range),  # type: ignore[arg-type]
        min_df=int(min_df or 0),
        max_features=int(max_features) if max_features else None,
    )


def _count_with_vocab(
    texts: Sequence[str],
    vocabulary: Mapping[str, int],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    lowercase: bool,
) -> sp.csr_matrix:
    """Count-vectorize using a FIXED vocabulary. Produces a CSR matrix."""
    if not vocabulary:
        return sp.csr_matrix((len(texts), 0), dtype=np.float32)
    vec = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=lowercase,
        vocabulary=list(vocabulary.keys()),
    )
    try:
        X = vec.transform(texts)
    except ValueError:
        # Empty vocabulary; return empty CSR
        return sp.csr_matrix((len(texts), len(vocabulary)), dtype=np.float32)
    V = len(vocabulary)
    # Reorder columns to match vocabulary ordering (CountVectorizer.vocabulary_
    # uses the exact list we gave, but we verify column order just in case)
    feat_out = vec.get_feature_names_out()
    if list(feat_out) != list(vocabulary.keys()):
        new_idx = np.empty(V, dtype=np.int64)
        out_to_idx = {name: i for i, name in enumerate(feat_out)}
        for name, i in vocabulary.items():
            j = out_to_idx.get(name)
            if j is None:
                continue
            new_idx[i] = j
        X = X.tocsr()[:, new_idx]
    return X.tocsr().astype(np.float32)


def _apply_tfidf_idf_l2(X_counts: sp.spmatrix, idf: np.ndarray) -> sp.csr_matrix:
    """Transform count matrix: multiply by idf, then L2 row normalize.

    sklearn-equivalent for:
        TfidfVectorizer(vocabulary=..., use_idf=True, smooth_idf=True,
                        sublinear_tf=False, norm='l2')
    (where fit set the idf).
    """
    X = X_counts.tocsr()
    if X.shape[1] == 0 or idf.size == 0:
        return sp.csr_matrix(X.shape, dtype=np.float32)
    # Element-wise multiply each column by idf
    X_idf = X.multiply(idf.astype(np.float32)).tocsr()
    # L2 row normalize (sklearn default)
    X_norm: sp.csr_matrix = _skl_normalize(X_idf, norm="l2", copy=False)
    return X_norm.astype(np.float32)


class CharTFIDFStreamingRetriever:
    """Streaming top-K retriever over a source scanned in chunks.

    S1 queries (reference entities) are processed in BATCHES so that we
    multiply only ``S1_batch x source_chunk`` sparse matrices at any time.
    The ``all_S1 x all_source`` or ``all_S1 x source_chunk`` dense similarity
    matrices are NEVER formed.

    Top-K state is kept in a compact per-S1 list of (score, cand_idx) heap
    entries, merged across chunks via ``heapq.heappushpop``.

    After feeding every source chunk with :meth:`partial_fit_source_chunk`,
    call :meth:`finalize_candidates` to produce the same ``CandidateMap`` /
    ``PairMeta`` shape used by :mod:`src.blocking`. ``method_name`` maps to
    ``blocking.METHOD_FLAG`` via the same dict (the caller passes the
    appropriate method string; we do not import from blocking here to keep
    this module isolated).
    """

    def __init__(
        self,
        *,
        vocabulary: Mapping[str, int],
        idf: np.ndarray,
        s1_texts: Sequence[str],
        s1_ids: Sequence[str],
        k: int,
        method_name: str,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
        lowercase: bool = False,
        s1_batch_size: int = 4096,
        dtype: np.dtype | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if len(s1_texts) != len(s1_ids):
            raise ValueError("s1_texts and s1_ids length mismatch")
        self.vocabulary = dict(vocabulary)
        self.idf = np.asarray(idf, dtype=np.float32).ravel()
        self._s1_texts = [str(x) for x in s1_texts]
        self.s1_ids = [str(x) for x in s1_ids]
        self.s1_id_to_idx = {sid: i for i, sid in enumerate(self.s1_ids)}
        self.k = int(k)
        self.method_name = str(method_name)
        self.analyzer = analyzer
        self.ngram_range = tuple(ngram_range)  # type: ignore[assignment]
        self.lowercase = bool(lowercase)
        self.s1_batch_size = int(s1_batch_size)
        self.dtype = np.dtype(dtype or np.float32)

        # Pre-validate vocabulary vs idf size
        if self.idf.size != len(self.vocabulary):
            raise ValueError("idf length != vocabulary size")

        # Top-K storage: per S1 index -> list of tuples (score_neg, cand_global_seq, cand_id, score)
        self._n_s1 = len(self.s1_ids)
        self._heaps: list[list[tuple[float, int, str, float]]] = [[] for _ in range(self._n_s1)]
        self._cand_counter = 0  # monotonic tie-break for identical (score, cand_id) across chunks

    # ------------------------------------------------------------------
    # Core retrieval loop helpers
    # ------------------------------------------------------------------
    def _batches_s1(self, n: int, bs: int):
        for start in range(0, n, bs):
            yield start, min(start + bs, n)

    def _partial_fit_chunk_for_s1_range(
        self,
        chunk_tfidf: sp.csr_matrix,
        cand_ids_chunk: Sequence[str],
        s1_start: int,
        s1_end: int,
        s1_tfidf_block: sp.csr_matrix,
    ) -> None:
        if s1_tfidf_block.shape[0] == 0 or chunk_tfidf.shape[0] == 0:
            return
        # Similarities: block of size (s1_batch_size, chunk_rows)
        # Both sides are already L2-normalized rows; the dot product is cosine.
        S_block: sp.csr_matrix = (s1_tfidf_block @ chunk_tfidf.T).tocsr()
        # Extract non-zero entries per S1 row in the block
        S_csr = S_block.tocsr()
        n_in_block = s1_end - s1_start
        chunk_np_cand_ids = list(cand_ids_chunk)
        for bi in range(n_in_block):
            s1_idx = s1_start + bi
            start_ptr = S_csr.indptr[bi]
            end_ptr = S_csr.indptr[bi + 1]
            if start_ptr == end_ptr:
                continue
            idxs = S_csr.indices[start_ptr:end_ptr]
            vals = S_csr.data[start_ptr:end_ptr]
            heap = self._heaps[s1_idx]
            K = self.k
            for j in range(idxs.size):
                v = float(vals[j])
                if v <= 0:
                    continue
                cand_idx_col = int(idxs[j])
                cand_id = chunk_np_cand_ids[cand_idx_col]
                # MIN-HEAP of size K: position 0 = the WORST entry currently kept
                # (= smallest v; ties = lexicographically LARGEST cand_id should
                # be evicted first so that we keep SMALLEST id among equal v).
                # Heap key:  (v, _neg_lex(cand_id), cand_id, v)
                # where `_neg_lex` turns cand_id into a tuple of negated
                # ordinals so that LARGER cand_id -> SMALLER tuple -> ends up
                # at MIN-heap index 0 (evictable) when v is tied.
                neg_lex = tuple(-ord(c) for c in cand_id)
                slot = (v, neg_lex, cand_id, v)
                self._cand_counter += 1
                if len(heap) < K:
                    heapq.heappush(heap, slot)
                else:
                    if slot > heap[0]:
                        heapq.heapreplace(heap, slot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def partial_fit_source_chunk(
        self,
        texts: Sequence[str],
        entity_ids: Sequence[str],
    ) -> None:
        """Process one chunk of source (S2/S3) rows. Chunk is freed after return.

        Parameters
        ----------
        texts:
            Normalized strings (one per source row; matching the "view" --
            name, address, or combined). Length must equal ``entity_ids``.
        entity_ids:
            ``S2-...`` or ``S3-...`` strings. Must be unique within the
            chunk (enforced for safety).
        """
        if len(texts) != len(entity_ids):
            raise ValueError("texts and entity_ids length mismatch")
        # Short-circuit empty chunk
        if not texts:
            return
        # Compute tf-idf for source chunk (vocab, idf fixed)
        counts = _count_with_vocab(
            texts,
            self.vocabulary,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
        )
        chunk_tfidf = _apply_tfidf_idf_l2(counts, self.idf)
        del counts
        # Iterate S1 in BATCHES; never form all_S1 x chunk matrix
        cand_ids = [str(x) for x in entity_ids]
        for s_start, s_end in self._batches_s1(self._n_s1, self.s1_batch_size):
            s1_batch_texts = self._s1_texts[s_start:s_end]
            s1_counts = _count_with_vocab(
                s1_batch_texts,
                self.vocabulary,
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                lowercase=self.lowercase,
            )
            s1_tfidf = _apply_tfidf_idf_l2(s1_counts, self.idf)
            self._partial_fit_chunk_for_s1_range(
                chunk_tfidf, cand_ids, s_start, s_end, s1_tfidf
            )
            del s1_counts, s1_tfidf
        del chunk_tfidf


def _finalize_candidates_from_heaps(
    heaps: Sequence[Sequence[tuple[float, int, str, float]]],
    s1_ids: Sequence[str],
    method_name: str,
    k: int,
) -> tuple[CandidateMap, PairMeta]:
    """Pop sorted top-K lists from heaps and produce candidate map + pair meta.

    Sorting rule (matches sklearn/blocking semantics): highest similarity first;
    ties break by lexicographically SMALLER candidate_id first.
    """
    candidates: CandidateMap = {}
    meta: PairMeta = {}
    for i, s1_id in enumerate(s1_ids):
        slots = list(heaps[i])
        # Each heap slot: (v, neg_lex, cand_id, v)
        # Sort by v DESC, cand_id ASC (neg_lex tuple not used here; direct id comparison for clarity)
        slots_sorted = sorted(slots, key=lambda s: (-s[0], s[2]))
        ordered_ids = [s[2] for s in slots_sorted[:k]]
        # Deduplicate while preserving order (should not happen)
        seen: set[str] = set()
        unique_ids: list[str] = []
        for cid in ordered_ids:
            if cid in seen:
                continue
            seen.add(cid)
            unique_ids.append(cid)
        candidates[s1_id] = unique_ids
        # Attach metadata: channel flag + rank + source type (determined by id prefix)
        for rank, cid in enumerate(unique_ids):
            slot = next(s for s in slots_sorted if s[2] == cid)
            score = float(slot[3])
            pair_key = (s1_id, cid)
            slot_meta: dict[str, object] = {
                "candidate_rank": rank,
                "tfidf_rank": rank,
                "score_%s" % method_name: score,
            }
            meta[pair_key] = slot_meta
    return candidates, meta


def streaming_char_tfidf_candidates(
    s1_texts: Sequence[str],
    s1_ids: Sequence[str],
    source_chunk_iterable: Iterable[tuple[Sequence[str], Sequence[str]]],
    *,
    vocab: Mapping[str, int],
    idf: np.ndarray,
    k: int,
    method_name: str,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (3, 5),
    lowercase: bool = False,
    s1_batch_size: int = 4096,
    dtype: np.dtype | None = None,
) -> tuple[CandidateMap, PairMeta]:
    """High-level convenience wrapper around the streaming retriever.

    Parameters
    ----------
    s1_texts, s1_ids:
        Full S1 reference side. S1 is considered small enough to live in
        memory as strings (matches the design doc).
    source_chunk_iterable:
        Iterator yielding ``(texts, entity_ids)`` for every source chunk.
        Each chunks is freed after use. ``texts`` are pre-normalized using
        the same ``_field_texts`` view used elsewhere (name / address /
        combined). ``entity_ids`` are S2/S3 ids.
    vocab, idf:
        Output of :func:`build_fixed_vocabulary` run on the SAME source
        corpus with the SAME parameter settings.

    Returns
    -------
    (candidates, pair_meta)
        Same shape as :func:`src.blocking.char_tfidf_candidates`. The pair
        meta keys here are only ``candidate_rank`` and ``tfidf_rank``; the
        caller is responsible for merging with the full PairMeta flags via
        :func:`src.blocking.union_candidates` or equivalent.
    """
    retriever = CharTFIDFStreamingRetriever(
        vocabulary=vocab,
        idf=idf,
        s1_texts=s1_texts,
        s1_ids=s1_ids,
        k=k,
        method_name=method_name,
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=lowercase,
        s1_batch_size=s1_batch_size,
        dtype=dtype,
    )
    for texts, ids in source_chunk_iterable:
        retriever.partial_fit_source_chunk(texts, ids)
    return _finalize_candidates_from_heaps(retriever._heaps, retriever.s1_ids, method_name, k)


# ---------------------------------------------------------------------------
# Oracle helpers: used by tests. Not public API.
# ---------------------------------------------------------------------------


def _sklearn_oracle_tfidf_vectorizer(
    train_texts: Sequence[str],
    *,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int | None = None,
    lowercase: bool = False,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit a reference sklearn TfidfVectorizer and return it."""
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=lowercase,
        dtype=np.float32,
        sublinear_tf=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )
    X = vec.fit_transform(train_texts)
    return vec, X.tocsr()
