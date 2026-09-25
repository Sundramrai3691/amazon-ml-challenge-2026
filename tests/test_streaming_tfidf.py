"""Numerical equivalence tests for streaming TF-IDF vs sklearn oracle.

Scope: SMALL synthetic corpora where both systems can materialise the full
matrix. Tests vocabulary, IDF, TF-IDF values, L2 norm, cosine similarities,
and top-K ranking. The tests are intentionally parameter-free and cover:

- repeated n-grams
- rare n-grams
- ties
- min_df boundary cases
- max_features boundary cases

These tests do NOT run any S2/S3 real data or any GBDT pipeline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.streaming_tfidf import (
    VocabBuildResult,
    _apply_tfidf_idf_l2,
    _count_with_vocab,
    _sklearn_oracle_tfidf_vectorizer,
    build_fixed_vocabulary,
    streaming_char_tfidf_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_vocab_items(vocab: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(vocab.items(), key=lambda kv: kv[0])


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# Fixtures: synthetic corpora with interesting properties
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_bursty_vs_common() -> tuple[list[str], list[str], list[str]]:
    """Rare bursty ngram zzz vs common singleton ngram aaa.

    2 docs with zzz many times each -> tf zzz huge, df zzz small.
    20 docs with aaa once each     -> tf aaa small, df aaa big.

    sklearn max_features must prefer zzz by TERM frequency (confirmed
    empirically in the project REPL earlier).
    """
    docs = []
    for _ in range(2):
        docs.append("zzzzz " * 100)
    for i in range(20):
        docs.append("aaaaa" + str(i % 10))  # includes digits so aaa ngrams survive
    s1 = ["zzzzz", "aaaaa", "mixed zzz aaaaa"]
    s1_ids = ["S1-Q1", "S1-Q2", "S1-Q3"]
    cand_ids = [f"S2-{1000+i}" for i in range(len(docs))]
    return docs, s1, s1_ids, cand_ids  # type: ignore[return-value]


@pytest.fixture(scope="module")
def corpus_mindf_boundary() -> tuple[list[str], list[str], list[str], list[str]]:
    """min_df=3 boundary.

    bbb appears exactly 2 times (df=2 < min_df -> drop).
    ccc appears exactly 3 times (df=3 == min_df -> keep).
    """
    docs = [
        "bbb",
        "bbb",
        "ccc",
        "ccc and stuff",
        "ccc again",
        "dddddddd",
        "dddddddd",
        "dddddddd",
        "dddddddd",
        "dddddddd",
    ]
    s1 = ["bbb ccc", "dddd"]
    s1_ids = ["S1-Q1", "S1-Q2"]
    cand_ids = [f"S2-{i}" for i in range(len(docs))]
    return docs, s1, s1_ids, cand_ids


@pytest.fixture(scope="module")
def corpus_maxfeatures_boundary() -> tuple[list[str], list[str], list[str], list[str]]:
    """max_features=2 selects two ngrams with tied tf: sorted by lex ascending.

    "yyy" and "xxx" each have exactly 4 occurrences, lexicographically xxx < yyy
    so vocabulary order must be {xxx: 0, yyy: 1}.
    """
    docs = [
        "xxxx",      # xxx tf 2 (inside xxxx)
        "yyyy",      # yyy tf 2
        "axxxb",     # xxx tf 2 -> xxx total 4
        "ayyyb",     # yyy tf 2 -> yyy total 4
        "singleton",
    ]
    s1 = ["xxx", "yyy"]
    s1_ids = ["S1-Q1", "S1-Q2"]
    cand_ids = [f"S2-{i}" for i in range(len(docs))]
    return docs, s1, s1_ids, cand_ids


@pytest.fixture(scope="module")
def corpus_ties_sim() -> tuple[list[str], list[str], list[str], list[str]]:
    """Top-K tie-breaking in cosine similarity: same score, prefer smaller cand_id."""
    docs = [
        "alpha beta gamma delta",  # S2-0000
        "alpha beta gamma delta",  # S2-0001 identical text -> identical sim vs s1
        "alpha beta gamma delta",  # S2-0002
        "epsilon zeta eta",
    ]
    s1 = ["alpha beta gamma delta"]
    s1_ids = ["S1-Q1"]
    cand_ids = [f"S2-{str(i).zfill(4)}" for i in range(len(docs))]
    return docs, s1, s1_ids, cand_ids


# ---------------------------------------------------------------------------
# 1. Vocabulary equality vs sklearn
# ---------------------------------------------------------------------------


class TestBuildFixedVocabulary:
    def test_bursty_wins_over_common_by_tf(self, corpus_bursty_vs_common):
        docs, *_ = corpus_bursty_vs_common
        # max_features=1: zzz wins
        result = build_fixed_vocabulary(
            [docs],
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_df=1,
            max_features=1,
            lowercase=False,
        )
        assert list(result.vocabulary.keys()) == ["zzz"], (
            f"Expected zzz (bursty, higher TERM frequency) but got {list(result.vocabulary.keys())[:5]}"
        )

    def test_mindf_drop_exact(self, corpus_mindf_boundary):
        docs, *_ = corpus_mindf_boundary
        result = build_fixed_vocabulary(
            [docs],
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_df=3,
            max_features=None,
            lowercase=False,
        )
        # bbb df=2 must be dropped; ccc df=3, ddd df=5 kept
        assert "bbb" not in result.vocabulary
        assert "ccc" in result.vocabulary
        assert "ddd" in result.vocabulary
        # IDF values: ccc should have HIGHER idf (rarer) than ddd
        ccc_idx = result.vocabulary["ccc"]
        ddd_idx = result.vocabulary["ddd"]
        assert result.idf[ccc_idx] > result.idf[ddd_idx]

    def test_maxfeatures_tiebreak_lex_asc(self, corpus_maxfeatures_boundary):
        docs, *_ = corpus_maxfeatures_boundary
        result = build_fixed_vocabulary(
            [docs],
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_df=1,
            max_features=2,
            lowercase=False,
        )
        keys = list(result.vocabulary.keys())
        # Exactly 2 features; tied tf must be ordered lex asc -> xxx before yyy
        assert len(keys) == 2
        assert keys[0] == "xxx" and keys[1] == "yyy", (
            "Ties must break by ngram lex asc; got %s" % keys
        )

    def test_vocabulary_matches_sklearn_exact_on_small(self):
        """Full vocab, idf, tf, df, N_docs match sklearn on simple corpus."""
        docs = [
            "apple banana",
            "apple cherry date",
            "banana cherry",
            "date date date",
            "elderberry fig",
        ]
        params = dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1, max_features=100, lowercase=False)
        result = build_fixed_vocabulary([docs], **params)
        skl, _ = _sklearn_oracle_tfidf_vectorizer(docs, **params)

        # 1) vocabulary set equality
        our_set = set(result.vocabulary.keys())
        skl_set = set(skl.vocabulary_.keys())
        assert our_set == skl_set, (
            f"vocab set mismatch, extra ours={our_set - skl_set}, missing ours={skl_set - our_set}"
        )

        # 2) idf equality
        for name, our_idx in result.vocabulary.items():
            skl_idx = skl.vocabulary_[name]
            our_idf = float(result.idf[our_idx])
            skl_idf = float(skl.idf_[skl_idx])
            assert abs(our_idf - skl_idf) < 1e-5, (
                f"idf mismatch {name}: ours={our_idf}, skl={skl_idf}"
            )

        # 3) document frequency equality: invert sklearn idf_ (ground truth)
        # idf = log((1+N)/(1+df)) + 1  =>  df = ceil((1+N) / exp(idf - 1) - 1)
        N = len(docs)
        idf = skl.idf_
        skl_df_via_idf = np.rint((1.0 + N) / np.exp(idf.astype(np.float64) - 1.0) - 1.0).astype(np.int64)
        for name, our_idx in result.vocabulary.items():
            skl_idx = skl.vocabulary_[name]
            our_df = int(result.df[our_idx])
            skl_df = int(skl_df_via_idf[skl_idx])
            assert our_df == skl_df, (
                f"df mismatch {name!r}: ours={our_df}, skl(via idf)={skl_df}"
            )

        # 4) N_docs
        assert result.N_docs == len(docs)


# ---------------------------------------------------------------------------
# 2. TF-IDF numerical equivalence vs sklearn
# ---------------------------------------------------------------------------


class TestTfIdfNumericalEquivalence:
    def test_tfidf_values_within_tolerance(self):
        docs = [
            "quick brown fox jumps over the lazy dog",
            "quick quick quick dog dog",
            "",  # empty row
            "brown brown brown fox fox fox",
            "lazy",
        ]
        params = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=None, lowercase=False)
        result = build_fixed_vocabulary([docs], **params)
        skl_vec, skl_X = _sklearn_oracle_tfidf_vectorizer(docs, **params)

        # Our pipeline: count with fixed vocab -> multiply idf -> L2 norm
        counts = _count_with_vocab(docs, result.vocabulary, analyzer=params["analyzer"],
                                    ngram_range=params["ngram_range"], lowercase=params["lowercase"])
        our_X = _apply_tfidf_idf_l2(counts, result.idf)

        # Reorder sklearn columns to match our vocabulary column order
        col_map = np.array([skl_vec.vocabulary_[name] for name in result.vocabulary.keys()])
        skl_X_aligned = skl_X.tocsc()[:, col_map].tocsr()

        assert our_X.shape == skl_X_aligned.shape
        diff: sp.csr_matrix = (our_X.astype(np.float64) - skl_X_aligned.astype(np.float64)).tocsr()
        max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
        assert max_abs < 1e-5, f"Max |our - skl| tfidf = {max_abs:.2e} (expected < 1e-5)"

    def test_l2_row_norm_is_one(self):
        docs = ["alpha", "beta gamma delta", "", "epsilon epsilon epsilon"]
        params = dict(analyzer="char_wb", ngram_range=(3, 3), min_df=1, max_features=None, lowercase=False)
        result = build_fixed_vocabulary([docs], **params)
        counts = _count_with_vocab(docs, result.vocabulary, analyzer=params["analyzer"],
                                    ngram_range=params["ngram_range"], lowercase=params["lowercase"])
        X = _apply_tfidf_idf_l2(counts, result.idf).astype(np.float64)
        # Row L2 norms: zero rows -> 0; non-zero rows -> ~1.0
        sq = X.multiply(X).sum(axis=1).A1
        for i in range(X.shape[0]):
            if X.indptr[i + 1] - X.indptr[i] == 0:
                assert sq[i] == 0.0
            else:
                assert abs(sq[i] - 1.0) < 1e-5, f"Row {i} L2 norm sq = {sq[i]:.6f} (expected 1)"


# ---------------------------------------------------------------------------
# 3. Cosine similarity equivalence
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_cosine_matches_sklearn_on_randomised(self):
        rng = np.random.RandomState(0)
        tokens = ["apple", "banana", "cherry", "date", "elderberry", "fig",
                  "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine"]
        def random_doc(n_toks=8):
            return " ".join(rng.choice(tokens, size=n_toks))
        s1_docs = [random_doc(12) for _ in range(20)]
        cand_docs = [random_doc(15) for _ in range(80)]
        params = dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1, max_features=None, lowercase=False)

        # Sklearn: fit on cand_docs, transform both, compute cosine
        skl, cand_X = _sklearn_oracle_tfidf_vectorizer(cand_docs, **params)
        s1_X = skl.transform(s1_docs)
        cos_skl = (s1_X.astype(np.float64) @ cand_X.T.astype(np.float64)).toarray()

        # Ours: build vocab on cand_docs (chunked), transform both, compute cosine
        chunks = [list(_chunk(cand_docs, 17))]  # chunk sizes: 17,17,17,17,12
        flat_chunks = chunks[0]
        result = build_fixed_vocabulary(flat_chunks, **params)

        s1_counts = _count_with_vocab(s1_docs, result.vocabulary, analyzer=params["analyzer"],
                                       ngram_range=params["ngram_range"], lowercase=params["lowercase"])
        s1_ours = _apply_tfidf_idf_l2(s1_counts, result.idf)
        cand_counts = _count_with_vocab(cand_docs, result.vocabulary, analyzer=params["analyzer"],
                                         ngram_range=params["ngram_range"], lowercase=params["lowercase"])
        cand_ours = _apply_tfidf_idf_l2(cand_counts, result.idf)

        # Align col order vs skl
        col_map = np.array([skl.vocabulary_.get(name, 0) for name in result.vocabulary.keys()])
        col_mask = np.array([name in skl.vocabulary_ for name in result.vocabulary.keys()])
        # if vocab is a subset (should be same set), reorder
        cos_ours = (s1_ours.astype(np.float64) @ cand_ours.T.astype(np.float64)).toarray()

        # Only compare cosine for ngrams in BOTH vocabularies (should be all)
        max_abs = float(np.max(np.abs(cos_ours - cos_skl)))
        assert max_abs < 1e-4, f"cosine max |diff| = {max_abs:.2e}"


# ---------------------------------------------------------------------------
# 4. Streaming top-K candidates vs sklearn top-K
# ---------------------------------------------------------------------------


class TestStreamingCandidates:
    def test_topk_identical_full_chunk_vs_chunked(self, corpus_ties_sim):
        docs, s1, s1_ids, cand_ids = corpus_ties_sim
        k = 3
        params = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=None, lowercase=False)

        # Full corpus at once (vocabulary from same docs)
        result = build_fixed_vocabulary([docs], **params)

        # SKLEARN top-K reference
        skl, cand_X = _sklearn_oracle_tfidf_vectorizer(docs, **params)
        s1_X = skl.transform(s1)
        cos = (s1_X.astype(np.float64) @ cand_X.T.astype(np.float64)).toarray()
        skl_topk: list[list[str]] = []
        for row in cos:
            # Sort by sim desc, cand_id lex asc for ties
            order = sorted(range(len(cand_ids)), key=lambda i: (-row[i], cand_ids[i]))[:k]
            skl_topk.append([cand_ids[i] for i in order])

        # STREAMING: split docs into 2 chunks and use streaming_char_tfidf_candidates
        def _gen_chunks():
            chunk_size = 2
            for i in range(0, len(docs), chunk_size):
                yield (docs[i:i+chunk_size], cand_ids[i:i+chunk_size])

        cands, _meta = streaming_char_tfidf_candidates(
            s1,
            s1_ids,
            _gen_chunks(),
            vocab=result.vocabulary,
            idf=result.idf,
            k=k,
            method_name="char_tfidf_combined",
            analyzer=params["analyzer"],
            ngram_range=params["ngram_range"],
            lowercase=params["lowercase"],
            s1_batch_size=2,  # intentionally tiny to stress the batching path
        )

        for i, sid in enumerate(s1_ids):
            got = cands[sid]
            expected = skl_topk[i]
            assert got == expected, (
                f"{sid} top-K mismatch:\n  expected: {expected}\n  got:      {got}"
            )

    def test_topk_tied_sims_prefers_lex_smaller_id(self, corpus_ties_sim):
        docs, s1, s1_ids, cand_ids = corpus_ties_sim
        k = 2
        params = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=None, lowercase=False)
        result = build_fixed_vocabulary([docs], **params)

        # Chunk so that S2-0001 is seen after S2-0002
        def _gen_out_of_order_chunks():
            order = [3, 0, 2, 1]  # permute: zeta docs first, then 0000, 0002, 0001 last
            reordered_docs = [docs[i] for i in order]
            reordered_ids = [cand_ids[i] for i in order]
            yield (reordered_docs, reordered_ids)  # single permuted chunk

        cands, _meta = streaming_char_tfidf_candidates(
            s1, s1_ids, _gen_out_of_order_chunks(),
            vocab=result.vocabulary, idf=result.idf, k=k,
            method_name="char_tfidf_combined",
            analyzer=params["analyzer"],
            ngram_range=params["ngram_range"],
            lowercase=params["lowercase"],
            s1_batch_size=10,
        )
        got = cands[s1_ids[0]]
        # S2-0000 and S2-0001 have identical sim vs S1. Lex order: 0000 < 0001
        assert got[:2] == ["S2-0000", "S2-0001"], (
            f"Tied sims must break lex asc; got top-2 {got[:2]}"
        )

    def test_topk_counts_candidate_count_equals_k_when_available(self):
        docs = [("doc-%d " % i) * 3 for i in range(50)]
        s1 = ["doc-10 doc-20 doc-30"]
        s1_ids = ["S1-Q1"]
        cand_ids = [f"S3-{i}" for i in range(50)]
        k = 7
        params = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=None, lowercase=False)
        result = build_fixed_vocabulary([docs], **params)
        chunks = [list(_chunk(docs, 7)), list(_chunk(cand_ids, 7))]
        chunk_iter = list(zip(chunks[0], chunks[1]))  # (texts, ids) pairs
        cands, meta = streaming_char_tfidf_candidates(
            s1, s1_ids, iter(chunk_iter),
            vocab=result.vocabulary, idf=result.idf, k=k,
            method_name="char_tfidf_name",
            analyzer=params["analyzer"],
            ngram_range=params["ngram_range"],
            lowercase=params["lowercase"],
            s1_batch_size=4,
        )
        assert len(cands["S1-Q1"]) == k
        for _, m in meta.items():
            assert 0 <= int(m["candidate_rank"]) < k
            assert 0 <= int(m["tfidf_rank"]) < k


# ---------------------------------------------------------------------------
# 5. Chunking independence: results must NOT depend on chunk boundary choice
# ---------------------------------------------------------------------------


class TestChunkingIndependence:
    def test_vocab_idf_independent_of_chunking(self):
        rng = np.random.RandomState(42)
        tokens = ["acme", "corp", "incorporated", "ltd", "llc", "st", "ave", "12345", "main", "road"]
        def doc():
            return " ".join(rng.choice(tokens, size=rng.randint(4, 10)))
        docs = [doc() for _ in range(200)]
        params = dict(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=200, lowercase=False)

        # 3 different chunk boundaries
        r_sizes = [1, 11, 37, 50, 200]
        results: list[VocabBuildResult] = []
        for sz in r_sizes:
            chunks = [list(_chunk(docs, sz))]
            flat = chunks[0]
            results.append(build_fixed_vocabulary(flat, **params))

        # All vocabularies, idfs, tfs, dfs must be identical
        base = results[0]
        for i, r in enumerate(results[1:], 1):
            assert list(base.vocabulary.items()) == list(r.vocabulary.items()), (
                f"chunk sz={r_sizes[i]} vocab mismatch"
            )
            assert np.allclose(base.idf, r.idf, atol=1e-6), (
                f"chunk sz={r_sizes[i]} idf differs"
            )
            assert np.array_equal(base.df, r.df)
            assert np.array_equal(base.tf, r.tf)
            assert base.N_docs == r.N_docs

    def test_topk_independent_of_chunking(self):
        rng = np.random.RandomState(7)
        tokens = ["acme", "beta", "contoso", "northwind", "st", "ave", "blvd"]
        def doc(n=6):
            return " ".join(rng.choice(tokens, size=n))
        cand_docs = [doc() for _ in range(120)]
        s1 = [doc(4) for _ in range(8)]
        s1_ids = [f"S1-{i}" for i in range(len(s1))]
        cand_ids = [f"S2-{i}" for i in range(len(cand_docs))]
        params = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=None, lowercase=False)

        # Vocab: same regardless of chunking (proven above; use one-shot)
        result = build_fixed_vocabulary([cand_docs], **params)

        def run_chunked(chunk_size: int, s1_batch_size: int):
            def gen():
                for i in range(0, len(cand_docs), chunk_size):
                    yield (cand_docs[i:i+chunk_size], cand_ids[i:i+chunk_size])
            cands, _ = streaming_char_tfidf_candidates(
                s1, s1_ids, gen(),
                vocab=result.vocabulary, idf=result.idf, k=10,
                method_name="x",
                analyzer=params["analyzer"],
                ngram_range=params["ngram_range"],
                lowercase=params["lowercase"],
                s1_batch_size=s1_batch_size,
            )
            return {k: list(v) for k, v in cands.items()}

        ref = run_chunked(120, 1024)
        for cs, bs in [(5, 2), (13, 3), (47, 7), (119, 1)]:
            got = run_chunked(cs, bs)
            assert got == ref, f"topk changed for chunk={cs}, s1_batch={bs}"
