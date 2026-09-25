# Scalable retrieval architecture design

Scope: Candidate generation only. Matching (GBDT + threshold) is unchanged.
Goal: Execute exact blocking + char TF-IDF retrieval against the full ~1.7 M entity
test corpus without ever materializing the complete S2 or S3 DataFrames in RAM,
without ever forming a source_chunk × all_S1 similarity matrix, and while
retaining per-pair retrieval-channel flags needed by the matcher features and
diagnostics.

Not implemented yet. This is a design-only document.

## 0. CRITICAL CORRECTION (2026-09-25): sklearn max_features semantics

The original version of this document incorrectly claimed that `TfidfVectorizer`
`max_features` selects the top features by **document frequency**. This is
**wrong**.

Empirical verification against the installed sklearn version (run on this
project): given 2 documents with `zzz` appearing 600 times total across 2
documents vs 20 documents with `aaa` appearing 60 times total across 20
documents, `max_features=1` selects `zzz`. The correct rule is:

> **sklearn `max_features` selects the top-V features ordered by GLOBAL TERM
> FREQUENCY (sum of counts across the full corpus), not document frequency.**

Ties are broken deterministically by **feature / ngram lexicographic ascending
order** — this is the stable behavior of `Counter.most_common()` under CPython
for equal-count keys.

Every section below that referenced document-frequency ranking has been
corrected. All sklearn-equivalence claims must be re-verified against this
definition.

Reference semantics the streaming retriever must reproduce exactly:

| Parameter                     | Value                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `analyzer`                    | `char_wb`                                                                       |
| `ngram_range`                 | `(3, 5)`                                                                        |
| `min_df`                      | `2` (global document count, not fraction)                                       |
| `max_features` selection rule | top V by **global term-frequency** descending, tie-break by ngram lex ascending |
| `smooth_idf`                  | `True`                                                                          |
| `sublinear_tf`                | `False`                                                                         |
| `norm`                        | `l2` (applied to rows AFTER tf-idf weighting)                                   |
| `use_idf`                     | `True`                                                                          |
| `lowercase`                   | `False` (normalization is handled upstream by `_field_texts`)                   |
| `dtype`                       | `np.float32` for our implementation                                             |

IDF formula (sklearn default, smooth_idf=True, use_idf=True):

```
idf[t] = log((1 + N) / (1 + df[t])) + 1
```

where `N` = total number of documents in the source (S2 or S3 corpus
separately), and `df[t]` = number of source documents that contain ngram `t` at
least once.

TF-IDF per cell:

```
tfidf[d,t] = tf[d,t] * idf[t]
```

(i.e., raw term count × idf; NO 1+log(tf) sublinear scaling).

Final row representation:

```
row_norm[d] = sqrt(sum(tfidf[d,t]^2 over t))
X[d,t] = tfidf[d,t] / row_norm[d]   (zero row → unchanged zero row)
```

Cosine similarity between query q and candidate c is the dot product of their
L2-normalized rows.

## 1. Problem scale and why the current code will not fit

Current `char_tfidf_candidates()` in [blocking.py](file:///C:/Users/HP/Amazon_SC26/src/blocking.py#L303-L341):

1. `pool = pd.concat` / direct load of full S2 or S3 into a DataFrame.
2. `TfidfVectorizer.fit_transform(_field_texts(pool))` on the ENTIRE pool.
3. `S1_all @ Sx_all.T` via `_sparse_topk()` with `batch_size=256` on the S1 side.

Smoke (500 S1 × 20 k S2 × 20 k S3) already took **533 s loading + 59.5 s TF-IDF**.
At full scale (S1 ~ 550 k, S2 ~ 600 k, S3 ~ 550 k ≈ 1.7 M entities) keeping 6
views × 1.15 M pool rows of sparse TF-IDF in RAM is ~1.5–2 GB on top of the
source DataFrames, and the loader (already the dominant bottleneck) cannot grow
proportionally.

Requirements restated for the design:

- **Never materialize full S2 / S3.** Stream chunks; drop rows after use.
- **S1 in batches.** S1 itself can be kept in memory (it is the smaller reference
  set), but never multiply a chunk against the full S1 matrix at once.
- **No source_chunk × all_S1 matrix.** At 10 k S1 × 50 k chunk that is 500 M
  cells; even sparse this blows up.
- **Top-K only per S1, per channel.** Preserve per-S1 heap state across chunks.
- **Preserve exact + TF-IDF semantics.** Candidate selection order and channel
  flags must agree with `blocking.METHOD_FLAG` used by current pair features
  (`retrieved_by_exact_name/address`, `retrieved_by_char_name/address/combined`).
- **Minimize source scans.** Ideal: 2 passes per source (vocab fit + retrieval).
- **Realistic RAM / runtime / pass estimates.**
- **Sklearn equivalence where claimed.** If we produce different top-K orderings
  we silently change recall and invalidate the smoke calibration.

## 2. Approach A — Current sklearn TF-IDF + query batching (status quo)

| Dimension                             | Value                                                       |
| ------------------------------------- | ----------------------------------------------------------- |
| Full S2/S3 in memory?                 | **Yes.** (hard violation of the constraint)                 |
| S1 batches?                           | Yes, on the query side only (256 rows per batch)            |
| Source scans                          | 1 (because `fit_transform` over the full pool)              |
| Sklearn-equivalent?                   | **Exact.** It IS sklearn.                                   |
| Channel flags                         | Preserved via `_merge_meta` per `METHOD_FLAG`.              |
| RAM (retrieval only, full scale est.) | ~1.4 GB sparse matrices + ~240 MB DataFrames ≈ 1.7 GB peak. |
| Runtime (smoke, measured)             | 59.5 s TF-IDF + 533 s loading.                              |

**Verdict for full scale:** Violates the "never materialize full S2/S3" hard
constraint. RAM is borderline even if loading succeeded. Keep as the correctness
baseline; all other approaches must reproduce its candidate ordering on the
smoke subset to within a documented delta before graduating.

## 3. Approach B — Chunked sparse retrieval with fixed vocabulary

**Recommended approach.** Two phases, exactly 4 total source scans for S2+S3.

### Phase 1 — Vocabulary + IDF fit (1 scan per source)

Stream Sx in 50 k-row chunks. Per view (name / address / combined):

1. For each chunk, run `CountVectorizer(analyzer='char_wb',
ngram_range=(3,5), lowercase=False)` with `vocabulary=None` and NO
   `max_features` yet to obtain per-chunk sparse count matrix.
2. Aggregate GLOBALLY across all chunks:
   - `tf[t]` (global term-frequency counter): for each ngram string `t` in the
     vocabulary, sum the column sums of each chunk's count matrix.
   - `df[t]` (global document-frequency counter): for each chunk, compute
     presence per ngram per document (`chunk_mat > 0` → `sum(axis=0)`), then
     add that vector into the running global `df`.
   - `N_docs`: running integer += chunk.shape[0] each iteration.
3. After the last chunk:
   - Apply `min_df` filter: drop all ngrams with `df[t] < min_df`.
   - Keep the top `max_features` surviving ngrams by **global term-frequency
     `tf[t]` descending**. Ties break by **ngram lexicographic ascending**
     (matching `Counter.most_common` tie-break behavior as empirically
     verified).
   - Assign `vocabulary[t] = idx` in the preserved order.
4. Compute the smoothed IDF vector using sklearn's formula:
   `idf[t] = log((1 + N) / (1 + df[t])) + 1` where `N` = total rows seen in
   the source.
5. Persist `{vocabulary: {ngram: idx}, idf: np.ndarray, tf_sorted: [..],
df_sorted: [..], N_docs: int, analyzer, ngram_range, min_df, max_features}`
   to `artifacts/tfidf_vocab_{view}_{source}.npz/json`.
6. Exact blocking indexes are built in the **same** streaming pass — when each
   chunk is in memory compute the 6 exact keys and append to `BlockIndex`
   dicts.

**Trade-off on vocabulary size / exactness.** Because `char_wb` over 3–5 gram
can in principle produce up to (26+1+digits)^3 ≈ 29,800 distinct 3-grams and
scales polynomially for 4,5-grams, the full-ngram-space Counter over a 1.2 M
document source typically fits in RAM (<200 MB Python dict + numpy arrays for
aggregated counts). If this is ever shown to be too large, the exact method
requires a **two-pass deterministic alternative**:

- Pass 1a: stream all chunks, run `CountVectorizer` per chunk, collect ALL
  ngram keys seen, union them, write to a temporary sorted plain-text file of
  keys, dedupe by line sort (`sort -u` equivalent — use Python `sorted(set())`
  if fits, otherwise external sort via disk `heapq.merge` of chunk-keyed
  lists).
- Pass 1b: re-scan with the FULL fixed key list as vocabulary of size
  `V_potential`, accumulate global `tf` and `df` exactly, then apply `min_df` +
  `max_features` by tf descending as before. This trades 1 extra source pass
  for exact counting with bounded RAM O(V_potential_per_chunk); we will adopt
  it only if the single-pass Counter approach is empirically shown to exceed
  250 MB peak. For the current design we start with single-pass counting.

**RAM of Phase 1:** bounded by chunk size (50 k rows × 3 views × sparse mat ≈
a few hundred MB) plus the final global aggregated arrays (~V=30,000 after
min_df + max_features → ~0.5 MB for int32 arrays; the intermediate pre-filter
Counters for raw ngrams are the hot path, estimated < 200 MB peak for 1.2 M
documents).

### Phase 2 — Streaming transform + per-S1 top-K heaps (1 scan per source)

Precompute S1 once:

- For each view, run `TfidfVectorizer(vocabulary=fixed_vocab, ...).transform(S1_texts)`
  once and keep the 3 resulting sparse CSR matrices in memory. S1 is the
  reference set; it is small enough.

Per source (S2, S3), per chunk:

1. Load chunk rows (50 k). Keep `entity_id` list only in a Python list; drop
   the source DataFrame columns that are not needed for exact indexing once we
   built those in Phase 1.
2. For each view:
   - Build chunk term-count matrix via `CountVectorizer(vocabulary=fixed_vocab)`.
   - Apply IDF elementwise: `tfidf_chunk = chunk_counts.multiply(idf_vec)` (scipy
     sparse elementwise multiply, O(nnz)).
   - Apply L2 row normalization (scipy `normalize(..., norm='l2', copy=False)`).
   - Iterate S1 in **batches of B = 4096 queries** (tunable). For each batch:
     - Compute `S1_batch_sparse @ chunk_tfidf.T` → sparse similarities of shape
       (B, chunk_rows). Note: `S1_batch` is already L2-normalized sklearn output.
     - For each query row in the batch, extract non-zero similarities and feed
       them to a per-(S1, channel) top-K heap.
3. After the last chunk, each (S1, view) heap holds the global top-K matches
   for that TF-IDF channel because we compare every candidate score against
   the heap and only keep the K largest.

Exact blocking is handled in the same streaming pass because we built
`BlockIndex` structures in Phase 1 — exact lookups are O(1) per S1 key and do
not need any chunk scanning.

### Per-S1 state (top-K heaps)

Replace the full candidate list with bounded heaps:

- Structure: `dict[s1_id] -> {"char_name": heap, "char_address": heap,
"char_combined": heap, "exact": set}`.
- Each heap stores `(similarity, candidate_id, candidate_rank_within_chunk)`
  tuples of size K. Python `heapq.nsmallest`-style min-heap is used so the
  smallest of the top-K entries sits at position 0 and is evicted first.
- After all chunks: heap → sorted list → assign monotonically increasing
  `tfidf_rank` and `candidate_rank` exactly like `_merge_meta` does today.
- Final union with exact matches: same logic as `union_exact_with_tfidf_k()` +
  `cap_preferring_exact()`. Channel flags are set by re-running `_merge_meta`
  with the final sorted candidate list — no semantic change.

### Sklearn equivalence proof (Approach B vs A)

| sklearn detail                            | Approach B matches?                                                                                                                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analyzer='char_wb'`                      | Yes, same CountVectorizer per chunk.                                                                                                                                                |
| `ngram_range=(3,5)`                       | Yes.                                                                                                                                                                                |
| `min_df=2` global                         | Yes — after Phase 1 global df aggregated across all chunks, remove vocab entries with df < min_df BEFORE idf computation.                                                           |
| `max_features` selection rule             | Yes — top V by **global term-frequency `tf[t]`** descending (documented above as empirically verified against sklearn). Tie-break: ngram lex ascending (Counter.most_common style). |
| `smooth_idf=True` (default)               | Yes, same formula: `log((1+N)/(1+df)) + 1`. Verified vs sklearn source on 2 synthetic corpora.                                                                                      |
| `sublinear_tf=False` (default)            | Yes, raw TF counts not log-scaled.                                                                                                                                                  |
| `use_idf=True`                            | Yes, idf multiplication happens before normalization.                                                                                                                               |
| L2 row normalization applied after TF-IDF | Yes, scipy `normalize(norm='l2')` applied after IDF multiply. Identical semantics; zero rows remain zero rows.                                                                      |
| Cosine similarity                         | Yes = normalized dot product of two L2-normalized vectors.                                                                                                                          |
| Top-K selection by similarity             | Yes, heap-based same-tie-break keeps ordering identical: equal sims break by candidate_id lex ascending to match deterministic `np.argsort` on equal floats.                        |

Minor difference that does **not** affect candidate selection: if two candidates
tie on similarity to the 10th decimal, the chunk processing order in B may
break the tie differently than A's within-chunk argpartition unless we add the
explicit `(sim, -id_ord, cand_id)` heap key. We therefore enforce a
deterministic tie-break: for equal similarity, prefer the candidate with the
**lexicographically smaller** entity ID. This behavior must match the
`_sparse_topk` refactored tie-break used as the oracle in Approach A.

### Full-scale estimates for Approach B

Assumptions: S1 = 550 k, S2 = 600 k, S3 = 550 k, V = 30 k features, chunk =
50 k rows, K = 50, 3 views per source.

**RAM upper bound (Phase 2 peak):**

- S1 query matrices (3 views × 550 k rows × ~100 nnz/row × 8 bytes) →
  ~1.3 GB. **Reduce by batching S1 queries:** only keep B = 4096 S1 rows in
  sparse form at once → ~10 MB for query mat. The per-S1 heaps stay resident.
- Per-S1 heap state: 6 channels × 550 k S1 × K=50 entries × (4 byte score +
  10 byte id) ≈ 550k × 300 × 14 bytes ≈ **2.3 GB.** Too big.
- **Mitigation:** Do NOT keep per-channel heaps. Maintain ONE heap per S1
  across ALL 6 channels. When a heap entry is popped for output, recover the
  channel flags by re-running the exact index lookups (cheap) + re-querying the
  chunk that produced the TF-IDF match (expensive). Alternative: record the
  "channel origin" per heap entry as a 1-byte enum. Then per-S1 heap = 550k ×
  K=50 × (4 + 10 + 1) bytes ≈ **412 MB** for 6 channels combined. Feasible.
- Chunk TF-IDF matrices: 50 k rows × ~200 nnz × 4 bytes × 3 views ≈ **120 MB**
  per chunk. Freed after chunk.

Final RAM estimate: **~600–800 MB** retrieval-only peak. Fits on a 16 GB
laptop alongside the GBDT (≤ 1 GB) with headroom.

**Runtime estimate (full scale):**

- Phase 1 vocab fit: 2 × (S2 scan + S3 scan) at ~500 k rows/min (measured in
  smoke at 20 k / ~2.5 s → ~480 k/min) → ~2.5 min per pass × 2 passes for
  vocab → **~10 min.**
- Phase 2 retrieval: 2 more passes. Each chunk: 3 × CountVectorizer (~1.5 s)
  - IDF+norm (~0.5 s) + S1_batches × sparse matmul: ~(550k/4096) = 135 batches
    × (B=4096 × 50k matmul ~ 0.2 s) → ~27 s per chunk × 23 chunks = ~10 min per
    source × 2 sources = **~20 min.**
- Exact blocking: built for free during Phase 1 lookups, <1 min.
- **Total retrieval wall-clock: ~30 min.** Acceptable; ~6× the smoke (646 s)
  against ~50× more data (reasonable scaling with streaming).
- **Total source scans: 4** (2 vocab build + 2 retrieval), each scan visits
  S2 + S3 exactly once.

### Channel metadata preservation

After Phase 2 heaps are flattened to candidate lists, call `_merge_meta()`
exactly as in the current code. Because the per-heap entry records which of
the 6 retrieval channels produced it (1 byte enum per heap entry), the mapping
to `METHOD_FLAG` in `blocking.py` is one dict lookup per candidate. The 5
retrieval flags used by pair features (`retrieved_by_exact_name`,
`retrieved_by_exact_address`, `retrieved_by_char_name`,
`retrieved_by_char_address`, `retrieved_by_char_combined`) are populated
identically to today. The `retrieval_channel_contribution()` diagnostic in
[error_analysis.py](file:///C:/Users/HP/Amazon_SC26/src/error_analysis.py#L174-L275)
therefore works unchanged.

### Transition / validation strategy before full scale

1. Implement Approach B on the smoke subset **side by side** with A.
2. Assert `set(candidates_A[s1]) == set(candidates_B[s1])` for every S1; if
   V=20000 and K=10 on the 20 k smoke the match rate should be 100% (up to
   documented tie-break policy).
3. Measure per-channel overlap:
   `recall(recovered_by_A_char_name, recovered_by_B_char_name) ≥ 0.999`.
4. Only then remove Approach A from the hot path (keep it behind a flag as the
   oracle).

## 4. Approach C — Inverted character n-gram retrieval (postings lists)

Core idea: build a classical inverted index. For each ngram term t in the
final vocabulary of size V, store a postings list of
`[(doc_id, term_freq_in_doc), ...]`. Per S1 query: iterate the postings of
each query ngram, accumulate partial cosine scores, and keep top K via a heap.

| Dimension             | Value                                                                                                                                                                            |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full S2/S3 in memory? | **No.** Only postings + vocabulary.                                                                                                                                              |
| S1 batches?           | Not required; index is random-access by term.                                                                                                                                    |
| Source scans          | 1 for build. Zero additional scans after.                                                                                                                                        |
| Sklearn-equivalent?   | **Approximate** unless we also store per-doc L2 norm and explicitly divide; simple accumulator-based WAND-style top-K approximates cosine and ordering can drift on tied scores. |
| Channel flags         | Same enum trick as B; straightforward to attach origin.                                                                                                                          |
| RAM (full scale est.) | 1.2 M docs × avg 150 kept-postings/doc × ~5 bytes varint → ~900 MB after aggressive compaction. Without compaction ~4.8 GB → hard fail.                                          |
| Runtime               | ~3–5 min build, ~30 s–2 min queries; fastest on repeated queries but single-pass fastest of the four.                                                                            |
| Implementation risk   | High. Correct L2-normalized cosine from posting lists requires per-doc norm storage; skipping it causes subtle ranking drift vs sklearn.                                         |

**Verdict:** Rejected for now. Approach B offers ~30 min total with
**guaranteed** sklearn equivalence and ~600–800 MB RAM, which is enough for
the problem size. Revisit C only if: (a) 30 min retrieval is later shown to
be too slow on the leaderboard timeline, or (b) RAM pressure on the actual
submission VM is tighter than 16 GB.

## 5. Approach D — Chunked `sklearn.NearestNeighbors` estimator wrapper

Wrapper that mimics the sklearn estimator API: `fit(X)` ingests chunks via
`partial_fit`-style calls (NearestNeighbors does not actually support
partial_fit, so we implement it ourselves) and `kneighbors(X)` streams.
Under the hood this IS Approach B but wrapped in a class with
`fit(chunk_iter)` / `kneighbors_batch(queries, return_distance=True)` methods
so the rest of `blocking.py` treats the sparse retriever like a drop-in model.

| Dimension               | Value                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Materialize?            | No. Same as B.                                                                                  |
| Scans                   | Same as B = 4 total.                                                                            |
| Sklearn-equivalent?     | **Exact if the inner similarity is B.** The wrapper is API-shape only.                          |
| Channel flags           | Same as B.                                                                                      |
| RAM                     | Identical to B.                                                                                 |
| Additional value over B | None computationally; only reduces LOC in `blocking.py` by ~80 lines and adds a unit-test seam. |

**Verdict:** Nice-to-have refactor on top of B but not a distinct algorithm.
Implement after B is validated, only if code clarity warrants the extra file.

## 6. Decision table and migration plan

| #   | Item                                                                                                                                                                                                                                                                                                                                                                                                     | Owner / status |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| 1   | **Approach B selected** as the full-scale retrieval target.                                                                                                                                                                                                                                                                                                                                              | Design done.   |
| 2   | Approach A remains the smoke oracle; no removal until 100% set-equality on smoke.                                                                                                                                                                                                                                                                                                                        | Later.         |
| 3   | Approach D (estimator wrapper) is deferred to cleanup after B lands.                                                                                                                                                                                                                                                                                                                                     | Deferred.      |
| 4   | Approach C (inverted postings) deferred. Documented only for the record.                                                                                                                                                                                                                                                                                                                                 | Deferred.      |
| 5   | First implementation step: add `build_fixed_vocabulary(text_chunk_iterable, *, analyzer, ngram_range, min_df, max_features) -> VocabBuildResult(vocabulary, idf_arr, tf_arr, df_arr, N_docs)` to `src/streaming_tfidf.py` + tests that assert vocabulary equals sklearn TfidfVectorizer on a 5 k-row synthetic corpus (covering rare/bursty, singleton, tie, and min_df/max_features boundary cases).    | Not done.      |
| 6   | Second step: add `CharTFIDFStreamingRetriever(vocab, idf, k, method_name, analyzer='char_wb', ngram_range=(3,5))` class with: `partial_fit_source_chunk(texts, entity_ids)` (builds source chunk tf-idf + merges per-S1 top-K heaps) and `finalize_candidates(s1_ids) -> (CandidateMap, PairMeta)`. Passes the same smoke unit tests that `char_tfidf_candidates` does today against the sklearn oracle. | Not done.      |
| 7   | Third step: wrap the exact-blocking build in a streaming pass so the BlockIndexes are built during vocab scan without re-reading S2/S3.                                                                                                                                                                                                                                                                  | Not done.      |
| 8   | Fourth step: verify 100% candidate set overlap + 99.9% channel flag agreement A vs B on smoke with identical seeds.                                                                                                                                                                                                                                                                                      | Not done.      |

## 7. What explicitly this design does NOT change

- GBDT matcher (LightGBM, 200 trees, 31 leaves) — untouched.
- Threshold sweep grid + macro F0.5 / singleton handling — untouched.
- Normalization views: `normalize_name`, `normalize_address`, `compact_alnum`,
  legal suffixes, address abbreviations — all unchanged; the design uses the
  same outputs from `_field_texts()`.
- Exact blocking 6-method set (`EXACT_METHODS` in [blocking.py](file:///C:/Users/HP/Amazon_SC26/src/blocking.py#L26-L33)) —
  the only change is building the BlockIndexes during Phase 1 instead of on a
  fully materialized DataFrame; semantics are identical.
- Pair feature names and retrieval channel features — the five
  `retrieved_by_*` flags + `candidate_rank` + `tfidf_rank` are produced in the
  same order and range so the already-trained GBDT smoke model scores pairs
  identically.
- Output format of `candidate_pairs.tsv` and `matching_results.tsv` —
  unchanged. Validator passes without modification.
