# Conflict Pipeline — First Run Runbook

Every module of the conflict pass is implemented but none has executed against a
live Neo4j or a real LLM endpoint. This runbook walks the first end-to-end run
with a checkpoint after each phase, so a failure is localised to one layer
instead of surfacing as "nothing happened."

Work through the phases in order. Do not skip a checkpoint — a silent failure in
an early phase looks identical to "no conflicts found" at the end.

---

## Phase 0 — Preparation

**Database safety.** If the Neo4j instance is shared with the KG team, do **not**
wipe it. Start a local instance from the repo's `docker-compose.yml` instead and
point `.env` at it.

**Model.** `run_classification.py` calls Ollama (`CLASSIFICATION_MODEL_NAME`,
default `qwen3:4b`). Confirm that model is actually pulled on the host
`CLASSIFICATION_MODEL_ENDPOINT` points at (`ollama list`) before starting —
discovering this at phase 4, after the earlier phases have already run, wastes
a cycle. Same for `GATE_NLI_MODEL` at phase 3.

**Wipe.**

```cypher
MATCH (n) DETACH DELETE n
```

---

## Phase 1 — Ingest

Ingest **one chunking folder only** — `chunks_data/maruf/` **or**
`chunks_data/linus/`, never both.

The two folders contain largely the same theses chunked by different methods, so
each physical paper appears under two different `doc_id`s. Ingesting both makes
the pipeline report conflicts between a thesis and itself. Those artifacts will
dominate the output and bury any real signal.

```bash
python main.py
```

### Checkpoint 1a — Description document scoping (CRITICAL)

This is the single most important check in the runbook. Everything downstream
depends on it.

```cypher
MATCH (d:Description)
WITH d.topicName AS topic, d.typeName AS type,
     count(*) AS n, collect(DISTINCT d.source_id) AS sources
WHERE n > 1
RETURN topic, type, n, sources ORDER BY n DESC LIMIT 20
```

**Pass:** at least one row where `sources` holds two or more distinct values.

**Fail (empty result):** either Descriptions are still merging across documents,
or the two ingested documents share no Topic × Type. Check the second
possibility first:

```cypher
MATCH (d:Description) RETURN d.source_id, count(*) AS n
```

Two sources with reasonable counts but no shared Topic × Type means the corpus
overlap is too thin — pick different documents. Only one source, or a merge, means
the doc-scoping fix did not survive to Neo4j.

### Checkpoint 1b — Structural integrity

```cypher
// id format — must be empty
MATCH (d:Description) WHERE size(split(d.id, '|')) <> 2 RETURN d.id LIMIT 20;

// source_id complete — must be 0
MATCH (d:Description) WHERE d.source_id IS NULL RETURN count(d);

// ghost nodes from id rewriting — must be empty
MATCH (d:Description) WHERE d.text IS NULL RETURN d.id LIMIT 20;

// source_id resolves to a Source — must be empty
MATCH (d:Description)
WHERE NOT EXISTS { MATCH (s:Source) WHERE s.id = d.source_id }
RETURN DISTINCT d.source_id LIMIT 20;
```

A non-empty ghost-node result means a relationship endpoint was not renamed
alongside its node.

### Checkpoint 1c — STEP D is genuinely off

```cypher
MATCH (c:Contradiction) RETURN count(c)
```

**Must return 0.** Any Contradiction here was produced during construction,
which should no longer be possible.

### Checkpoint 1d — Temporal coverage

```cypher
MATCH (s:Source) RETURN s.id, s.date_raw, s.year ORDER BY s.id
```

Record three numbers: how many Sources have a parsed `year`, how many have
`date_raw` but no `year`, how many have neither.

If **zero** Sources have a `year`, supersession can never fire and branch 2 is
dormant. That is not a blocker for the rest of the run — proceed, then build the
manual `doc_id → year` override file afterwards.

### Checkpoint 1e — Volume baseline

```cypher
MATCH (d:Description)
WHERE d.typeName IN ['Result','Metrics Evaluation','Conclusion']
RETURN d.typeName, count(*) AS n ORDER BY n DESC
```

This is the pool blocking will draw from. Expect it to be small. It sets the
ceiling on how many candidates can possibly exist.

### Checkpoint 1f — Ingest idempotency

Record `MATCH (d:Description) RETURN count(d)`, re-run `python main.py` on the
same input, and count again. **The number must not change.** Growth means
`doc_id` is not stable across runs.

---

## Phase 2 — Blocking

```bash
python run_blocking.py
```

### Checkpoint 2

Report:

- Descriptions embedded, broken down by `typeName`
- candidates from S1 only, S2 only, and both
- intra-document vs cross-document split
- `similarity` distribution for kNN-only pairs (min / median / max)

Then inspect 10 random `knn`-only pairs and 10 random `exact` pairs by hand. Are
they plausibly related?

**Tuning signals:**

| Observation | Reading |
|---|---|
| kNN-only median similarity low, pairs look unrelated | `k` too high, or a similarity floor is needed |
| All similarities high | `k` could be raised for better recall |
| Nearly all candidates intra-document | corpus topics barely overlap; the pass cannot show its value yet |
| Zero candidates | check checkpoint 1a passed, then check the vector index exists |

Re-run `run_blocking.py` and confirm the candidate count is unchanged.

---

## Phase 3 — Gates

```bash
python run_gates.py
```

### Checkpoint 3

Report:

- total candidates evaluated
- rejections per gate (G1, G2, G3, G5) and overall pass rate
- `nli_confidence` distribution for rejected vs passed
- genuine `unclear` verdicts vs error-path `unclear` (these must be
  distinguishable — see gate amendment 2)
- LLM calls on first run vs second run (second must be **0**)

Inspect 10 pairs rejected by G5 and 10 that passed everything.

**Tuning signals:**

| Pass rate | Reading |
|---|---|
| above ~80% | gates too loose; almost nothing is being filtered |
| below ~20% | likely too strict; real conflicts are being discarded |
| high `nli_neutral` rejection count | the NLI prompt is still treating "different studies" as grounds for neutral — the amendment did not fully take |

The last row matters most. Opposing findings from different methods or datasets
are the largest conflict category; if G5 rejects them as `neutral`, the pipeline
silently drops exactly what it exists to find.

---

## Phase 4 — Classification

```bash
python run_classification.py
```

### Checkpoint 4a — Output breakdown

Report:

- clusters classified, by `resolution_type`
- `supersedes` edges written, by `basis`
- pairs recorded as `insufficient_evidence`
- clusters where `unresolved` was overridden to `insufficient_evidence`
  (amendment 4)
- LLM calls on first run vs second run (second must be **0**)

### Checkpoint 4b — Invariants (all must return empty)

```cypher
// wrong endpoint type on supersedes
MATCH (a)-[r:supersedes]->(b)
WHERE NOT a.typeName IN $allowed OR NOT b.typeName IN $allowed
RETURN a.id, b.id;

// anti-cycle
MATCH (a)-[:supersedes]->(b)-[:supersedes]->(a) RETURN a.id, b.id;

// self-loop
MATCH (a)-[:supersedes]->(a) RETURN a.id;

// singleton Contradiction
MATCH (c:Contradiction)
WHERE COUNT { (c)<-[:has_contradiction]-() } < 2
RETURN c.id;

// scope_conditions present iff scope_difference
MATCH (c:Contradiction)
WHERE (c.resolution_type = 'scope_difference' AND c.scope_conditions IS NULL)
   OR (c.resolution_type <> 'scope_difference' AND c.scope_conditions IS NOT NULL)
RETURN c.id, c.resolution_type;

// duplicate participant sets (random node ids mean no id check catches this)
MATCH (c:Contradiction)<-[:has_contradiction]-(d:Description)
WITH c, collect(DISTINCT d.id) AS parts
WITH apoc.coll.sort(parts) AS sorted_parts, collect(c.id) AS cs
WHERE size(cs) > 1
RETURN sorted_parts, cs;
```

### Checkpoint 4c — Manual quality read

This is the only check that tells you whether the output is *useful* rather than
merely *well-formed*.

Read every `scope_difference` cluster's `scope_conditions`. Does it name a real,
specific difference — a method, a dataset, a population — or is it vague filler?

Read every `supersedes` edge's `basis` and `reason`. Is the direction right? Is
the correction real, or did the model treat "newer" as sufficient?

Read every `summary`. Does it name the specific conflicting detail from each
side, as the spec requires, or does it fall back to "these disagree"?

---

## Phase 5 — Full idempotency

Re-run all three passes:

```bash
python run_blocking.py && python run_gates.py && python run_classification.py
```

Expected: zero LLM calls across all three, no new nodes or edges, no changed
candidate counts.

---

## What counts as success on the first run

**Not** "N conflicts found."

The corpus is small — a handful of theses on one research domain, yielding a few
dozen assertion-bearing Descriptions. Finding **zero** genuine conflicts is a
plausible and acceptable outcome. It says something about the corpus, not about
the pipeline.

What is being validated here is that the machinery runs:

- checkpoint 1a passes — Descriptions are document-scoped
- indexes are created, embeddings are populated
- candidates are generated, gates issue verdicts, the classifier writes shapes
- every invariant in checkpoint 4b is empty
- every phase is idempotent on a second run

If all of that holds, the pipeline is correct and the next constraint is corpus
size, not code.

---

## After the run

| Task | Notes |
|---|---|
| Clean up dead constants | `ALLOWED_CONTRADICTION_LEVEL` removed, `SUPERSEDES_ENDPOINT_TYPE` replaced by `ALLOWED_TYPES`. Only once nothing references them |
| Manual `doc_id → year` override | Only if checkpoint 1d showed poor coverage. Eleven lines of CSV, and branch 2 becomes live |
| Threshold tuning | Driven by the phase 2 and phase 3 numbers. Do not tune before seeing them |
| Clean the superseded doc sections | `README.md` and `PROJECT_CONTEXT.md` still describe the old schema under a warning callout |

Deferred to v2, per `docs/conflict_pipeline.md` §7: the IE/IM methodology
hierarchy for auto-resolving `unresolved`, the hedging gate, the `status`
lifecycle, and tracing back to original chunk text.