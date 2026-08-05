# Conflict Pipeline — Detection, Classification & Output Ontology

**Status:** authoritative.
**Supersedes:** `docs/conflict_ontology.md`, which originated from a
miscommunication with the KG development team and is void. Any rule about the
`Contradiction` node found elsewhere in the codebase or its docs — including
constants in `graph_model.py` — is superseded by this document.

This spec covers the **on-demand whole-KB conflict pass**: a separate stage that
runs after graph construction, reads the entire knowledge graph, and writes back
conflict structures.

---

## 1. Scope

### What this pass does

Finds pairs of claims that oppose each other across the whole knowledge base,
classifies why they oppose, and records the outcome in the graph.

### What it does not do

- It does not run during ingestion. It is invoked separately, on demand.
- It does not modify `Description`, `Topic`, `Type`, `Source`, or `Agent` nodes.
- It does not judge which claim is factually correct, except in the narrow
  supersession case where one claim explicitly corrects another.
- It does not require human review. Output is fully automatic.

### Sole ownership of `Contradiction`

This pass is the **only** producer of `Contradiction` nodes and
`has_contradiction` edges.

Per-chunk contradiction detection during construction (STEP D of the extraction
prompt) is **disabled**. It operated on a single chunk, could not see across
documents, frequently misread a paper's citation of prior work as a conflict,
and would now emit nodes lacking every property defined below.

---

## 2. Stages

```
[prerequisites in builder]
        │
        ▼
   BLOCKING  ── candidate pairs ──▶  SQLite
        │
        ▼
    GATES    ── pass / reject ────▶  SQLite
        │
        ▼
 CLASSIFICATION ──────────────────▶  Neo4j
```

### 2.0 Prerequisites (implemented in the builder, not this pass)

- `Description.id` is document-scoped: `content_id|canon_source`, with a
  `source_id` property. Without this, two documents covering the same
  Topic × Type collapse into one node and blocking can never produce a pair.
- `Source` carries `date_raw` (verbatim) and `year` (string, parsed).
  `year` is stored as a string because `_Node.properties` is typed
  `Dict[str, str]`. **Cypher comparisons must use `toInteger(s.year)`.**

### 2.1 Blocking — candidate generation

Union of two strategies:

- **S1 (exact)** — same `topicName` AND same `typeName`.
- **S2 (kNN)** — same `typeName`, and one is within the top-k nearest
  neighbours of the other by `Description` embedding.

S2 exists because Topic deduplication in the builder is deliberately
string-based (`difflib`, 0.92) and not semantic. "Digital Twin" and "Digital
Twin System" become separate Topic nodes; S1 alone would never compare them.

Rules:

- Type must match exactly, and must be in `ALLOWED_TYPES` (§3.1).
- Intra-document pairs are **kept**. Contradictions between distant chapters of
  one document are a core reason this pass exists.
- No self-pairs. Unordered pairs are normalised and deduplicated.
- Idempotent: re-running produces no duplicate rows.

Candidates are recorded with `strategy` ∈ {`exact`, `knn`, `both`}. A pair found
by both strategies is stronger evidence than one found only by kNN.

### 2.2 Gates — cheap rejection

Ordered, short-circuiting, cheapest first. Verdicts are written back to the
candidate row; **rows are never deleted**.

| Gate | Rejects when |
|---|---|
| G1 | Either text is empty, whitespace-only, or below `min_text_length` |
| G2 | `topicName` is in the generic-topic blocklist |
| G3 | Same document AND text similarity above threshold (extraction artifact) |
| G4 | *(skipped in v1 — see §7)* |
| G5 | NLI verdict is `entailment` or `neutral` above confidence threshold |

**Governing principle — fail open.** Every gate errs toward keeping a pair when
uncertain. A false rejection is invisible: we never see the conflict we dropped.
A false pass costs one LLM call downstream. The asymmetry is severe.

G5 must not treat "different studies" as grounds for `neutral`. Opposing
findings from different methods, datasets, or populations are the single largest
conflict category; deciding whether those differences reconcile the opposition
is classification's job, not the gate's.

### 2.3 Classification and write-back

Covered in §4.

---

## 3. Output ontology

### 3.1 `ALLOWED_TYPES`

One constant, used by both blocking and supersession endpoint validation:

```
Result, Metrics Evaluation, Conclusion, Decision, Progress Update
```

These are the assertion-bearing types. Descriptions of type `Background`,
`Method`, `Dataset`, `Theoretical Basis`, `Existing Research`, etc. describe what
was done or assumed, not what was found, and are never conflict participants.

### 3.2 Node `Contradiction`

Reified conflict. Two or more claims that oppose each other and all still stand.

| Property | Required | Content |
|---|---|---|
| `summary` | yes | Synthesis. At least two sentences. **Must name the specific conflicting detail from every side** — a number, an outcome, a direction. "These disagree" is forbidden. |
| `resolution_type` | yes | `scope_difference` \| `known_controversy` \| `unresolved` |
| `scope_conditions` | no | Bridge context. Filled only for `scope_difference`. States the condition under which each side holds: *"Under discrete-event simulation → A; under survey of firms → B."* |
| `confidence` | yes | Float 0–1, classifier's confidence in `resolution_type` |
| `participants_hash` | yes | Hash of the sorted participant `Description` ids. Drives staleness detection (§5) |
| `generated_by` | yes | Model identifier + prompt version |
| `generated_at` | yes | ISO timestamp |
| `pipeline_version` | yes | Version stamp for re-evaluation |
| `evidence_used` | no | Which fields drove the conclusion — e.g. `method`, `dataset`, `population`, `year`, `controversy_marker`. Auditability |

There is deliberately **no `level` property**. "How hard is the collision" was
found to carry no operational value; `resolution_type` answers the question that
matters — why they collide and what to do about it.

There is deliberately **no `status` property in v1**. See §5 for why, and for
what replaces it.

### 3.3 Edge `HAS_CONTRADICTION`

`Description → Contradiction`.

No required properties.

| Property | Required | Content |
|---|---|---|
| `position` | no | Short summary of this participant's stance. Only meaningful when `resolution_type = known_controversy` with three or more participants, where the node otherwise records that N claims conflict without recording who sits on which side |

**Cardinality: at least two.** A Contradiction with one participant is
meaningless. A three-way conflict is **one node with three edges**, never three
pairwise structures — otherwise six claims produce fifteen structures and every
new paper forces regeneration of all of them.

### 3.4 Edge `SUPERSEDES`

`Description → Description`. Edge-only, no node: a superseded claim is an
update, not a standing conflict, and there is nothing to synthesise.

**Direction:** `source = newer/correct`, `target = older/replaced`.
Read as "(A) supersedes (B)".

| Property | Required | Content |
|---|---|---|
| `basis` | yes | Which signal triggered the decision — `explicit_correction`, `citing_contrast`, `claimed_improvement`, `retraction`. **This is the debugging handle.** When supersedes edges turn out wrong, `basis` identifies which detection signal is unreliable |
| `reason` | no | Short human-readable justification |
| `confidence` | yes | Float 0–1 |
| `pipeline_version` | yes | Version stamp |
| `generated_at` | yes | ISO timestamp |

**Constraints:**

1. Both endpoints are `Description` with `typeName` in `ALLOWED_TYPES`.
2. No self-loop (`source != target`).
3. **Anti-cycle.** `A supersedes B` and `B supersedes A` must never coexist.
   That situation is a Contradiction, not a supersession. The classifier picks
   exactly one relationship per pair, never both.
4. Both sides must have a parsed `year` on their `Source`, and the years must
   differ. Without that, direction cannot be justified — see §4.1.

### 3.5 Casing: two layers, one rule

Two conventions coexist by design. Mixing them up is the single most likely
way to reintroduce a silent bug in this pass — it happened once already, in
`src/conflict/classifier.py`, and every check that should have caught it
(§8's invariants) passed anyway, because a query naming the wrong-case
relationship type doesn't error, it just matches nothing.

- **Pre-write** — the extraction prompt, `sanitize_graph`,
  `_FIXED_RELATION_DIRS` in `graph_model.py`, and anything else feeding
  `Neo4jGraph.add_graph_documents`: relationship types are **lowercase**
  snake_case (`has_description`, `has_contradiction`, `relates_to`, ...). This
  is the ontology's own naming convention and is correct as written.
- **Post-write** — this pass's raw Cypher, and any other code that reads from
  or writes directly to Neo4j, bypassing `add_graph_documents`: relationship
  types are **UPPERCASE** (`SUPERSEDES`, `HAS_CONTRADICTION`).

The split isn't a style choice, it's forced by `langchain_neo4j`:
`Neo4jGraph.add_graph_documents` uppercases every relationship type on write
(`el.type.replace(" ", "_").upper()`, `langchain_neo4j/graphs/neo4j_graph.py:667`)
but leaves node LABELS untouched. So `MATCH (d:Description)` is correct
everywhere, unchanged — only relationship types split by layer.

Post-write constants live in `src/conflict/config.py`
(`SUPERSEDES_TYPE`, `HAS_CONTRADICTION_TYPE`) — deliberately **not** derived
from `graph_model.py`'s pre-write constants by `.upper()`-ing them at the call
site. The two layers must stay visibly, intentionally separate; one silently
derived from the other is how this bug happens again.

---

## 4. Classification decision rules

Input: a candidate pair that passed all gates.

### 4.1 Step 1 — supersession test (pairwise)

Applies only if **both** sides have a parsed `year` and the years differ.
If not, skip to step 2.

Ask: *is the newer claim a correction or update of the older, such that the
older is now wrong?*

Accepted evidence:

- Explicit correction language — "contrary to prior findings", "we revise",
  "earlier work assumed", "has since been shown"
- The newer work cites the older, and the claim appears in a contrastive or
  corrective context
- The newer work explicitly claims a methodological improvement over the older
- Retraction or erratum on the older

**A year difference alone is never sufficient.** Two independent studies
published in different years are not a supersession; they are two studies.

**Evidence-level guard:** supersession is rejected if the newer claim rests on
markedly weaker evidence than the older. A single case study does not supersede
a meta-analysis merely by being newer.

→ Yes: emit `supersedes`, record `basis`. Done, no node.
→ No: continue to step 2.

This step has exactly these two outcomes — there is no "insufficient evidence"
branch here. If the model cannot confidently judge supersession, that is a
"No": the pair still deserves a fair contradiction assessment in step 2. §4.4's
insufficient-evidence path applies to cluster classification (§4.3) only.

### 4.2 Step 2 — clustering

Supersession is pairwise; contradiction is not.

Take all pairs not resolved as supersession and compute connected components
over them. Each component becomes **one** `Contradiction` node. A component of
size two is the ordinary pairwise case and needs no special handling.

Classification then runs **once per cluster**, not once per pair.

### 4.3 Step 3 — cluster classification

In order:

**a. Scope difference.** Can the opposition be explained by a difference in
method, dataset, population, setting, unit of analysis, measurement, or time
period?

Assemble that context from the same document: for each participant, fetch the
`Description`s sharing its `source_id` whose `typeName` is `Method`, `Dataset`,
or `Experiment`.

→ `resolution_type = scope_difference`, fill `scope_conditions`.

**b. Known controversy.** Does any participant's text explicitly acknowledge the
topic as contested — "controversial", "paradoxical", "conflicting evidence",
"remains debated"?

→ `resolution_type = known_controversy`.

**c. Unresolved.** Sufficient context was retrieved, scope was checked, and
nothing explains the opposition.

→ `resolution_type = unresolved`.

**Enforced precondition — scope must have been checkable.** `unresolved`
asserts "nothing explains this"; it must not be used to mean "we could not
look." If zero participants have any Method/Dataset/Experiment context
available, scope was not checkable at all, and `unresolved` is not a valid
output — only `known_controversy` (needs only the claim texts) or
`insufficient_evidence` may be returned in that case. If some but not all
participants have context, `unresolved` remains valid but `evidence_used` must
record the partial coverage and `confidence` must be lowered accordingly. This
is enforced by the pipeline itself, not left to the prompt alone.

**Precedence note.** When both (a) and (b) apply, `scope_difference` wins — it
is the more actionable finding. But the controversy marker must be recorded in
`evidence_used` and mentioned in `summary`, and `confidence` lowered: an author
stating the topic is contested is evidence that the field does *not* consider it
settled by scope alone.

### 4.4 Insufficient evidence

If the required context could not be assembled — no `Method`/`Dataset`
Descriptions available, texts too thin to judge — **write nothing to the graph.**

Record the outcome in the SQLite candidate store instead. This is a statement
about the pipeline's limits, not about the world, and the graph must contain
only what we assert about the world. It also keeps the pair re-runnable once
extraction improves, at far lower cost than sweeping the whole graph again.

---

## 5. Staleness and re-evaluation

The graph grows continuously, so a conflict classified today may be wrong
tomorrow. A `status` field is the wrong mechanism for this: status is static
while the graph is not, and skipping on status alone locks in stale verdicts.

What actually makes a `Contradiction` stale is **the arrival of a new
participant**.

On each run, for every existing `Contradiction`:

1. Recompute which `Description`s would now block into that cluster.
2. Hash the sorted id list.
3. If it differs from the stored `participants_hash` → re-classify.
4. If `pipeline_version` differs from current → re-classify.
5. Otherwise → skip.

This gives exactly the desired behaviour: unchanged clusters are not
re-examined, clusters that gained new knowledge always are.

**Implementation note — an external index is a cache, not the source of
truth.** An implementation may keep a fast-path index of participant-set
hashes outside Neo4j (e.g. in the SQLite candidate store) to avoid
recomputing clusters from scratch every run. That index alone must never be
sufficient to decide "skip": Neo4j is wiped and re-ingested regularly in this
project, and SQLite persists across that wipe. If skip logic trusted the
external index alone, every cluster would read as "already classified" after
every `Contradiction` it refers to is gone — producing a graph with zero
Contradictions and no error raised. A cluster may be skipped only when BOTH
hold:

  (a) the external index's hash + `pipeline_version` match, AND
  (b) a `Contradiction` node with that exact `participants_hash` actually
      exists in Neo4j — one indexed property lookup, since `participants_hash`
      is already a required node property (§3.2).

**Shrinking clusters.** Re-classification can also cause a cluster to lose a
participant (e.g. a gate re-run later rejects a pair that used to link it into
the cluster). If reconciling `has_contradiction` edges to the current
participant set leaves fewer than two edges, the node must be deleted
(`DETACH DELETE`) in the same transaction as the reconciliation — never left
for a later pass to clean up. `_cleanup_singleton_contradictions`
(construction-time) does not run during this pass, so a singleton created here
would sit invalid in the graph until the next full ingestion, violating §3.3's
cardinality floor and failing the §8 singleton invariant in the meantime.

`status` (`open` / `explained` / `closed`) is deferred, not rejected. It only
earns its place once a producer other than the pipeline exists — a human
reviewer, or a third claim that settles the matter. Until then it would hold a
constant value. Adding the property later is cheap; unwinding logic that
depended on it is not.

---

## 6. Known limitations

**`Description.text` is LLM-written prose, not the author's original sentence.**
Both NLI and any hedging analysis therefore operate on a paraphrase. Signal
survives but is weakened. The path to the original text exists —
`Description ← MENTIONS ← Chunk` — and is deferred as a v2 refinement.

**Topic dedup is string-based.** Under-merging is the dominant failure mode:
a real conflict between differently-named topics is never blocked together and
disappears silently. S2 mitigates this but does not eliminate it.

**Temporal coverage depends on title-page extraction.** `year` is populated only
when the chunk containing the publication date was processed and the LLM caught
it. A manual `doc_id → year` override file is the recommended fallback.

**Self-reported LLM confidence is not calibrated.** Values from small models
cluster tightly regardless of actual certainty. Treat `confidence` as a
tie-breaker, not a threshold, until it has been measured.

**Same-year pairs are never evaluated for supersession**, even when one
explicitly, textually corrects the other. §4.1 requires the two `Source.year`
values to differ, because year-granularity alone cannot establish a temporal
order between publications from the same year — writing `supersedes` would mean
guessing a direction. Such a pair falls through to step 2 and, if clustering
and classification support one, is recorded as a `Contradiction` rather than a
`supersedes` edge. This asserts nothing false, but it does mean a real
same-year correction is under-classified as a standing conflict rather than a
resolved update. Not treated as a spec gap — this is deliberate conservatism,
not an oversight.

---

## 7. Deferred to v2

| Item | Why deferred |
|---|---|
| Hedging / factuality gate (G4) | `Description.text` is paraphrase; cues are unreliable at this layer. Claim certainty is judged by the classification LLM, which reads fuller context |
| Auto-resolution of `unresolved` via methodology hierarchy | Requires an evidence hierarchy for the IE/IM domain, which does not yet exist, and execution-level method attributes (sample size, response rate) that are not currently extracted |
| `status` lifecycle | No producer other than the pipeline exists yet (§5) |
| Document-frequency heuristic for generic topics | Corpus too small for percentages to be stable |
| Tracing back to original chunk text | Meaningful complexity for an uncertain gain (§6) |

---

## 8. Invariants — check after every run

All relationship types below are UPPERCASE — see §3.5 "Casing: two layers, one
rule". A query written with `supersedes`/`has_contradiction` (lowercase) does
not error; it silently matches zero rows regardless of the graph's actual
state, which makes every invariant below look like it passed when it never ran.

```cypher
// Wrong endpoint type on SUPERSEDES — must be empty
MATCH (a)-[r:SUPERSEDES]->(b)
WHERE NOT a.typeName IN $allowed OR NOT b.typeName IN $allowed
RETURN a.id, b.id;

// Anti-cycle violation — must be empty
MATCH (a)-[:SUPERSEDES]->(b)-[:SUPERSEDES]->(a)
RETURN a.id, b.id;

// Self-loop — must be empty
MATCH (a)-[:SUPERSEDES]->(a) RETURN a.id;

// Singleton Contradiction — must be empty
MATCH (c:Contradiction)
WHERE COUNT { (c)<-[:HAS_CONTRADICTION]-() } < 2
RETURN c.id;

// scope_conditions present iff scope_difference — must be empty
MATCH (c:Contradiction)
WHERE (c.resolution_type = 'scope_difference' AND c.scope_conditions IS NULL)
   OR (c.resolution_type <> 'scope_difference' AND c.scope_conditions IS NOT NULL)
RETURN c.id, c.resolution_type;

// Duplicate Contradictions with identical participant sets — must be empty.
// Node ids are random (contradiction-{uuid4}), so a failed identity lookup
// during re-classification would produce a second node that no id-based check
// can catch — this is the only invariant that catches it.
MATCH (c:Contradiction)<-[:HAS_CONTRADICTION]-(d:Description)
WITH c, collect(DISTINCT d.id) AS parts
WITH apoc.coll.sort(parts) AS sorted_parts, collect(c.id) AS cs
WHERE size(cs) > 1
RETURN sorted_parts, cs;
```