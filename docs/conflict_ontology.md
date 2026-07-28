# Conflict Ontology — Contradiction & Supersedes

Spec for the **on-demand conflict-detection pass** (the loop in the flowchart:
`Knowledge Base → Contradict Detection & Classification (LLM) → Contradiction? →
Add Edge and/or Nodes`). This pass runs **separately** from graph construction,
**only when called**, and reads the **whole** Knowledge Base (not one chunk),
which is why it can catch conflicts that per-chunk construction cannot.

Two axes, don't confuse them:
- **When it runs** → a separate on-demand pass (per the diagram). ✅
- **What shapes it writes** → the ontology defined *here*. This is a **single
  source of truth**: both construction and this detector must produce the same
  node/edge shapes, or Cypher queries have to handle two variants. The canonical
  constants live in [`src/graph/graph_model.py`](../src/graph/graph_model.py).

The classifier produces **one of two** outcomes per detected conflict:

| Situation | Output | Creates a node? |
|-----------|--------|-----------------|
| Two facts genuinely conflict and both stand (unresolved) | **Contradiction** node + `has_contradiction` edges | Yes |
| One Result is a corrected / newer version of another (the new one wins) | **`supersedes`** edge | No — edge only |

---

## 1. Contradiction (already defined — reuse as-is)

Fully specified in the construction ontology; the detector emits the **same
shapes**. Do not redefine.

**Node `Contradiction`** — reified conflict.
- Required property `summary`: ≥2 sentences, must name the specific conflicting
  detail from *both* sides (a number, date, or claim). "These disagree" is
  forbidden.
- Anchors ≥2 `has_contradiction` edges under one id (a 3-way conflict is one
  node with 3 edges, not 3 pairwise structures).

**Edge `has_contradiction`**: `Description → Contradiction`.
- Required property `level` ∈ `{direct, partial, apparent}`
  (`ALLOWED_CONTRADICTION_LEVEL`).

**Cardinality**: a Contradiction needs **≥2** `has_contradiction` edges
(`MIN_CONTRADICTION_PARTICIPANTS = 2`). A singleton is meaningless. Because a 2nd
participant may only appear later, this is enforced **downstream over the whole
graph**, not eagerly — see
[`KnowledgeGraph._cleanup_singleton_contradictions`](../src/graph/knowledge_graph.py).

Constants: `ALLOWED_LABELS` (includes `Contradiction`), `_FIXED_RELATION_DIRS["has_contradiction"]`,
`ALLOWED_CONTRADICTION_LEVEL`, `MIN_CONTRADICTION_PARTICIPANTS`.

---

## 2. Supersedes (new — this is the piece to define)

A corrected / newer **Result** replaces an earlier one. Edge-only, **no node** —
"supersedes basically just updates information".

**Edge `supersedes`**: `Description → Description`.
- **Both endpoints must be Result Descriptions** (`typeName == "Result"`,
  `SUPERSEDES_ENDPOINT_TYPE = "Result"`). Not backgrounds, not methods — only
  findings/results get corrected.
- **Direction**: `source = newer/correct`, `target = older/replaced`. Read
  "(A) supersedes (B)".
- Optional property `reason`: short text on why it's an update
  (e.g. "revised figure in the final chapter"). No `level`/classification vocab —
  updates are not graded like contradictions.

**Constraints** (enforced by this pass / a downstream whole-graph query, since
the two Results usually live in different chunks or documents):
1. Both endpoints are `Description` with `typeName == "Result"`.
2. No self-loop (`source != target`).
3. **Anti-cycle**: `A supersedes B` and `B supersedes A` must never co-exist —
   that is a Contradiction, not a supersession. The classifier picks exactly one
   relationship for a given pair, never both.

Constants: `SUPERSEDES_RELATION`, `SUPERSEDES_ENDPOINT_TYPE` in `graph_model.py`.
(Deliberately **not** in `_FIXED_RELATION_DIRS` — that map is the per-chunk
construction ontology, and `supersedes` is never emitted at construction time.)

### Classifier decision rule (Contradiction vs supersedes)

```
conflict between two Result facts A (newer) and B (older):
  is A a correction / update of B, where B is now wrong?
    → yes:  (A)-[:supersedes {reason?}]->(B)          # no node
    → no (both still stand, genuinely opposed):
            create Contradiction{summary} + 
            (A)-[:has_contradiction {level}]->(C) and
            (B)-[:has_contradiction {level}]->(C)      # reified node
```

### Example Cypher

Create a supersedes edge (both must be Result Descriptions):
```cypher
MATCH (newer:Description {typeName: 'Result'}), (older:Description {typeName: 'Result'})
WHERE newer.id = $newer_id AND older.id = $older_id AND newer.id <> older.id
MERGE (newer)-[r:supersedes]->(older)
SET r.reason = $reason      // optional
```

Downstream validation — flag edges that break the ontology (run after the pass):
```cypher
// wrong endpoint type
MATCH (a)-[r:supersedes]->(b)
WHERE a.typeName <> 'Result' OR b.typeName <> 'Result'
RETURN a.id, b.id;          // -> should be empty; delete if not

// anti-cycle: a pair superseding each other
MATCH (a)-[:supersedes]->(b)-[:supersedes]->(a)
RETURN a.id, b.id;          // -> should be empty; keep only the correct direction
```

---

## Scope note

Construction (per-chunk) currently also emits Contradictions, but the prompt
itself notes it only sees one chunk, so it misses cross-chunk / cross-document
conflicts ([`src/prompts/graph_extractor.py`](../src/prompts/graph_extractor.py),
"HONEST NOTE"). This on-demand pass is what closes that gap. `supersedes` is
**only** produced here, never at construction.
