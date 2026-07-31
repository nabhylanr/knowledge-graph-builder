from langchain.prompts import PromptTemplate

# §4.3 n-ary cluster classification. v1 fixes over the first draft:
# - "insufficient_evidence" reframed explicitly as a control signal (routes to
#   SQLite-only, nothing written to Neo4j) rather than reading like a 4th
#   stored resolution_type.
# - "unresolved" is gated on scope having been checkable at all — see
#   docs/conflict_pipeline.md §4.3(c) "Enforced precondition". The prompt
#   states the rule; classifier.py enforces it in code regardless of what the
#   model returns, since this is a correctness invariant, not just guidance.
# - Explicit note that participants may share a source_id (intra-document
#   conflicts are deliberately kept, §2.1) so the model doesn't read a repeated
#   source as an input error.
_CLUSTER_CLASSIFICATION_PROMPT_TEMPLATE = """You are classifying WHY a set of research claims oppose each other, so a
knowledge graph can record it as a synthesized conflict.

The following {n} claims have been identified as mutually opposing (each pair
among them was independently flagged as contradicting on the same kind of
measurement or outcome). Two or more claims may come from the SAME source —
that is expected, not an input error: intra-document conflicts (e.g. between
distant chapters of one paper) are deliberately kept by this pipeline.

{participants_block}

Decide, IN ORDER, which ONE of the following best explains the opposition:

a. "scope_difference" — the opposition is explained by a difference in method,
   dataset, population, setting, unit of analysis, measurement, or time period
   between the claims. Use the context provided above for each claim's source
   to judge this. If you choose this, you MUST fill "scope_conditions": a
   short statement of the condition under which EACH side holds, e.g. "Under
   discrete-event simulation → Claim 1; under survey of 300 firms → Claim 2."

b. "known_controversy" — at least one claim's own text explicitly acknowledges
   the topic as contested (words like "controversial", "paradoxical",
   "conflicting evidence", "remains debated"), REGARDLESS of whether (a) also
   applies. If a controversy marker is present, you must still check (a) first
   — see the precedence rule below.

c. "unresolved" — you have sufficient information to judge, scope was checked
   (context was available for at least one claim above) and does not explain
   it, and no controversy marker is present. Do NOT use this if NO claim has
   ANY method/dataset/experiment context available above — in that case scope
   could not be checked at all; use "insufficient_evidence" instead.

d. "insufficient_evidence" — this is a CONTROL SIGNAL, not a fourth stored
   outcome. Choosing it means the cluster is recorded in the pipeline's own
   tracking only, and NOTHING is written to the knowledge graph. Use it when
   you could not reach any of the above with reasonable confidence — e.g. the
   claim texts are too thin, or (per the "unresolved" rule above) no scope
   context exists at all and no controversy marker is present either. Use this
   rarely — only when truly stuck, not merely when you are somewhat unsure.

PRECEDENCE RULE: if BOTH "scope_difference" AND a controversy marker apply,
answer "scope_difference" (it is the more actionable finding) — but you MUST
still mention the controversy marker in "summary", include it in
"evidence_used", and LOWER your "confidence": an author calling the topic
contested is evidence the field does not consider it settled by scope alone.

"summary" must be at least two sentences and MUST name the specific
conflicting detail from EVERY participating claim (a number, an outcome, a
direction) — "these claims disagree" alone is forbidden, it must say what each
one actually claims. Leave "summary" null if resolution_type is
"insufficient_evidence".

"evidence_used" is a list of short tags describing which signals drove your
conclusion, e.g. ["method", "dataset", "population", "year", "controversy_marker"].
If only some (not all) participants had scope context available and you chose
"unresolved", include a "partial_context" tag.

Respond with ONLY a JSON object, no other text:
{{
  "resolution_type": "scope_difference|known_controversy|unresolved|insufficient_evidence",
  "summary": "<at least two sentences, or null if insufficient_evidence>",
  "scope_conditions": "<only if scope_difference, else null>",
  "confidence": <float 0.0-1.0>,
  "evidence_used": ["tag1", "tag2"],
  "positions": [{{"description_id": "...", "position": "<short stance, only if known_controversy with 3+ participants, else omit>"}}],
  "insufficient_evidence_reason": "<only if insufficient_evidence, else null>"
}}
"""

CLASSIFICATION_PROMPT_VERSION = "classification-v1"


def get_cluster_classification_prompt() -> PromptTemplate:
    """Returns the §4.3 n-ary cluster-classification prompt (stage-3 classification, docs/conflict_pipeline.md)."""
    return PromptTemplate(
        template=_CLUSTER_CLASSIFICATION_PROMPT_TEMPLATE,
        input_variables=["n", "participants_block"],
    )


def format_participants_block(participants: list) -> str:
    """
    participants: list of dicts, each with keys
    {index, id, source_id, topic, text, context_texts (list[str])}.
    """
    blocks = []
    for p in participants:
        if p["context_texts"]:
            context = "\n".join(f"    - {t}" for t in p["context_texts"])
        else:
            context = "    (none available)"
        blocks.append(
            f'Claim {p["index"]} (id: {p["id"]}, from {p["source_id"]}, topic: "{p["topic"]}"): "{p["text"]}"\n'
            f'  Context from the same source ({p["source_id"]}) — method/dataset/experiment description(s), if any:\n'
            f'{context}'
        )
    return "\n\n".join(blocks)
