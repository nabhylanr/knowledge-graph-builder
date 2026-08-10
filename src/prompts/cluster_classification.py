from langchain.prompts import PromptTemplate

# §4.3 n-ary cluster classification. v2 fixes over v1:
# - Added step 0 ("not_a_conflict"), checked BEFORE scope_difference/
#   known_controversy/unresolved. v1 asked the model to explain WHY claims
#   oppose without ever asking WHETHER they actually do — and scope_difference
#   is trivially satisfiable (almost any two claims from different papers
#   differ in method or dataset), so the model always found a reason, even for
#   pairs that were never in opposition to begin with (e.g. two claims about
#   different measurements that merely share a topic).
# - "insufficient_evidence" reframed explicitly as a control signal (routes to
#   SQLite-only, nothing written to Neo4j) rather than reading like a 4th
#   stored resolution_type. "not_a_conflict" is the same kind of control signal.
# - "unresolved" is gated on scope having been checkable at all — see
#   docs/conflict_pipeline.md §4.3(c) "Enforced precondition". The prompt
#   states the rule; classifier.py enforces it in code regardless of what the
#   model returns, since this is a correctness invariant, not just guidance.
# - Explicit note that participants may share a source_id (intra-document
#   conflicts are deliberately kept, §2.1) so the model doesn't read a repeated
#   source as an input error.
# v3 fix over v2: step 0's "leave every other field null/empty" instruction
# was written before "confidence" existed as a concept the model needed to
# keep filling — the model correctly generalised it to mean "including
# confidence" and started omitting a required field, which Pydantic then
# rejected outright (ClusterVerdict.confidence had no default). Fixed here by
# carving out an explicit exception for confidence, plus a standalone
# reinforcing paragraph next to the JSON schema. ClusterVerdict.confidence now
# also has a 0.5 default as a second line of defence (classification_client.py)
# — a missing number should not throw away an otherwise-usable verdict.
_CLUSTER_CLASSIFICATION_PROMPT_TEMPLATE = """You are given a set of research claims that an upstream filter flagged as
POSSIBLY opposing each other. Your first job is to check whether they actually
do; if they do, classify WHY, so a knowledge graph can record it as a
synthesized conflict.

The following {n} claims were each independently flagged, pairwise, as
worth checking for opposition on the same kind of measurement or outcome —
that flag is a candidate signal, not a confirmed conflict. Two or more claims
may come from the SAME source — that is expected, not an input error:
intra-document conflicts (e.g. between distant chapters of one paper) are
deliberately kept by this pipeline.

{participants_block}

FIRST, before anything else:

0. "not_a_conflict" — do these claims make OPPOSING assertions about the SAME
   measurement, outcome, or finding? A shared topic is NOT sufficient. Two
   claims can discuss the same subject while asserting entirely different
   things — that is not a conflict. Ask specifically: do they point in
   OPPOSITE directions about the same thing? If instead they differ in WHAT
   they measure, or address different aspects of a shared subject, answer
   "not_a_conflict" — even if the texts sound superficially similar. Check
   this FIRST: almost any two claims from different papers differ in method
   or dataset, so "scope_difference" below is trivially satisfiable and must
   never be used as a substitute for this check. Example of NOT a conflict:
   "Claim 1 measures asset turnover using Sales/Assets" vs. "Claim 2 identifies
   total assets as a factor in supply chain integration performance" — two
   different topics, not opposing assertions about one.

   → If they do not oppose each other: "resolution_type": "not_a_conflict".
     This is a CONTROL SIGNAL, not a stored outcome — the cluster is recorded
     in the pipeline's own tracking only, and NOTHING is written to the
     knowledge graph. Fill "not_a_conflict_reason" with a short statement of
     what the claims actually differ in instead of opposing. Leave "summary",
     "scope_conditions", "evidence_used", "positions", and
     "insufficient_evidence_reason" null/empty as shown in the schema below —
     but "confidence" is NOT one of the fields you leave out: you MUST still
     fill it (your confidence that these claims do NOT oppose each other).
     See the note on "confidence" below the lettered list — it applies here
     too, without exception.

ONLY IF the claims DO make opposing assertions about the same measurement or
outcome, decide, IN ORDER, which ONE of the following best explains the
opposition:

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
   the claims DO oppose each other but you could not reach any of (a)/(b)/(c)
   with reasonable confidence — e.g. the claim texts are too thin, or (per the
   "unresolved" rule above) no scope context exists at all and no controversy
   marker is present either. Use this rarely — only when truly stuck, not
   merely when you are somewhat unsure. Do not use it when the real answer is
   that the claims never opposed each other at all — that is "not_a_conflict"
   (step 0), a different thing from being unable to judge a genuine opposition.

PRECEDENCE RULE: if BOTH "scope_difference" AND a controversy marker apply,
answer "scope_difference" (it is the more actionable finding) — but you MUST
still mention the controversy marker in "summary", include it in
"evidence_used", and LOWER your "confidence": an author calling the topic
contested is evidence the field does not consider it settled by scope alone.

"summary" must be at least two sentences and MUST name the specific
conflicting detail from EVERY participating claim (a number, an outcome, a
direction) — "these claims disagree" alone is forbidden, it must say what each
one actually claims. Leave "summary" null if resolution_type is
"insufficient_evidence" or "not_a_conflict".

"evidence_used" is a list of short tags describing which signals drove your
conclusion, e.g. ["method", "dataset", "population", "year", "controversy_marker"].
If only some (not all) participants had scope context available and you chose
"unresolved", include a "partial_context" tag.

"confidence" (a float 0.0-1.0) is REQUIRED in every single response, for every
resolution_type without exception — including "not_a_conflict" and
"insufficient_evidence". Other fields may be null or empty depending on which
resolution_type you chose; "confidence" is never one of them. Do not omit it.

Respond with ONLY a JSON object, no other text:
{{
  "resolution_type": "not_a_conflict|scope_difference|known_controversy|unresolved|insufficient_evidence",
  "summary": "<at least two sentences, or null if insufficient_evidence/not_a_conflict>",
  "scope_conditions": "<only if scope_difference, else null>",
  "confidence": <float 0.0-1.0>,
  "evidence_used": ["tag1", "tag2"],
  "positions": [{{"description_id": "...", "position": "<short stance, only if known_controversy with 3+ participants, else omit>"}}],
  "insufficient_evidence_reason": "<only if insufficient_evidence, else null>",
  "not_a_conflict_reason": "<only if not_a_conflict, else null>"
}}
"""

CLASSIFICATION_PROMPT_VERSION = "classification-v3"


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
