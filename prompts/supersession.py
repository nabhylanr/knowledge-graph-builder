from langchain.prompts import PromptTemplate

# §4.1 pairwise supersession test. v1 originally only received the two claim
# texts + years — the evidence-level guard ("if the newer claim rests on
# markedly weaker evidence...") was unenforceable without knowing what evidence
# either claim rests on. v1 fix: pass the same Method/Dataset/Experiment scope
# context Prompt B receives, for both sides — it's assembled once per run
# regardless, so this is free.
_SUPERSESSION_PROMPT_TEMPLATE = """You are determining whether a NEWER research claim is a correction or update
of an OLDER research claim, such that the older claim is now considered wrong.

OLDER claim (published {older_year}): "{older_text}"
Context available for the OLDER claim's source — method/dataset/experiment description(s), if any:
{older_context_block}

NEWER claim (published {newer_year}): "{newer_text}"
Context available for the NEWER claim's source — method/dataset/experiment description(s), if any:
{newer_context_block}

Decide: does the newer claim explicitly correct, revise, or update the older
one — not merely restate it with a different result, but actively supersede it?

Evidence that supports "yes":
- Explicit correction language in the newer claim's text: "contrary to prior
  findings", "we revise", "earlier work assumed", "has since been shown", or
  similar
- The newer work appears to cite/reference the older finding in a contrastive
  or corrective context
- The newer work explicitly claims a methodological improvement over the older
  approach
- A retraction or erratum is mentioned for the older work

A year difference ALONE is NEVER sufficient — two independent studies published
in different years are simply two studies, not a supersession, unless the text
itself signals correction.

Evidence-level guard: use the context above to judge the relative strength of
evidence behind each claim. If the newer claim appears to rest on markedly
weaker evidence than the older one (e.g. a single case study superseding what
reads like a meta-analysis or large-scale study), answer "no" even if
correction language is present — a newer, weaker study does not supersede an
older, stronger one merely by being newer.

If you are not confident the newer claim is an explicit correction (as opposed
to simply a different, independent finding), answer "no" — supersession is a
narrow, deliberate claim, and defaulting to "no" is always safe: an unresolved
pair is still considered for the standing-contradiction path next.

Respond with ONLY a JSON object, no other text:
{{"decision": "yes|no", "basis": "explicit_correction|citing_contrast|claimed_improvement|retraction|null", "reason": "<one short sentence, or null>", "confidence": <float 0.0-1.0>}}

If decision is "no", set "basis" and "reason" to null.
"""

SUPERSESSION_PROMPT_VERSION = "supersession-v1"


def get_supersession_prompt() -> PromptTemplate:
    """Returns the §4.1 pairwise supersession-test prompt (stage-3 classification, docs/conflict_pipeline.md)."""
    return PromptTemplate(
        template=_SUPERSESSION_PROMPT_TEMPLATE,
        input_variables=[
            "older_year", "older_text", "older_context_block",
            "newer_year", "newer_text", "newer_context_block",
        ],
    )


def format_context_block(context_texts: list) -> str:
    """Renders a claim's Method/Dataset/Experiment context texts, or a placeholder if none exist."""
    if not context_texts:
        return "(none available)"
    return "\n".join(f"  - {t}" for t in context_texts)
