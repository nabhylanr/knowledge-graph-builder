from langchain.prompts import PromptTemplate

# v2 (post-review): v1's "neutral" definition ("about different things, or too
# unrelated to compare") caused the model to wave off exactly the conflict
# class this gate exists to catch — two studies reporting opposite findings for
# the same kind of measurement, but under different methods/datasets/settings.
# A careful model reads "different method, different population" as grounds for
# "neutral" with high confidence, and that verdict gets discarded before ever
# reaching classification. Whether the differing conditions actually RECONCILE
# the opposition is the classification stage's job, not this gate's — the gate
# only asks "do these point in opposite directions on the surface?". v2 makes
# that split explicit instead of letting the model conflate the two questions.
_NLI_PROMPT_TEMPLATE = """You are checking whether two short research-claim statements are in factual conflict.

Statement A: "{text_a}"
Statement B: "{text_b}"

Decide the relationship between them:
- "contradiction": A and B make opposing claims about the same kind of measurement, outcome, or property — e.g. one reports an increase and the other a decrease, one reports success and the other failure, or they give incompatible numbers for the same quantity.
- "entailment": A and B state the same fact, or one is a more specific or more general restatement of the other, with no opposition.
- "neutral": A and B concern genuinely different subjects — different quantities, different phenomena — so there is nothing to agree or disagree about.
- "unclear": you cannot confidently judge (ambiguous wording, missing context, or they seem related but you cannot tell whether the specific claims oppose).

IMPORTANT: These two statements may come from different studies, datasets, methods, populations, or time periods. Do NOT treat that as a reason to answer "neutral". If the two statements point in opposite directions about the same kind of measurement or outcome, answer "contradiction" — even if the underlying conditions clearly differ. Judging whether those differing conditions explain the opposition is a separate later step, not your task here.

Answer "neutral" only when the two statements are about genuinely different subjects, not merely different studies of the same subject.

If you are still not sure after applying the above, choose "unclear" rather than guessing — a wrong "contradiction" or "entailment" is worse than an honest "unclear".

Respond with ONLY a JSON object, no other text:
{{"label": "contradiction|entailment|neutral|unclear", "confidence": <float 0.0-1.0>, "rationale": "<one short sentence>"}}
"""

NLI_PROMPT_VERSION = "nli-v2"


def get_nli_gate_prompt() -> PromptTemplate:
    """Returns the G5 NLI-verdict prompt (stage-2 gate filtering, docs/conflict_ontology.md)."""
    return PromptTemplate(
        template=_NLI_PROMPT_TEMPLATE,
        input_variables=["text_a", "text_b"],
    )
