from langchain.prompts import PromptTemplate


def get_graph_extractor_prompt() -> PromptTemplate:
    """
    Returns the KG extraction prompt — v8 (English).

    Full ontology replacement. See ontology_spec_v8.md for the rationale behind
    each change. Summary of what moved since v7:

    1. Type vocabulary replaced with the paper/meeting taxonomy supplied by the
       user (14 Paper Types, 7 Meeting Types), fully disjoint — no more "SHARED"
       bucket, so a Topic's Type alone tells you which domain it came from.
    2. has_[type] and has_[type]_description are no longer dynamically named.
       Fixed to has_type and has_description. The Type node's id already encodes
       which type it is, so the dynamic name only made the relationship schema
       unbounded without adding information.
    3. relates_to now has a hard source-Type -> target-Type constraint table
       (RELATION VOCABULARY below), not just a prose description. sanitize_graph
       can reject a relates_to edge whose endpoint Types don't match the allowed
       pair for that relation.
    4. Added `uses` (Method/Experiment -> Dataset) and `extends` / `compares_to`
       (positioning against prior work) — common paper moves the v7 vocabulary
       had no relation for.
    5. spoke_about / writes_about gained an optional `stance` property (raised,
       proposed, decided, reported, gave_feedback) so meeting attribution
       ("who raised this Issue") doesn't need a new relationship per Type.
    6. Action Item Topics may carry an optional `status` property (open,
       in_progress, done, blocked) — only when the text states it.
    7. Source gained an optional `date` property, so meetings/papers can be
       placed in time.
    8. Explicit rule against duplicate Descriptions when a Topic has more than
       one Type: each Description needs its own specific detail, or the second
       Type should be dropped.
    9. (v8.1) Added a 7th node type, Contradiction, plus a fixed has_contradiction
       edge (Description -> Contradiction, with a closed `level` property) to
       capture fact-level conflicts between two Descriptions. This is deliberately
       a NEW node + ONE fixed edge name — NOT a family of dynamically-named edges
       like "contradict_level1" / "contradict_level2". Dynamic relationship names
       were already rejected once for has_[type] (see point 2 above); the same
       reasoning applies here, so `level` is a property, never part of the name.
       relates_to's existing `contradicts` value (Topic -> Topic) is untouched and
       still the right choice for a Topic/finding-level conflict — has_contradiction
       is strictly for when the conflict is only visible at the Description detail
       level (specific numbers/claims), see ONTOLOGY section below for the boundary.

    HONEST NOTE (still true from v7): a better prompt reduces many issues, but
    hard numeric constraints and self-loops may still leak partially on a model
    as small as llama-3.1-8b-instant. Anything that leaks is caught
    deterministically by `sanitize_graph` — do not keep rewriting the prompt to
    chase it. sanitize_graph should be updated alongside this prompt to enforce
    the relates_to type-pair table in section "RELATION VOCABULARY" below.

    HONEST NOTE (Contradiction, v8.1): this prompt only ever sees {input_text}
    for ONE chunk at a time — it has no memory of Descriptions written for
    earlier chunks or other documents. So has_contradiction can only catch a
    conflict where BOTH Descriptions are being written in the SAME chunk (the
    same call). It will NOT catch the more interesting cross-chunk/cross-document
    case (e.g. Chapter 2's related-work summary vs. this paper's own Chapter 4
    result). Catching that needs a separate pass over already-stored Descriptions
    (embedding similarity + an LLM judge), which is out of scope for this prompt.
    Also: a Contradiction with only one participant is meaningless and is NOT
    something sanitize_graph can always reject on its own (see graph_model.py) —
    a maintenance query removes singleton Contradiction nodes after the whole
    document is stored. Do not rely on sanitize_graph alone to catch that case;
    just try not to emit a Contradiction node unless you can name a second
    conflicting Description for it in the same output.
    """

    prompt = """
You are a HIGHLY DISCIPLINED Knowledge Graph extraction algorithm.
Extract Nodes and Relationships from INPUT TEXT following this ontology EXACTLY.
Do NOT hallucinate. Do NOT use node/relationship types outside this ontology.

==============================
MANDATORY STEPS BEFORE WRITING JSON
==============================

Do the following internally (do NOT write them to the output):

STEP A — Read INPUT TEXT and identify the 1-3 MOST IMPORTANT (top-level) Topics
   that will be the parents of the whole graph. Choose Topics that are the
   CONTRIBUTION or MAIN FOCUS of THIS document itself — do NOT pick a Topic that
   is only mentioned as a reference / related work / other people's discussion.
   These are the ONLY Topics allowed to have a has_source edge. Every other Topic
   MUST connect to the graph via has_subtopic (a chain from one of these 1-3
   Topics) and MUST NOT have its own has_source.

STEP B — For each Topic, decide its Type from the vocabulary below. Paper Types
   and Meeting Types are DISJOINT — a Topic gets exactly one Type from whichever
   vocabulary actually describes it. A single document may contain Topics from
   both vocabularies (e.g. a meeting where a supervisor gives Feedback on a
   Method the student presented) — that is expected, not an error. If two Type
   candidates look similar, PICK THE CLOSEST ONE — do not invent a new Type for a
   small nuance.

STEP C — Only after STEP A and B: look for two Topics the text EXPLICITLY
   connects to each other (not just co-occurring in the same paragraph) and add
   a relates_to edge with a relation property from the RELATION VOCABULARY
   below — the source Topic's Type and target Topic's Type MUST match the
   allowed pair for that relation, or do not add the edge. Example trigger
   phrases: "to address this gap, we propose...", "the team decided X, which
   resulted in...", "this contradicts the earlier finding that...". Do NOT
   force a relates_to between Topics that merely appear near each other — skip
   this step entirely if nothing in the text states a connection. Separately, if
   a Topic's Type is Action Item and the text names who is responsible, add an
   assigned_to edge from that Topic to the Agent; if a deadline is mentioned, add
   it as a due_date property on that edge; if the text states whether the task is
   done, in progress, blocked, or still open, set a status property on the Topic
   itself (not on the edge).

STEP D — While writing this chunk's Descriptions, check whether two of THEM
   (not just their Topics) state directly incompatible facts — specific numbers,
   dates, or claims that cannot both be true, about the same or closely related
   subject. Do this only among Descriptions you are writing IN THIS SAME OUTPUT —
   you cannot see Descriptions from other chunks, so do not invent a contradiction
   against something you cannot cite from THIS input text. If you find such a
   pair, create ONE Contradiction node (id: "Contradiction <short label>",
   Title Case) with a `summary` property stating the specific conflicting detail
   from both sides, then add a has_contradiction edge from EACH conflicting
   Description to that Contradiction node with a `level` property from the
   CONTRADICTION LEVEL VOCABULARY below. A Contradiction needs at least TWO
   conflicting Descriptions — if you can only name one side, do not create the
   node at all. This is separate from relates_to's `contradicts` value: use
   `contradicts` (Topic -> Topic) when the text itself frames two findings/ideas
   as opposed; use has_contradiction (Description -> Contradiction) when the
   conflict is only visible once you compare the specific detail inside two
   Descriptions. Use both together only if the text supports both levels
   explicitly — do not encode the same single conflict twice by default.

==============================
INPUT METADATA
==============================

Source file name: {source_name}
Source format: {source_format}

SOURCE ID RULES (MANDATORY, OFTEN VIOLATED):
- Create EXACTLY ONE Source node with id = "{source_name}" and name = "{source_name}".
- COPY the string {source_name} VERBATIM, character for character. Do NOT add a
  prefix ("Source ..."), do NOT change spacing/capitalization, do NOT create any
  variation. If your id differs from {source_name} even slightly, the system reads
  it as a different document — a fatal error.
- Do NOT create a Source from a table, figure, or chapter. Those are Topics.
- If the text states a date for the meeting or the paper's publication, add it as
  a date property on the Source. Do not guess a date that is not in the text.

==============================
ONTOLOGY — NODE TYPES (7 only)
==============================

1. Agent — A named person or organization that contributes to the document.
   Examples: named authors, speakers, supervisors, named institutions.
   NOT an Agent: worker, robot, system, station, order, SKU, pod, equation, table, figure.
   Required property: name

2. Role — The function of an Agent in a specific context.
   Examples: Author, Co-author, Supervisor, Speaker, Presenter, Moderator, Researcher.
   Required property: name

3. Topic — Any concept, system, method, metric, problem, section, figure, table, or
   domain term discussed in the text. A subtopic is not a separate type — it is a
   normal Topic linked via has_subtopic.
   Build a hierarchy where the text supports it (no forced depth, do not fabricate).
   Required property: name.
   Optional: abbreviation, chapterNumber, tableNumber, figureNumber.
   Optional, ONLY when Type is Action Item and the text states it: status (one of
   "open", "in_progress", "done", "blocked"), priority.

4. Type — A semantic category classifying a Topic's function.
   Type classifies — it is NOT the content itself.
   Naming: exact casing from the GUIDED TYPE VOCABULARY below. Do not invent a new
   Type; if nothing fits closely, prefer the closest existing Type.
   Required property: name

5. Source — The uploaded file the information is extracted from. See SOURCE ID RULES above.
   Required property: name. Optional: format, date.

6. Description — A textual explanation of why a Topic belongs to a given Type.
   Specific to one Topic-Type pair. Minimum 2 sentences, and MUST contain at least
   ONE specific detail from the text (a number, name, technical term, or measured
   result). Do NOT write generic tautological sentences like "X is a method." or
   "Y is a problem." — such sentences carry no information and are FORBIDDEN.
   If a Topic has more than one Type, each Type's Description must contain a
   DIFFERENT specific detail — if you cannot write a Description that doesn't
   just restate the other Type's Description, the Topic does not need the second
   Type; drop it instead.
   id format: Description::<TopicName>::<TypeName>
   Required properties: text, topicName, typeName
   Linked FROM the Type node (Type -> has_description -> Description).
   NEVER from a Topic directly, NEVER from an Agent directly.

7. Contradiction — A reified conflict between two (or more) Descriptions that
   state incompatible facts. It is NOT a Topic and NOT a Description itself —
   it exists only to hold a `summary` of why the conflict exists and to anchor
   >=2 has_contradiction edges under one addressable id (so a 3-way conflict
   does not need a fabricated pairwise structure). See STEP D above for when
   to create one, and CONTRADICTION LEVEL VOCABULARY below for the `level`
   values used on its incoming edges.
   Required property: summary — minimum 2 sentences, MUST name the specific
   conflicting detail from both sides (a number, date, or claim) — the same
   non-tautological rule as Description applies; "these two Descriptions
   disagree" alone is FORBIDDEN, it must say what they each claim.
   A Contradiction with only one has_contradiction edge is meaningless — see
   STEP D: only create it when you can cite a second conflicting Description
   in this same output.

==============================
ONTOLOGY — RELATIONSHIPS (11 only, direction MUST match the table exactly)
==============================

| Relationship     | Source Node | Target Node | Meaning                                              |
|------------------|-------------|-------------|-------------------------------------------------------|
| role_in_meeting  | Agent       | Role        | Agent had this role in a meeting                       |
| role_in_paper    | Agent       | Role        | Agent contributed to a paper                            |
| spoke_about      | Agent       | Topic       | Agent verbally discussed this Topic (meeting context)   |
| writes_about     | Agent       | Topic       | Agent wrote about this Topic (paper context)            |
| has_source       | Topic       | Source      | Only the 1-3 top-level Topics (STEP A)                  |
| has_type         | Topic       | Type        | fixed name — do NOT vary it per Type                    |
| has_description  | Type        | Description | direction: Type -> Description, NOT Topic -> Description|
| has_subtopic     | Topic       | Topic       | broader -> narrower                                     |
| relates_to       | Topic       | Topic       | explicit semantic link (STEP C); requires a `relation` property, endpoints must match the RELATION VOCABULARY type-pair table |
| assigned_to      | Topic       | Agent       | ONLY when the Topic's Type is Action Item; the Agent is the owner |
| has_contradiction| Description | Contradiction | Description states a fact that conflicts with another Description's fact (STEP D); requires a `level` property from CONTRADICTION LEVEL VOCABULARY below |

All relationship names above are FIXED strings, always lowercase snake_case,
exactly as written — never "has_method", "has_decision", "contradict_level1",
"contradict_level2", or any other variant. NO "::" or other symbols in a
relationship name. A classification (like has_contradiction's `level`) is
ALWAYS a property value, NEVER part of the relationship name — see MISTAKES.

On spoke_about and writes_about, you MAY add an optional `stance` property — one
of "raised", "proposed", "decided", "reported", "gave_feedback" — ONLY when the
text makes it unambiguous who originated the point. If it's ambiguous, omit the
property rather than guess.

On has_contradiction, you MUST add a `level` property from CONTRADICTION LEVEL
VOCABULARY below, classifying how strong the conflict is.

==============================
RELATION VOCABULARY (for relates_to only — pick exactly one, lowercase)
==============================

Each relation is only valid between the listed source Type(s) and target Type(s).
If the two Topics' Types don't match a row below, do not use that relation — and
if no row fits at all, do not add a relates_to edge.

  addresses      — Method -> Problem | Research Goal
                   a proposed method targets this problem/goal
  resolves       — Decision -> Issue | Open Question
                   a decision settles the issue/question
  produces       — Decision -> Action Item
                   a decision generates a task
  evaluates      — Experiment | Metrics Evaluation -> Result
                   a measurement yields this result
  uses           — Method | Experiment -> Dataset
                   the method/experiment is applied to this dataset
  motivates      — Background | Research Gap | Problem -> Method | Research Goal
                   the reason this method/goal exists
  identifies     — Background | Existing Research -> Research Gap
                   prior work reveals this gap
  extends        — Method -> Existing Research | Theoretical Basis
                   this method builds on prior work or theory
  compares_to    — Result -> Existing Research | Result
                   benchmarking or comparison
  contradicts    — Result -> Result   OR   Feedback -> Idea | Decision
                   a direct conflict
  responds_to    — Feedback -> Idea | Decision | Progress Update | Method | Result | Limitation
                   feedback given about this item (non-conflicting)
  follows_up_on  — Progress Update -> Progress Update | Action Item
                   a later update references an earlier one

If none of these describe the connection, or the Types don't match, do NOT add a
relates_to edge — do not invent a thirteenth value and do not force a mismatched
pair.

==============================
CONTRADICTION LEVEL VOCABULARY (for has_contradiction only — pick exactly one)
==============================

  direct    — the two Descriptions state incompatible facts/numbers for what
              reads as the same claim under the same conditions
  partial   — the conflict only holds under some scope/condition; the two
              Descriptions are not fully incompatible everywhere
  apparent  — looks contradictory, but may just be wording ambiguity or a
              possible wrong Topic merge rather than a confirmed real conflict
              — pick this over "direct" whenever you are not certain

Do not invent a fourth value. If none of these three feels right, you are
probably looking at two Descriptions that are merely different, not
contradictory — do not create the Contradiction node at all.

MISTAKES THAT HAVE HAPPENED BEFORE — DO NOT REPEAT:
- WRONG: Source -> writes_about -> Agent   (Source is never the source of writes_about)
- WRONG: Source -> has_type -> Topic   (Source never has a has_type edge)
- WRONG: Agent -> has_description -> Description   (must come from Type, not Agent)
- WRONG: Topic X -> has_subtopic -> Topic X (same node)   (self-loop, FORBIDDEN:
  a relationship's source id and target id must NEVER be identical)
- WRONG: Agent -> assigned_to -> Topic   (must be Topic -> Agent, not reversed)
- WRONG: a relates_to edge with no relation property, a relation value outside
  the RELATION VOCABULARY list above, or a relation used between Types that
  don't match its allowed pair (e.g. "uses" from a Decision instead of a Method)
- WRONG: "has_method", "has_decision", or any relationship name that varies by
  Type — the has_type edge name never changes
- WRONG: status property on an assigned_to edge (status belongs on the Topic
  node itself; due_date belongs on the assigned_to edge)
- WRONG: assigned_to on a Topic whose Type is Decision, Issue, or anything other
  than Action Item
- WRONG: "contradict_level1", "contradict_level2", or any relationship name
  that encodes the level in the name — level is a property ON has_contradiction,
  the relationship name never changes
- WRONG: Contradiction -> has_contradiction -> Description   (must be
  Description -> Contradiction, not reversed)
- WRONG: a has_contradiction edge with no `level` property, or a level value
  outside direct/partial/apparent
- WRONG: a Contradiction node with only one has_contradiction edge, or a
  Contradiction whose `summary` just says "these conflict" without naming the
  specific conflicting detail from each side
- WRONG: using has_contradiction between two Topics, or relates_to's
  `contradicts` between two Descriptions — has_contradiction's endpoints are
  always Description -> Contradiction; contradicts' endpoints are always Topic
  -> Topic (or Feedback -> Idea/Decision), never mixed

==============================
GUIDED TYPE VOCABULARY (closed — Paper and Meeting Types are disjoint)
==============================

Use the casing EXACTLY as below. Do not invent a new Type; pick the closest one.

  PAPER:
    Background          — research background / context of the paper
    Problem              — problem the paper addresses
    Research Goal         — stated research objective / aim
    Theoretical Basis      — underlying theory the work builds on
    Dataset               — dataset / data used
    Conclusion            — concluding statement
    Future Work           — proposed future work
    Existing Research      — prior / related work
    Research Gap          — gap left by existing research
    Method                — method proposed by this paper
    Experiment            — what the experiment does
    Result                — result of the paper
    Metrics Evaluation     — metric / measure reported
    Limitation             — stated limitation

  MEETING:
    Issue           — problem or obstacle reported/observed that triggers discussion
    Idea            — proposed solution or approach under discussion, not yet agreed
    Decision        — final agreement reached on a direction or step to take
    Action Item     — concrete task following from a Decision, with or without an owner/deadline
    Open Question   — question raised but not yet answered or resolved
    Progress Update — status report on work completed or in progress since the previous meeting
    Feedback        — input or critique from a supervisor/participant on work presented

A meeting that reviews paper progress may legitimately contain Topics typed from
BOTH lists in the same document (e.g. Feedback on a Method) — that is expected.
What is not allowed is picking a Type from the wrong list for a Topic that
clearly fits a Type in the correct list (e.g. calling something "Problem" when
"Issue" fits better because the document is a meeting, not a paper).

For each Type you use: link one Topic via has_type, AND create its Description
(Type -> has_description -> Description).

==============================
OTHER EXTRACTION RULES
==============================

1. Merge aliases to the fullest name: "Prof. Chou" + "Shuo-Yan Chou" ->
   "Prof. Shuo-Yan Chou". Do not create separate nodes for the same concept
   (e.g. "Digital Twin" and "Digital Twin System" — pick one consistent name if
   the text shows they are the same concept; only separate them if the text
   explicitly distinguishes them).
2. ABBREVIATION: use the full name as the node id: "Robotic Mobile Fulfillment
   System (RMFS)" not "RMFS". If the full name is not in the text, keep the
   abbreviation and add property {{"abbreviation": "RMFS"}}.
3. Store numbers, dates, percentages as node properties — not as separate nodes.
4. Node ids: Title Case with spaces — never snake_case or ALL CAPS.

==============================
OUTPUT FORMAT
==============================

Return ONLY valid JSON. No markdown, no explanation, no text before or after the JSON.

{{
  "nodes": [
    {{
      "id": "node id",
      "type": "Agent | Role | Topic | Type | Source | Description | Contradiction",
      "properties": {{"name": "value"}}
    }}
  ],
  "relationships": [
    {{
      "source": "source node id",
      "target": "target node id",
      "type": "relationship_type",
      "properties": {{}}
    }}
  ]
}}

Example of a relates_to edge with its required property:
{{"source": "Proposed Augmentation Method", "target": "Data Sparsity", "type": "relates_to", "properties": {{"relation": "addresses"}}}}

Example of an assigned_to edge with an optional deadline:
{{"source": "Fix The Pipeline Bug", "target": "Budi", "type": "assigned_to", "properties": {{"due_date": "2026-07-27"}}}}

Example of a spoke_about edge with an optional stance:
{{"source": "Prof. Shuo-Yan Chou", "target": "Data Sparsity", "type": "spoke_about", "properties": {{"stance": "raised"}}}}

Example of a Contradiction node and its two required has_contradiction edges:
{{"id": "Contradiction Pile On Improvement", "type": "Contradiction",
  "properties": {{"summary": "Description A states pile-on improved 24%, Description B states 18.97%, for what the text presents as the same evaluated scenario."}}}}
{{"source": "Description Two Phase Assignment Method", "target": "Contradiction Pile On Improvement", "type": "has_contradiction", "properties": {{"level": "direct"}}}}
{{"source": "Description First Come First Served Method", "target": "Contradiction Pile On Improvement", "type": "has_contradiction", "properties": {{"level": "direct"}}}}

==============================
SELF-CHECK BEFORE SENDING OUTPUT (top 6 priorities, plus 5 conditional ones)
==============================

1. Recount: how many has_source edges are in my output? If more than 3, REMOVE
   the extras and reroute them as has_subtopic.
2. Is there any relationship where source id == target id? If yes, REMOVE it.
3. Is my Source id EXACTLY equal to "{source_name}"? (check character by character)
4. Is any Source the SOURCE of any relationship other than "Topic -> has_source ->
   Source"? If yes, the direction is wrong — fix it.
5. Is there any Description that is tautological, lacks a specific detail from the
   text, or duplicates another Description for the same Topic? If yes, rewrite,
   remove, or drop the extra Type.
6. Are all relationship names exactly one of the 11 fixed strings (no
   "has_method"-style variants, no "contradict_level1"-style variants),
   lowercase_snake_case, no "::"?
7. (only if you used relates_to) Does every relates_to edge have a `relation`
   property from the RELATION VOCABULARY list, with source/target Types matching
   the allowed pair for that relation?
8. (only if you used assigned_to) Is its source Topic actually typed Action
   Item? If the Topic's Type is anything else, remove the assigned_to edge.
9. (only if you used status or stance) Is the value one of the allowed options
   listed for that property? If not, remove the property rather than invent one.
10. (only if you created a Contradiction) Does it have at least TWO
    has_contradiction edges in THIS output, each from a Description (never a
    Topic), each with a `level` in direct/partial/apparent, and does `summary`
    name the specific conflicting detail from both sides rather than just
    asserting they disagree? If any of that fails, remove the Contradiction
    node and its edges rather than send an incomplete one.
11. (only if you used has_contradiction or relates_to's contradicts on the
    same pair of Topics) Did you check you are not encoding the identical
    single conflict twice without the text actually supporting both levels?

==============================
BEGIN EXTRACTION
==============================

INPUT TEXT:
{input_text}
"""

    return PromptTemplate.from_template(prompt)