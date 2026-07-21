from langchain.prompts import PromptTemplate


def get_graph_extractor_prompt() -> PromptTemplate:
    """
    Returns the KG extraction prompt — v7 (English).

    Tuned for meeting + paper datasets and for weaker/limited models. Works
    together with the deterministic post-processor `sanitize_graph`: the prompt
    lowers the *rate* of violations (preserving recall), the sanitizer *guarantees*
    the structure. Changes from v6:

    8. Added `relates_to` (Topic -> Topic, typed via a `relation` property) so the
       graph can express assertions the flat has_[type]/has_subtopic edges
       cannot: "this Method addresses that Research Problem", "this Decision
       resolves that Issue". Without it, a query like "which paper addresses
       gap X" or "what did this decision resolve" has no edge to walk.
    9. Added `assigned_to` (Topic -> Agent) so an Action Item's owner is a real
       edge instead of buried in free text. `sanitize_graph` only keeps it when
       the source Topic is actually typed Action Item.

    Changes from v5 (still in effect):

    1. has_source is now PROCEDURAL — the model must first choose the 1-3 main
       Topics (STEP A) rather than only obeying a passive "max 3" rule, which
       small models frequently violated.
    2. Guided Type vocabulary slimmed — "research gap / opportunity / problem
       formulation" all fold into "Research Problem" to cut category splitting.
    3. Source id is locked to a literal copy of {source_name} to prevent duplicate
       Source nodes from per-chunk spelling drift.
    4. Explicit self-loop ban (source id == target id).
    5. Relationship-direction table + concrete WRONG examples mirroring real
       past failures (Source->Agent, Source->has_[type], Agent->Description).
    6. Self-check trimmed to the highest-priority points — long
       end-of-completion checklists are often ignored by small models.
    7. Descriptions must carry at least one specific detail from the text
       (number, name, technical term) — no generic tautological sentences.

    HONEST NOTE: a better prompt reduces many issues, but hard numeric constraints
    and self-loops may still leak partially on a model as small as
    llama-3.1-8b-instant. Anything that leaks is caught deterministically by
    `sanitize_graph` — do not keep rewriting the prompt to chase it.
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
   is only mentioned as a reference / related work / other authors' literature.
   These are the ONLY Topics allowed to have a has_source edge. Every other Topic
   MUST connect to the graph via has_subtopic (a chain from one of these 1-3
   Topics) and MUST NOT have its own has_source.

STEP B — For each Topic, decide its Type from the vocabulary below. If two Type
   candidates look similar, PICK THE CLOSEST ONE — do not invent a new Type for a
   small nuance.

STEP C — Only after STEP A and B: look for two Topics the text EXPLICITLY
   connects to each other (not just co-occurring in the same paragraph) and add
   a relates_to edge with a relation property from the RELATION VOCABULARY
   below. Example trigger phrases: "to address this gap, we propose...",
   "the team decided X, which resulted in...", "this contradicts the earlier
   finding that...". Do NOT force a relates_to between Topics that merely
   appear near each other — skip this step entirely if nothing in the text
   states a connection. Separately, if a Topic's Type is Action Item and the
   text names who is responsible, add an assigned_to edge from that Topic to
   the Agent; if a deadline or date is mentioned, add it as a due_date property
   on that edge.

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

==============================
ONTOLOGY — NODE TYPES (6 only)
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
   Required property: name. Optional: abbreviation, chapterNumber, tableNumber, figureNumber

4. Type — A semantic category classifying a Topic's function.
   Type classifies — it is NOT the content itself.
   Good: Method, Research Problem, Metric, Result, Contribution, Limitation,
       Future Work, System Component, Optimization Objective, Background, Dataset,
       Experimental Setup.
   Bad (use Topic instead): Order Batching, RMFS, Throughput Rate, FCFS, Table 4.3.
   Quick test: "Is this a semantic category or the actual content?" -> if content, use Topic.
   Naming: singular, Title Case, consistent ("Method" not "Methods").
   Required property: name

5. Source — The uploaded file the information is extracted from. See SOURCE ID RULES above.
   Required property: name. Optional: format

6. Description — A textual explanation of why a Topic belongs to a given Type.
   Specific to one Topic-Type pair. Minimum 2 sentences, and MUST contain at least
   ONE specific detail from the text (a number, name, technical term, or measured
   result). Do NOT write generic tautological sentences like "X is a method." or
   "Y is a research problem." — such sentences carry no information and are FORBIDDEN.
   id format: Description::<TopicName>::<TypeName>
   Required properties: text, topicName, typeName
   Linked FROM the Type node (Type -> has_[type]_description -> Description).
   NEVER from a Topic directly, NEVER from an Agent directly.

==============================
ONTOLOGY — RELATIONSHIPS (10 only, direction MUST match the table exactly)
==============================

| Relationship            | Source Node | Target Node | Meaning                                 |
|-------------------------|-------------|-------------|-----------------------------------------|
| role_in_meeting         | Agent       | Role        | Agent had this role in a meeting        |
| role_in_paper           | Agent       | Role        | Agent contributed to a paper            |
| spoke_about             | Agent       | Topic       | Agent verbally discussed this Topic     |
| writes_about            | Agent       | Topic       | Agent wrote about this Topic            |
| has_source              | Topic       | Source      | Only the 1-3 top-level Topics (STEP A)  |
| has_[type]              | Topic       | Type        | replace [type] with lowercase_underscore|
| has_[type]_description  | Type        | Description | direction: Type -> Description, NOT Topic -> Description |
| has_subtopic            | Topic       | Topic       | broader -> narrower                     |
| relates_to              | Topic       | Topic       | explicit semantic link (STEP C); requires a `relation` property from the vocabulary below |
| assigned_to             | Topic       | Agent       | ONLY when the Topic's Type is Action Item; the Agent is the owner  |

All relationship names: lowercase snake_case. NO "::" or other symbols in a
relationship name (INVALID example: "has_description::industry_4_0::research_problem").

==============================
RELATION VOCABULARY (for relates_to only — pick exactly one, lowercase)
==============================

  addresses       — a Method/Proposal answers a Research Problem/Issue
  resolves        — a Decision settles an Issue/Open Question
  produces        — a Decision generates an Action Item
  evaluates       — an Experimental Setup/Metric measures a Result
  follows_up_on   — a later Status Update follows an earlier Status Update/Action Item
  motivates       — a Background/Research Problem is the reason a Method exists
  contradicts     — a Result conflicts with another Result, or Feedback with a Proposal
  identifies      — a Background/Method names a Research Problem (gap-finding)

If none of these describe the connection, do NOT add a relates_to edge — do not
invent a ninth value.

MISTAKES THAT HAVE HAPPENED BEFORE — DO NOT REPEAT:
- WRONG: Source -> writes_about -> Agent   (Source is never the source of writes_about)
- WRONG: Source -> has_research_problem -> Topic   (Source never has a has_[type] edge)
- WRONG: Agent -> has_[type]_description -> Description   (must come from Type, not Agent)
- WRONG: Topic X -> has_subtopic -> Topic X (same node)   (self-loop, FORBIDDEN:
  a relationship's source id and target id must NEVER be identical)
- WRONG: Agent -> assigned_to -> Topic   (must be Topic -> Agent, not reversed)
- WRONG: a relates_to edge with no relation property, or a relation value
  outside the RELATION VOCABULARY list above
- WRONG: assigned_to on a Topic whose Type is Decision, Research Problem, or
  anything other than Action Item

==============================
GUIDED TYPE VOCABULARY (open, you need not use them all)
==============================

There is NO minimum number of Types. Extract ONLY the Types the text truly supports.
Use the casing EXACTLY as below. You may create a new Type ONLY if none fit
(Title Case with spaces).

  PAPER / THESIS: Research Problem (this ALSO covers research gap, research
      opportunity, problem formulation — ALL fold into this one, do not split them
      into separate Types), Background, Method, Dataset, Experimental Setup, Result,
      Metric, Contribution, Limitation, Future Work, System Component,
      Optimization Objective.
  MEETING / DISCUSSION: Decision, Action Item, Discussion, Proposal, Open Question,
      Status Update, Risk, Milestone, Project, Requirement.
  SHARED (either context): Method, Dataset, Result, Tool, Concept.

For each Type you use: link one Topic via has_[type], AND create its Description
(Type -> has_[type]_description -> Description).

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
      "type": "Agent | Role | Topic | Type | Source | Description",
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

==============================
SELF-CHECK BEFORE SENDING OUTPUT (top 6 priorities, plus 2 conditional ones)
==============================

1. Recount: how many has_source edges are in my output? If more than 3, REMOVE
   the extras and reroute them as has_subtopic.
2. Is there any relationship where source id == target id? If yes, REMOVE it.
3. Is my Source id EXACTLY equal to "{source_name}"? (check character by character)
4. Is any Source the SOURCE of any relationship other than "Topic -> has_source ->
   Source"? If yes, the direction is wrong — fix it.
5. Is there any Description that is tautological / lacks a specific detail from the
   text? If yes, rewrite or remove it.
6. Are all relationship names lowercase_snake_case with no "::"?
7. (only if you used relates_to) Does every relates_to edge have a `relation`
   property from the RELATION VOCABULARY list, with nothing invented?
8. (only if you used assigned_to) Is its source Topic actually typed Action
   Item? If the Topic's Type is anything else, remove the assigned_to edge.

==============================
BEGIN EXTRACTION
==============================

INPUT TEXT:
{input_text}
"""

    return PromptTemplate.from_template(prompt)