from langchain.prompts import PromptTemplate


def get_graph_extractor_prompt() -> PromptTemplate:
    """
    Returns the KG extraction prompt — v6 (English).

    Tuned for meeting + paper datasets and for weaker/limited models. Works
    together with the deterministic post-processor `sanitize_graph`: the prompt
    lowers the *rate* of violations (preserving recall), the sanitizer *guarantees*
    the structure. Key changes from v5:

    1. has_source is now PROCEDURAL — the model must first choose the 1-3 main
       Topics (STEP A) rather than only obeying a passive "max 3" rule, which
       small models frequently violated.
    2. Guided Type vocabulary slimmed — "research gap / opportunity / problem
       formulation" all fold into "Research Problem" to cut category splitting.
    3. Source id is locked to a literal copy of {source_name} to prevent duplicate
       Source nodes from per-chunk spelling drift.
    4. Explicit self-loop ban (source id == target id).
    5. Relationship-direction table + 3 concrete WRONG examples mirroring real
       v5 failures (Source->Agent, Source->has_[type], Agent->Description).
    6. Self-check trimmed from 14 points to the 6 highest-priority ones — long
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
ONTOLOGY — RELATIONSHIPS (8 only, direction MUST match the table exactly)
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

All relationship names: lowercase snake_case. NO "::" or other symbols in a
relationship name (INVALID example: "has_description::industry_4_0::research_problem").

MISTAKES THAT HAVE HAPPENED BEFORE — DO NOT REPEAT:
- WRONG: Source -> writes_about -> Agent   (Source is never the source of writes_about)
- WRONG: Source -> has_research_problem -> Topic   (Source never has a has_[type] edge)
- WRONG: Agent -> has_[type]_description -> Description   (must come from Type, not Agent)
- WRONG: Topic X -> has_subtopic -> Topic X (same node)   (self-loop, FORBIDDEN:
  a relationship's source id and target id must NEVER be identical)

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

==============================
SELF-CHECK BEFORE SENDING OUTPUT (top 6 priorities)
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

==============================
BEGIN EXTRACTION
==============================

INPUT TEXT:
{input_text}
"""

    return PromptTemplate.from_template(prompt)
