"""
`sanitize_graph` — the deterministic ontology enforcement.

Worth testing above everything else in this repo because it is the one place
that is BOTH stateful and order-dependent: `has_source_state`, `topic_registry`
and `source_meta_state` are created once per document and mutated by every
chunk's call in sequence. A change that looks local ("drop this edge earlier")
can therefore alter what a *later* chunk produces, and nothing else in the
pipeline would notice — the graph would just be quietly wrong.

Everything here is pure Python: no Neo4j, no LLM, no embeddings.
"""

import pytest

from src.graph.graph_model import (
    MAX_HAS_SOURCE,
    MEETING_PROCEDURE_TYPE,
    _canonical_id,
    _Graph,
    _Node,
    _Relationship,
    _type_domain,
    classify_expected_domain,
    sanitize_graph,
)

SOURCE = "meeting.jsonl"


def topic_graph(topic, type_names, *, with_has_source=True, description="A specific detail: 42 robots."):
    """One Topic with its Type(s), Description(s) and optionally a has_source."""
    if isinstance(type_names, str):
        type_names = [type_names]

    nodes = [_Node(id=topic, type="Topic", properties={}), _Node(id=SOURCE, type="Source", properties={})]
    rels = []
    for type_name in type_names:
        nodes.append(_Node(id=type_name, type="Type", properties={}))
        description_id = f"Description::{topic}::{type_name}"
        nodes.append(_Node(
            id=description_id,
            type="Description",
            properties={"text": description, "topicName": topic, "typeName": type_name},
        ))
        rels.append(_Relationship(source=topic, target=type_name, type="has_type", properties={}))
        rels.append(_Relationship(source=type_name, target=description_id, type="has_description", properties={}))
    if with_has_source:
        rels.append(_Relationship(source=topic, target=SOURCE, type="has_source", properties={}))
    return _Graph(nodes=nodes, relationships=rels)


def has_source_topics(graph):
    return sorted(r.source for r in graph.relationships if r.type == "has_source")


def rel_types(graph):
    return sorted(r.type for r in graph.relationships)


class TestHasSourceCap:
    """The 'document core' quota. Shared across chunks, so it is where an
    ordering bug does the most damage — see TestMeetingProcedure."""

    def test_cap_spans_chunks(self):
        state = {}
        seen = []
        for i in range(MAX_HAS_SOURCE + 2):
            out = sanitize_graph(topic_graph(f"Topic {i}", "Issue"), source_name=SOURCE, has_source_state=state)
            seen.extend(has_source_topics(out))
        assert len(seen) == MAX_HAS_SOURCE

    def test_without_shared_state_the_cap_is_per_call(self):
        # The documented fallback: state=None caps within one chunk only.
        seen = []
        for i in range(MAX_HAS_SOURCE + 2):
            out = sanitize_graph(topic_graph(f"Topic {i}", "Issue"), source_name=SOURCE)
            seen.extend(has_source_topics(out))
        assert len(seen) == MAX_HAS_SOURCE + 2

    def test_duplicate_edges_do_not_eat_the_quota(self):
        graph = topic_graph("Energy Use", "Issue")
        graph.relationships.append(
            _Relationship(source="Energy Use", target=SOURCE, type="has_source", properties={})
        )
        state = {}
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state=state)
        assert has_source_topics(out) == ["Energy Use"]
        assert state[next(iter(state))] == 1


class TestMeetingProcedure:
    """v8.5. A transcript's chunk 1 is greetings and 'can you hear me?', and the
    quota above is first-come — so without this the meeting's actual content can
    never claim a slot."""

    def test_type_is_tagged_as_meeting_domain(self):
        assert _type_domain(MEETING_PROCEDURE_TYPE) == "meeting"

    def test_procedural_topic_gets_no_has_source(self):
        out = sanitize_graph(
            topic_graph("Audio Check", MEETING_PROCEDURE_TYPE), source_name=SOURCE, has_source_state={}
        )
        assert has_source_topics(out) == []

    def test_procedural_topic_does_not_consume_a_slot(self):
        """The regression that matters: chunk 1 must leave all 3 slots for the
        content that follows it, not merely lose its own edge."""
        state = {}
        sanitize_graph(
            topic_graph("Audio Check", MEETING_PROCEDURE_TYPE), source_name=SOURCE, has_source_state=state
        )
        later = [
            has_source_topics(
                sanitize_graph(topic_graph(f"Finding {i}", "Result"), source_name=SOURCE, has_source_state=state)
            )
            for i in range(MAX_HAS_SOURCE)
        ]
        assert [t for edges in later for t in edges] == [f"Finding {i}" for i in range(MAX_HAS_SOURCE)]

    def test_topic_typed_procedural_AND_content_keeps_has_source(self):
        """'We agreed to move the weekly slot' is procedural in subject but is
        still a real Decision, so it stays eligible to be a top-level Topic."""
        out = sanitize_graph(
            topic_graph("Weekly Slot Change", [MEETING_PROCEDURE_TYPE, "Decision"]),
            source_name=SOURCE,
            has_source_state={},
        )
        assert has_source_topics(out) == ["Weekly Slot Change"]

    def test_procedural_topic_survives_as_a_node(self):
        """Barred from has_source, not deleted — the talk did happen."""
        out = sanitize_graph(
            topic_graph("Audio Check", MEETING_PROCEDURE_TYPE), source_name=SOURCE, has_source_state={}
        )
        assert "Audio Check" in [n.id for n in out.nodes]
        assert "has_type" in rel_types(out)


class TestRelationshipEnforcement:
    def test_self_loop_is_dropped(self):
        graph = topic_graph("Recursion", "Issue")
        graph.relationships.append(
            _Relationship(source="Recursion", target="Recursion", type="has_subtopic", properties={})
        )
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert "has_subtopic" not in rel_types(out)

    def test_dynamic_has_type_name_is_canonicalized(self):
        """A leaked 'has_decision' is renamed, not just rejected — this is what
        makes '10 fixed relationship names' true in practice."""
        graph = topic_graph("Go With Qwen", "Decision")
        for rel in graph.relationships:
            if rel.type == "has_type":
                rel.type = "has_decision"
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert "has_type" in rel_types(out)
        assert "has_decision" not in rel_types(out)

    def test_relates_to_without_a_relation_property_is_dropped(self):
        graph = topic_graph("A Method", "Method")
        graph.nodes.append(_Node(id="A Problem", type="Topic", properties={}))
        graph.relationships.append(
            _Relationship(source="A Method", target="A Problem", type="relates_to", properties={})
        )
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert "relates_to" not in rel_types(out)

    def test_relates_to_with_mismatched_endpoint_types_is_dropped(self):
        """'addresses' requires Method -> Problem; a Decision source is invalid."""
        graph = topic_graph("Some Decision", "Decision")
        graph.nodes.append(_Node(id="A Problem", type="Topic", properties={}))
        graph.nodes.append(_Node(id="Problem", type="Type", properties={}))
        graph.relationships.append(
            _Relationship(source="A Problem", target="Problem", type="has_type", properties={})
        )
        graph.relationships.append(_Relationship(
            source="Some Decision", target="A Problem", type="relates_to",
            properties={"relation": "addresses"},
        ))
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert "relates_to" not in rel_types(out)

    def test_assigned_to_requires_an_action_item(self):
        graph = topic_graph("Some Idea", "Idea")
        graph.nodes.append(_Node(id="Budi", type="Agent", properties={"name": "Budi"}))
        graph.relationships.append(
            _Relationship(source="Some Idea", target="Budi", type="assigned_to", properties={})
        )
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert "assigned_to" not in rel_types(out)

    def test_assigned_to_is_kept_on_an_action_item(self):
        graph = topic_graph("Fix The Pipeline", "Action Item")
        graph.nodes.append(_Node(id="Budi", type="Agent", properties={"name": "Budi"}))
        graph.relationships.append(
            _Relationship(source="Fix The Pipeline", target="Budi", type="assigned_to", properties={})
        )
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert "assigned_to" in rel_types(out)

    def test_invented_stance_value_drops_the_property_not_the_edge(self):
        graph = topic_graph("Data Sparsity", "Issue")
        graph.nodes.append(_Node(id="Prof Chou", type="Agent", properties={"name": "Prof Chou"}))
        graph.relationships.append(_Relationship(
            source="Prof Chou", target="Data Sparsity", type="spoke_about",
            properties={"stance": "grumbled_about"},
        ))
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        spoke = [r for r in out.relationships if r.type == "spoke_about"]
        assert len(spoke) == 1
        assert "stance" not in (spoke[0].properties or {})


class TestDescriptions:
    def test_empty_description_is_dropped(self):
        graph = topic_graph("Hollow", "Issue", description="   ")
        out = sanitize_graph(graph, source_name=SOURCE, has_source_state={})
        assert not [n for n in out.nodes if n.type == "Description"]

    def test_description_id_is_scoped_per_document(self):
        """v8.3 — without this, two documents covering one Topic x Type MERGE
        into a single node and cross-document conflict detection is impossible."""
        out_a = sanitize_graph(topic_graph("Shared Topic", "Result"), source_name="a.jsonl", has_source_state={})
        out_b = sanitize_graph(topic_graph("Shared Topic", "Result"), source_name="b.jsonl", has_source_state={})
        ids_a = {n.id for n in out_a.nodes if n.type == "Description"}
        ids_b = {n.id for n in out_b.nodes if n.type == "Description"}
        assert ids_a and ids_b
        assert not (ids_a & ids_b)


class TestTypeDomain:
    @pytest.mark.parametrize("type_name,expected", [
        ("Result", "paper"),
        ("Method", "paper"),
        ("Issue", "meeting"),
        ("Decision", "meeting"),
        ("Meeting Procedure", "meeting"),
        ("Vibes", "unknown"),
    ])
    def test_domain_tagging(self, type_name, expected):
        assert _type_domain(type_name) == expected


class TestExpectedDomain:
    """`classify_expected_domain` reads the controlled `source_kind` enum, NOT
    the free-text `doc_type` folder name — "gold_b1" is a paper corpus but says
    nothing of the sort, and this value is used to DELETE nodes."""

    @pytest.mark.parametrize("source_kind,expected", [
        ("pdf", "paper"),
        ("transcript", "meeting"),
        ("gold_b1", None),
        ("jsonl", None),
        (None, None),
    ])
    def test_classification(self, source_kind, expected):
        assert classify_expected_domain(source_kind) == expected


class TestDomainEnforcement:
    """A paper source cannot legitimately emit a meeting-domain Type. The prompt
    only lowers the odds (it is a soft hint and does not remove vocabulary), so
    this is where the guarantee actually lives — ~1-2% of gold_b1 Topics came
    back typed Claim / Action Item / Progress Update before it existed."""

    def _types(self, graph):
        return {n.id for n in graph.nodes if n.type == "Type"}

    def _descriptions(self, graph):
        return {n.id for n in graph.nodes if n.type == "Description"}

    def test_meeting_type_is_dropped_from_a_paper_source(self):
        out = sanitize_graph(
            topic_graph("Chunking Approach", "Claim"),
            source_name=SOURCE, has_source_state={}, expected_domain="paper",
        )
        assert self._types(out) == set()
        assert self._descriptions(out) == set()
        assert "has_type" not in rel_types(out)
        assert "has_description" not in rel_types(out)

    def test_the_topic_itself_survives(self):
        """Only the wrong-domain classification goes — the Topic is real content
        and keeps its place in the graph, including its has_source claim."""
        out = sanitize_graph(
            topic_graph("Chunking Approach", "Claim"),
            source_name=SOURCE, has_source_state={}, expected_domain="paper",
        )
        assert "Chunking Approach" in {n.id for n in out.nodes if n.type == "Topic"}
        assert has_source_topics(out) == ["Chunking Approach"]

    @pytest.mark.parametrize("expected_domain", ["meeting", None])
    def test_meeting_type_is_kept_when_the_source_is_not_a_paper(self, expected_domain):
        """A meeting legitimately produces both vocabularies (Feedback on a
        Method), and an unrecognised source_kind must not delete anything."""
        out = sanitize_graph(
            topic_graph("Chunking Approach", "Claim"),
            source_name=SOURCE, has_source_state={}, expected_domain=expected_domain,
        )
        assert self._types(out) == {"Claim"}
        assert len(self._descriptions(out)) == 1

    def test_paper_type_is_never_dropped_from_a_meeting_source(self):
        out = sanitize_graph(
            topic_graph("Retrieval Method", "Method"),
            source_name=SOURCE, has_source_state={}, expected_domain="meeting",
        )
        assert self._types(out) == {"Method"}

    def test_only_the_wrong_domain_type_of_a_mixed_topic_goes(self):
        out = sanitize_graph(
            topic_graph("Retrieval Method", ["Method", "Claim"]),
            source_name=SOURCE, has_source_state={}, expected_domain="paper",
        )
        assert self._types(out) == {"Method"}
        assert self._descriptions(out) == {f"Description Retrieval Method Method|{_canonical_id(SOURCE)}"}
        assert rel_types(out) == ["has_description", "has_source", "has_type"]

    def test_a_dropped_type_leaves_no_orphan_description(self):
        """The Description hangs off the Type, so dropping only the Type would
        leave a floating Description that anything scanning Description nodes
        directly (the conflict pass) would still pick up."""
        out = sanitize_graph(
            topic_graph("Chunking Approach", ["Claim", "Progress Update"]),
            source_name=SOURCE, has_source_state={}, expected_domain="paper",
        )
        assert self._descriptions(out) == set()
        assert out.relationships == [] or rel_types(out) == ["has_source"]
