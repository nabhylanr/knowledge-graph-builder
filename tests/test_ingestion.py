"""
Ingestion: the producer contract, the meeting adapter, and the extraction gate.

The theme running through these is that two producers populate the same fields
with different intent, and the pipeline has to reconcile them without either
side knowing. Most of what is asserted here is a decision that was made once,
for a measured reason, and would be silently undone by a plausible-looking edit.

No Neo4j, no LLM, no embeddings.
"""

import json

import pytest

from src.graph.graph_model import MAX_HAS_SOURCE, _Graph, _Node, _Relationship
from src.ingestion.build_ledger import content_digest
from src.ingestion.chunk_checkpoint import ChunkCheckpoint
from src.ingestion.chunk_record import ChunkRecord
from src.ingestion.chunks_ingestor import ChunksIngestor
from src.ingestion.graph_miner import GraphMiner
from src.ingestion.meeting_chunks import (
    DEFAULT_ELIGIBLE_MIN_CHARS,
    parse_meeting_id,
    records_from_payload,
    speech_length,
)
from src.schema import Chunk, ProcessedDocument

MEETING_ID = "llm-bi-weekly-meeting-20260311-140550-meeting-recording-2a6b195a"


def meeting_payload(*texts, meeting_id=MEETING_ID):
    return {
        "metadata": {"meeting_id": meeting_id, "input": "D:\\x\\speaker_transcript.json",
                     "conversation_chunks": len(texts)},
        "chunks": [
            {"chunk_id": f"{meeting_id}:chunk:{i + 1:04d}", "meeting_id": meeting_id,
             "text": t, "speaker_ids": ["SPEAKER_00"], "start_ms": i * 1000, "end_ms": (i + 1) * 1000}
            for i, t in enumerate(texts)
        ],
    }


class TestEffectiveEligibility:
    """Maruf's chunker sets extraction_eligible=false for any chunk with a
    quality note, including cosmetic PDF damage. Taken at face value that drops
    625 chunks of readable paper text. Ours means 'there is nothing here'."""

    def test_cosmetic_note_is_overridden_back_to_eligible(self):
        record = ChunkRecord(
            text="Thisfinding suggests top management support matters.",
            doc_id="p", index=0, extraction_eligible=False, quality_notes=["pdf_ligature"],
        )
        assert record.effective_extraction_eligible is True

    def test_several_cosmetic_notes_together_are_still_overridden(self):
        record = ChunkRecord(
            text="Readable text.", doc_id="p", index=0, extraction_eligible=False,
            quality_notes=["pdf_ligature", "line_break_hyphenation"],
        )
        assert record.effective_extraction_eligible is True

    def test_non_cosmetic_note_is_honoured(self):
        """layout_fragment is a bibliography line — genuinely not worth extracting."""
        record = ChunkRecord(
            text="Chan, K. W., Yim, C. K., & Lam, S.", doc_id="p", index=0,
            extraction_eligible=False, quality_notes=["layout_fragment"],
        )
        assert record.effective_extraction_eligible is False

    def test_a_mix_of_cosmetic_and_not_is_honoured(self):
        record = ChunkRecord(
            text="x", doc_id="p", index=0, extraction_eligible=False,
            quality_notes=["pdf_ligature", "layout_fragment"],
        )
        assert record.effective_extraction_eligible is False

    def test_flag_with_no_notes_is_honoured(self):
        """Covers both the producer's table_artifact chunks and every chunk the
        meeting adapter marks — neither carries a note."""
        record = ChunkRecord(text="[unknown] you", doc_id="m", index=0, extraction_eligible=False)
        assert record.effective_extraction_eligible is False

    def test_eligible_stays_eligible(self):
        assert ChunkRecord(text="Anything.", doc_id="p", index=0).effective_extraction_eligible is True

    def test_to_chunk_carries_the_effective_value(self):
        record = ChunkRecord(text="Thisfinding.", doc_id="p", index=0,
                             extraction_eligible=False, quality_notes=["pdf_ligature"])
        assert record.to_chunk().extraction_eligible is True


class TestSpeechLength:
    def test_speaker_tags_do_not_count(self):
        """Six 'Okay.' lines measure 100+ characters with their tags attached —
        the whole point of the threshold is to measure speech, not scaffolding."""
        tagged = "\n".join("[SPEAKER_00] Okay." for _ in range(6))
        assert len(tagged) > DEFAULT_ELIGIBLE_MIN_CHARS
        assert speech_length(tagged) < DEFAULT_ELIGIBLE_MIN_CHARS

    def test_unknown_speaker_tag_is_stripped_too(self):
        assert speech_length("[unknown] you") == len("you")


class TestParseMeetingId:
    @pytest.mark.parametrize("meeting_id,date,series", [
        (MEETING_ID, "2026-03-11", "llm-bi-weekly-meeting"),
        ("rmfs-meeting-1023-20251023-123751-8c01ff86", "2025-10-23", "rmfs-meeting-1023"),
        ("aesop-20251009-180503-56a78b78", "2025-10-09", "aesop"),
        ("0423-harry-20260422-151149utc-meeting-recording-64e1626c", "2026-04-22", "0423-harry"),
    ])
    def test_reads_date_and_series(self, meeting_id, date, series):
        assert parse_meeting_id(meeting_id) == (date, series)

    def test_a_spelled_out_date_in_the_name_does_not_win(self):
        """'oral-defense-june-25-2026-20260625-...' — the 4-digit 2026 must not
        be mistaken for the timestamp."""
        date, series = parse_meeting_id("oral-defense-june-25-2026-20260625-125907-meeting-recording-a2")
        assert date == "2026-06-25"
        assert series == "oral-defense-june-25-2026"

    @pytest.mark.parametrize("meeting_id", ["", "no-date-here", "x-20261332-120000-y"])
    def test_unparseable_ids_yield_nothing_rather_than_a_guess(self, meeting_id):
        assert parse_meeting_id(meeting_id) == (None, None)


class TestRecordsFromPayload:
    def test_short_chunks_are_marked_not_dropped(self):
        records, _ = records_from_payload(meeting_payload("[unknown] you", "A" * 200))
        assert [r["extraction_eligible"] for r in records] == [False, True]

    def test_index_stays_one_to_one_with_the_transcript(self):
        """Nothing is dropped, so index never has to be renumbered across a gap
        and the NEXT chain matches the recording's real order."""
        records, _ = records_from_payload(meeting_payload("[unknown] you", "B" * 200, "[unknown] you"))
        assert [r["index"] for r in records] == [0, 1, 2]
        assert all(r["n_chunks"] == 3 for r in records)

    def test_date_and_series_land_on_every_record(self):
        records, _ = records_from_payload(meeting_payload("C" * 200))
        assert records[0]["date"] == "2026-03-11"
        assert records[0]["series"] == "llm-bi-weekly-meeting"

    def test_threshold_of_zero_marks_nothing(self):
        records, _ = records_from_payload(meeting_payload("[unknown] you"), eligible_min_chars=0)
        assert records[0]["extraction_eligible"] is True

    def test_missing_meeting_id_is_an_error_not_a_silent_regrouping(self):
        payload = {"metadata": {}, "chunks": [{"text": "hello"}]}
        with pytest.raises(ValueError, match="meeting_id"):
            records_from_payload(payload)

    def test_records_round_trip_through_the_producer_contract(self):
        records, _ = records_from_payload(meeting_payload("D" * 200, "[unknown] you"))
        parsed = [ChunkRecord.model_validate(json.loads(json.dumps(r))) for r in records]
        assert [p.effective_extraction_eligible for p in parsed] == [True, False]
        assert parsed[0].doc_metadata()["series"] == "llm-bi-weekly-meeting"


class TestChunksIngestor:
    def _write(self, tmp_path, folder, name, records):
        target = tmp_path / folder
        target.mkdir(parents=True, exist_ok=True)
        path = target / name
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return path

    def test_doc_type_comes_from_the_parent_folder(self, tmp_path):
        records, _ = records_from_payload(meeting_payload("E" * 200))
        path = self._write(tmp_path, "meeting", "m.jsonl", records)
        doc = ChunksIngestor().load_from_file(str(path))[0]
        assert doc.metadata["doc_type"] == "meeting"
        assert doc.metadata["date"] == "2026-03-11"

    def test_paper_folder_yields_no_date_or_series(self, tmp_path):
        path = self._write(tmp_path, "paper", "p.jsonl",
                           [{"text": "Body.", "doc_id": "p", "index": 0, "source_kind": "pdf"}])
        doc = ChunksIngestor().load_from_file(str(path))[0]
        assert doc.metadata["doc_type"] == "paper"
        assert "date" not in doc.metadata and "series" not in doc.metadata

    def test_chunks_group_by_doc_id_not_by_file(self, tmp_path):
        path = self._write(tmp_path, "paper", "two.jsonl", [
            {"text": "a", "doc_id": "one", "index": 0},
            {"text": "b", "doc_id": "two", "index": 0},
        ])
        assert len(ChunksIngestor().load_from_file(str(path))) == 2


class TestExtractionGate:
    """graph_miner is the only place that can decline: the extraction prompt
    requires 1-3 Topics per chunk, so a contentless chunk yields invented ones."""

    def _miner(self, calls):
        class Stub:
            def extract_graph(self, text, source_name, source_format, doc_type=None):
                calls.append(text)
                return None

        miner = GraphMiner.__new__(GraphMiner)
        miner.graph_extractor = Stub()
        return miner

    def test_ineligible_chunks_never_reach_the_llm(self):
        calls = []
        doc = ProcessedDocument(filename="m", metadata={}, chunks=[
            Chunk(chunk_id=0, text="you", extraction_eligible=False),
            Chunk(chunk_id=1, text="Real discussion of energy use.", extraction_eligible=True),
            Chunk(chunk_id=2, text="Thank you.", extraction_eligible=False),
        ])
        self._miner(calls).mine_graph_from_doc_chunks(doc)
        assert calls == ["Real discussion of energy use."]

    def test_skipped_chunks_are_kept_on_the_document(self):
        """They still need storing: store_chunks_for_doc writes text + embedding
        for every chunk, which is what keeps them retrievable and in the NEXT chain."""
        calls = []
        doc = ProcessedDocument(filename="m", metadata={}, chunks=[
            Chunk(chunk_id=0, text="you", extraction_eligible=False),
            Chunk(chunk_id=1, text="Real content.", extraction_eligible=True),
        ])
        self._miner(calls).mine_graph_from_doc_chunks(doc)
        assert len(doc.chunks) == 2

    def test_a_fully_ineligible_document_costs_no_llm_calls(self):
        calls = []
        doc = ProcessedDocument(filename="m", metadata={}, chunks=[
            Chunk(chunk_id=0, text="you", extraction_eligible=False),
        ])
        self._miner(calls).mine_graph_from_doc_chunks(doc)
        assert calls == []
        assert len(doc.chunks) == 1


class Crash(BaseException):
    """A process death, not an error: `except Exception` must not swallow it, or
    the resume path being tested here never happens in the first place."""


class FakeKnowledgeGraph:
    """Records what reached storage. `crash_on` kills the run the way a real
    crash would — after some chunks are written and before the rest are."""

    def __init__(self, crash_on=None):
        self.crash_on = crash_on
        self.stored = []
        self.relationships = []
        self.finalized = []

    def store_single_chunk(self, doc, chunk):
        if chunk.chunk_id == self.crash_on:
            raise Crash(f"died on chunk {chunk.chunk_id}")
        self.stored.append(chunk.chunk_id)
        self.relationships.extend(chunk.relationships or [])
        return None

    def finalize_document(self, doc, accumulated_source_props=None, source_node_id=None):
        self.finalized.append(doc.filename)


class TestStreamingIngest:
    """Extraction and storage interleaved, checkpointed per chunk.

    Before this, nothing reached Neo4j until every chunk of every pending
    document had been extracted: a crash at chunk 80 of 129 lost all 80, and a
    crash on document 3 of 5 lost documents 1 and 2 as well. Storage was already
    idempotent, so what resuming buys is the LLM calls — hours of them.
    """

    SOURCE = "paper.jsonl"

    def _graph_for(self, topic):
        """One Topic, typed, described, and claiming a has_source slot — enough
        to exercise the cross-chunk sanitizer state."""
        description = f"Description::{topic}::Method"
        return _Graph(
            nodes=[
                _Node(id=topic, type="Topic", properties={}),
                _Node(id="Method", type="Type", properties={}),
                _Node(id=description, type="Description", properties={
                    "text": "A specific detail: 42 robots.", "topicName": topic, "typeName": "Method",
                }),
                _Node(id=self.SOURCE, type="Source", properties={}),
            ],
            relationships=[
                _Relationship(source=topic, target="Method", type="has_type", properties={}),
                _Relationship(source="Method", target=description, type="has_description", properties={}),
                _Relationship(source=topic, target=self.SOURCE, type="has_source", properties={}),
            ],
        )

    def _miner(self, calls, graph=True):
        outer = self

        class Stub:
            def extract_graph(self, text, source_name, source_format, doc_type=None):
                calls.append(text)
                return outer._graph_for(f"Topic {text}") if graph else None

        miner = GraphMiner.__new__(GraphMiner)
        miner.graph_extractor = Stub()
        return miner

    def _doc(self, n=5, ineligible=()):
        return ProcessedDocument(
            filename=self.SOURCE,
            metadata={"source_kind": "pdf"},
            chunks=[
                Chunk(chunk_id=i, text=f"c{i}", embedding=[0.1],
                      extraction_eligible=i not in ineligible)
                for i in range(n)
            ],
        )

    def _checkpoint(self, tmp_path):
        return ChunkCheckpoint(path=str(tmp_path / "chunk_checkpoint.json"))

    def test_chunks_are_stored_as_their_own_extraction_lands(self, tmp_path):
        kg, cp = FakeKnowledgeGraph(), self._checkpoint(tmp_path)
        self._miner([]).mine_and_store_doc_chunks(self._doc(), kg, cp)
        assert kg.stored == [0, 1, 2, 3, 4]
        assert kg.finalized == [self.SOURCE]

    def test_a_completed_document_leaves_no_checkpoint_behind(self, tmp_path):
        """It is redundant with the ledger entry the caller writes next, and
        keeping it would grow this file without bound across a corpus."""
        cp = self._checkpoint(tmp_path)
        self._miner([]).mine_and_store_doc_chunks(self._doc(), FakeKnowledgeGraph(), cp)
        assert cp.documents() == []

    def test_ineligible_chunks_are_stored_but_never_extracted(self, tmp_path):
        calls = []
        kg = FakeKnowledgeGraph()
        self._miner(calls).mine_and_store_doc_chunks(
            self._doc(ineligible={0, 4}), kg, self._checkpoint(tmp_path)
        )
        # sorted(): extraction runs on a thread pool, so the ORDER calls land in
        # is not fixed. Only the storage order is (see _extract_in_order).
        assert sorted(calls) == ["c1", "c2", "c3"]
        assert sorted(kg.stored) == [0, 1, 2, 3, 4]

    def test_a_crash_keeps_the_chunks_already_written(self, tmp_path):
        cp = self._checkpoint(tmp_path)
        kg = FakeKnowledgeGraph(crash_on=3)
        with pytest.raises(Crash):
            self._miner([]).mine_and_store_doc_chunks(self._doc(), kg, cp)

        doc = self._doc()
        assert kg.stored == [0, 1, 2]
        assert cp.done_chunk_ids(self.SOURCE, 1, content_digest(doc)) == {0, 1, 2}
        assert kg.finalized == []  # the document is not finished, so it is not finalized

    def test_resuming_does_not_pay_for_the_chunks_already_done(self, tmp_path):
        cp = self._checkpoint(tmp_path)
        with pytest.raises(Crash):
            self._miner([]).mine_and_store_doc_chunks(self._doc(), FakeKnowledgeGraph(crash_on=3), cp)

        calls, kg = [], FakeKnowledgeGraph()
        self._miner(calls).mine_and_store_doc_chunks(self._doc(), kg, cp)

        assert sorted(calls) == ["c3", "c4"]  # the expensive part: 0-2 are never re-extracted
        assert kg.stored == [3, 4]
        assert kg.finalized == [self.SOURCE]
        assert cp.documents() == []

    def test_resume_does_not_reset_the_has_source_cap(self, tmp_path):
        """The cap spans a document. Restarting the sanitizer state from empty on
        resume would let an interrupted document end up with 2x MAX_HAS_SOURCE
        has_source edges — a graph no uninterrupted run could produce."""
        cp = self._checkpoint(tmp_path)
        before = FakeKnowledgeGraph(crash_on=4)
        with pytest.raises(Crash):
            self._miner([]).mine_and_store_doc_chunks(self._doc(n=8), before, cp)

        after = FakeKnowledgeGraph()
        self._miner([]).mine_and_store_doc_chunks(self._doc(n=8), after, cp)

        has_source = [r for r in before.relationships + after.relationships if r.type == "has_source"]
        assert len(has_source) == MAX_HAS_SOURCE

    def test_a_failed_extraction_is_left_for_the_resume_to_retry(self, tmp_path):
        """Its text and embedding are stored (re-storing is idempotent), but
        extraction failures are usually transient — a server blip, unparseable
        JSON — so the chunk is not marked done."""
        cp = self._checkpoint(tmp_path)
        kg = FakeKnowledgeGraph(crash_on=2)
        with pytest.raises(Crash):
            self._miner([], graph=False).mine_and_store_doc_chunks(self._doc(), kg, cp)

        assert kg.stored == [0, 1]
        assert cp.done_chunk_ids(self.SOURCE, 1, content_digest(self._doc())) == set()

    def test_revised_content_is_rebuilt_rather_than_resumed(self, tmp_path):
        """Chunk ids are positions, and nothing bumps document_version
        automatically — resuming onto different text would silently skip it."""
        cp = self._checkpoint(tmp_path)
        with pytest.raises(Crash):
            self._miner([]).mine_and_store_doc_chunks(self._doc(), FakeKnowledgeGraph(crash_on=3), cp)

        revised = self._doc()
        for chunk in revised.chunks:
            chunk.text = f"revised {chunk.text}"

        calls = []
        self._miner(calls).mine_and_store_doc_chunks(revised, FakeKnowledgeGraph(), cp)
        assert sorted(calls) == ["revised c0", "revised c1", "revised c2", "revised c3", "revised c4"]

    def test_it_works_without_a_checkpoint_at_all(self, tmp_path):
        kg = FakeKnowledgeGraph()
        self._miner([]).mine_and_store_doc_chunks(self._doc(), kg, checkpoint=None)
        assert kg.stored == [0, 1, 2, 3, 4]
        assert kg.finalized == [self.SOURCE]


class TestDomainHint:
    """`doc_type` is the folder the chunks came from. It reached the Document
    node but never the extraction prompt, which is why ~1-2% of Topics in the
    gold_b1 paper corpus came back typed with Meeting Types."""

    def _capture(self):
        seen = {}

        class Stub:
            def extract_graph(self, text, source_name, source_format, doc_type=None):
                seen["doc_type"] = doc_type
                seen["source_format"] = source_format
                return None

        miner = GraphMiner.__new__(GraphMiner)
        miner.graph_extractor = Stub()
        return miner, seen

    def test_the_folder_classification_reaches_the_extractor(self):
        miner, seen = self._capture()
        miner.mine_graph_from_doc_chunks(ProcessedDocument(
            filename="p.jsonl",
            metadata={"doc_type": "gold_b1", "source_kind": "pdf"},
            chunks=[Chunk(chunk_id=0, text="a")],
        ))
        assert seen == {"doc_type": "gold_b1", "source_format": "pdf"}

    def test_a_document_with_no_folder_classification_sends_none(self):
        miner, seen = self._capture()
        miner.mine_graph_from_doc_chunks(ProcessedDocument(
            filename="p.jsonl", metadata={}, chunks=[Chunk(chunk_id=0, text="a")],
        ))
        assert seen["doc_type"] is None

    @pytest.mark.parametrize("source_kind,expected", [
        ("pdf", "paper"),
        ("transcript", "meeting"),
        (None, None),
    ])
    def test_the_hard_domain_comes_from_source_kind_not_the_folder(self, source_kind, expected):
        doc = ProcessedDocument(
            filename="p.jsonl",
            metadata={"doc_type": "gold_b1", **({"source_kind": source_kind} if source_kind else {})},
            chunks=[Chunk(chunk_id=0, text="a")],
        )
        assert GraphMiner._doc_context(doc)[3] == expected


class TestContentDigest:
    def _doc(self, chunks):
        return ProcessedDocument(filename="d", chunks=chunks)

    def test_order_independent(self):
        a = Chunk(chunk_id=0, text="first")
        b = Chunk(chunk_id=1, text="second")
        assert content_digest(self._doc([a, b])) == content_digest(self._doc([b, a]))

    def test_text_change_changes_the_digest(self):
        assert content_digest(self._doc([Chunk(chunk_id=0, text="a")])) != \
               content_digest(self._doc([Chunk(chunk_id=0, text="b")]))

    def test_eligibility_is_deliberately_not_part_of_it(self):
        """Mixing it in would re-extract every already-built document the first
        time the flag is honoured — see the docstring on content_digest."""
        eligible = self._doc([Chunk(chunk_id=0, text="a", extraction_eligible=True)])
        ineligible = self._doc([Chunk(chunk_id=0, text="a", extraction_eligible=False)])
        assert content_digest(eligible) == content_digest(ineligible)
