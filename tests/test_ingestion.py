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

from src.ingestion.build_ledger import content_digest
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
            def extract_graph(self, text, source_name, source_format):
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
