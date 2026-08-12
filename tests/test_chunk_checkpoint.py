"""
The chunk-level checkpoint — what makes a crashed document resumable.

Everything asserted here is about surviving a process that dies without warning,
so the interesting properties are all "what is on disk RIGHT NOW", not what an
in-memory object thinks. No Neo4j, no LLM, no embeddings.
"""

import json

from src.ingestion.chunk_checkpoint import ChunkCheckpoint

DOC = "paper.jsonl"
DIGEST = "a" * 64


def checkpoint(tmp_path, name="chunk_checkpoint.json"):
    return ChunkCheckpoint(path=str(tmp_path / name))


def reopened(cp):
    """A fresh reader of the same file — i.e. what the next process would see."""
    return ChunkCheckpoint(path=str(cp.path))


class TestPersistence:
    def test_mark_done_hits_the_disk_immediately(self, tmp_path):
        """An in-memory-only update would defeat the entire point: the crash this
        exists for can happen on the very next line."""
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 3, digest=DIGEST)
        assert reopened(cp).done_chunk_ids(DOC, 1, DIGEST) == {3}

    def test_nothing_done_is_an_empty_set_not_an_error(self, tmp_path):
        assert checkpoint(tmp_path).done_chunk_ids(DOC, 1, DIGEST) == set()

    def test_no_partial_file_is_left_behind(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        assert [p.name for p in tmp_path.iterdir()] == ["chunk_checkpoint.json"]

    def test_a_corrupt_file_starts_empty_rather_than_blocking_the_build(self, tmp_path):
        path = tmp_path / "chunk_checkpoint.json"
        path.write_text("{not json", encoding="utf-8")
        assert ChunkCheckpoint(path=str(path)).done_chunk_ids(DOC, 1, DIGEST) == set()

    def test_string_chunk_ids_survive_the_round_trip(self, tmp_path):
        """Chunk.chunk_id is Union[int, str] — a producer emitting "doc::0000"
        must not make the checkpoint unsortable or lossy."""
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, "doc::0001", digest=DIGEST)
        cp.mark_done(DOC, 1, "doc::0000", digest=DIGEST)
        assert reopened(cp).done_chunk_ids(DOC, 1, DIGEST) == {"doc::0000", "doc::0001"}


class TestScoping:
    def test_versions_of_one_document_are_separate(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        assert cp.done_chunk_ids(DOC, 2, DIGEST) == set()

    def test_documents_are_separate(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        assert cp.done_chunk_ids("other.jsonl", 1, DIGEST) == set()

    def test_changed_content_invalidates_the_progress(self, tmp_path):
        """Chunk ids are positions. Nothing bumps document_version automatically,
        so without the digest check a revised .jsonl would resume onto chunk 40 of
        entirely different text."""
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        cp.mark_done(DOC, 1, 1, digest=DIGEST)
        assert cp.done_chunk_ids(DOC, 1, "b" * 64) == set()

    def test_marking_under_a_new_digest_discards_the_old_progress(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        cp.mark_done(DOC, 1, 7, digest="b" * 64)
        assert reopened(cp).done_chunk_ids(DOC, 1, "b" * 64) == {7}


class TestSanitizerState:
    """The chunk list alone cannot resume faithfully: `sanitize_graph`'s
    has_source cap and Topic registry span a document's chunks, so restarting
    them from empty would let a resumed document produce a graph an
    uninterrupted run never would."""

    STATE = {
        "has_source": {"Paper Jsonl": 3},
        "topics": {"Chunking Approach": "Chunking Approach"},
        "source_meta": {"Paper Jsonl": {"year": "2021", "date_raw": "2021-08"}},
    }

    def test_state_round_trips(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST, state=self.STATE)
        assert reopened(cp).sanitizer_state(DOC, 1, DIGEST) == self.STATE

    def test_absent_state_is_three_empty_dicts(self, tmp_path):
        state = checkpoint(tmp_path).sanitizer_state(DOC, 1, DIGEST)
        assert state == {"has_source": {}, "topics": {}, "source_meta": {}}

    def test_state_is_copied_not_aliased(self, tmp_path):
        """The caller mutates what it gets back for the rest of the document —
        it must not be writing into the checkpoint's own stored dict."""
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST, state=self.STATE)
        restored = cp.sanitizer_state(DOC, 1, DIGEST)
        restored["has_source"]["Paper Jsonl"] = 99
        restored["source_meta"]["Paper Jsonl"]["year"] = "1999"
        assert cp.sanitizer_state(DOC, 1, DIGEST) == self.STATE

    def test_changed_content_invalidates_the_state_too(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST, state=self.STATE)
        assert cp.sanitizer_state(DOC, 1, "b" * 64)["has_source"] == {}


class TestClear:
    def test_clear_removes_the_entry_from_disk(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        assert cp.clear_document(DOC, 1) is True
        assert reopened(cp).done_chunk_ids(DOC, 1, DIGEST) == set()

    def test_clearing_what_is_not_there_is_not_an_error(self, tmp_path):
        assert checkpoint(tmp_path).clear_document(DOC, 1) is False

    def test_clear_leaves_other_documents_alone(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        cp.mark_done("other.jsonl", 1, 0, digest=DIGEST)
        cp.clear_document(DOC, 1)
        assert reopened(cp).done_chunk_ids("other.jsonl", 1, DIGEST) == {0}


class TestFileShape:
    def test_written_file_is_versioned_and_readable(self, tmp_path):
        cp = checkpoint(tmp_path)
        cp.mark_done(DOC, 1, 0, digest=DIGEST)
        raw = json.loads(cp.path.read_text(encoding="utf-8"))
        assert raw["version"] == 1
        assert list(raw["documents"]) == [f"{DOC}|1"]
