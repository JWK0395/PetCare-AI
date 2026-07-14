from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "manage_cornell_rag_db.py"
SPEC = importlib.util.spec_from_file_location("cornell_rag_db", MODULE_PATH)
assert SPEC and SPEC.loader
rag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rag
SPEC.loader.exec_module(rag)


def row(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "chunk_id": "cornell_dog_test_001",
        "document_id": "cornell_dog_test",
        "title": "Test title",
        "section_path": ["Test title", "Signs"],
        "species": ["dog"],
        "categories": ["Emergency"],
        "canonical_url": "https://www.vet.cornell.edu/test",
        "last_updated": None,
        "source_institution": "Cornell University College of Veterinary Medicine",
        "source_center": "Cornell Riney Canine Health Center",
        "language": "en",
        "medical_domain": "canine_health",
        "content_hash": "abc123",
        "content": "# Test title\n\nUseful medical information.",
    }
    value.update(changes)
    return value


class CorpusTests(unittest.TestCase):
    def test_reads_valid_jsonl_and_hashes_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps(row()) + "\n", encoding="utf-8")
            corpus = rag.read_corpus(path, expected_count=1)
        self.assertEqual(len(corpus.rows), 1)
        self.assertEqual(len(corpus.sha256), 64)

    def test_rejects_missing_required_field(self) -> None:
        value = row()
        del value["canonical_url"]
        with self.assertRaisesRegex(rag.RagDbError, "canonical_url"):
            rag.validate_corpus_rows([value], expected_count=1)

    def test_rejects_duplicate_chunk_id(self) -> None:
        with self.assertRaisesRegex(rag.RagDbError, "중복 chunk_id"):
            rag.validate_corpus_rows([row(), row()], expected_count=2)

    def test_rejects_wrong_species_shape(self) -> None:
        with self.assertRaisesRegex(rag.RagDbError, "species"):
            rag.validate_corpus_rows([row(species=["dog", "cat"])], expected_count=1)


class FormattingTests(unittest.TestCase):
    def test_document_and_query_prompts(self) -> None:
        self.assertEqual(
            rag.document_embedding_text(row()),
            "title: Test title | text: # Test title\n\nUseful medical information.",
        )
        self.assertEqual(
            rag.query_embedding_text("  강아지 구토  "),
            "task: question answering | query: 강아지 구토",
        )

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(rag.RagDbError):
            rag.query_embedding_text("   ")

    def test_null_metadata_becomes_empty_string(self) -> None:
        metadata = rag.chroma_metadata(row())
        self.assertEqual(metadata["last_updated"], "")
        self.assertEqual(metadata["species"], ["dog"])
        self.assertEqual(metadata["embedding_model"], rag.MODEL)
        self.assertEqual(metadata["embedding_dimension"], rag.DIMENSION)


class EmbeddingTests(unittest.TestCase):
    def test_dimension_and_finite_values_are_checked(self) -> None:
        rag.validate_embeddings([[0.1] * rag.DIMENSION], 1)
        with self.assertRaisesRegex(rag.RagDbError, "차원"):
            rag.validate_embeddings([[0.1] * 10], 1)
        bad = [0.1] * rag.DIMENSION
        bad[3] = float("nan")
        with self.assertRaisesRegex(rag.RagDbError, "유효하지 않은 숫자"):
            rag.validate_embeddings([bad], 1)

    def test_retryable_error_retries_then_succeeds(self) -> None:
        calls = []
        delays = []

        class TemporaryError(Exception):
            status_code = 429

        def operation() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise TemporaryError("rate limited")
            return "ok"

        result = rag.call_with_retry(
            operation,
            sleep=delays.append,
            random_value=lambda: 0.0,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_permanent_error_is_not_retried(self) -> None:
        calls = []

        class PermanentError(Exception):
            status_code = 400

        def operation() -> None:
            calls.append(1)
            raise PermanentError("bad request")

        with self.assertRaises(PermanentError):
            rag.call_with_retry(operation, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)


class CollectionTests(unittest.TestCase):
    def test_collection_model_dimension_and_hash_must_match(self) -> None:
        expected = {
            "embedding_model": rag.MODEL,
            "embedding_dimension": rag.DIMENSION,
            "corpus_sha256": "same",
            "expected_chunks": 732,
        }
        rag.validate_collection_compatibility(dict(expected), expected)
        wrong = dict(expected, embedding_model="another-model")
        with self.assertRaisesRegex(rag.RagDbError, "--rebuild"):
            rag.validate_collection_compatibility(wrong, expected)

    def test_only_missing_or_changed_rows_are_pending(self) -> None:
        rows = [
            row(chunk_id="one", content_hash="a"),
            row(chunk_id="two", content_hash="b"),
            row(chunk_id="three", content_hash="c"),
        ]
        pending = rag.pending_rows(rows, {"one": "a", "two": "old"})
        self.assertEqual([item["chunk_id"] for item in pending], ["two", "three"])

    def test_query_uses_species_array_filter_and_explicit_embedding(self) -> None:
        class FakeCollection:
            def __init__(self) -> None:
                self.kwargs = None

            def query(self, **kwargs: object) -> dict[str, object]:
                self.kwargs = kwargs
                return {
                    "ids": [["chunk-1"]],
                    "documents": [["medical text"]],
                    "metadatas": [[{
                        "document_id": "doc-1",
                        "title": "Title",
                        "species": ["cat"],
                        "section_path": ["Title"],
                        "canonical_url": "https://www.vet.cornell.edu/test",
                    }]],
                    "distances": [[0.2]],
                }

        collection = FakeCollection()
        results = rag.query_collection(collection, [0.1] * rag.DIMENSION, "cat", 5)
        self.assertEqual(collection.kwargs["where"], {"species": {"$contains": "cat"}})
        self.assertIn("query_embeddings", collection.kwargs)
        self.assertNotIn("query_texts", collection.kwargs)
        self.assertEqual(results[0].similarity, 0.8)


class EvaluationTests(unittest.TestCase):
    def result(self, document_id: str, species: str = "dog") -> object:
        return rag.SearchResult(
            rank=1,
            distance=0.1,
            chunk_id="chunk",
            document="body",
            metadata={
                "document_id": document_id,
                "title": "Title",
                "canonical_url": "https://www.vet.cornell.edu/test",
                "species": [species],
            },
        )

    def test_case_passes_for_expected_document_and_species(self) -> None:
        case = {"expected_document_ids": ["wanted"], "species": "dog"}
        passed, failures = rag.score_case(case, [self.result("wanted")])
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_case_fails_for_wrong_document_or_species(self) -> None:
        case = {"expected_document_ids": ["wanted"], "species": "dog"}
        passed, failures = rag.score_case(case, [self.result("other", "cat")])
        self.assertFalse(passed)
        self.assertIn("기대 문서가 top-k에 없음", failures)
        self.assertIn("다른 종 결과가 섞임", failures)

    def test_gold_cases_require_unique_ids(self) -> None:
        case = {
            "case_id": "same",
            "query": "query",
            "species": "dog",
            "expected_document_ids": ["doc"],
            "top_k": 5,
        }
        with self.assertRaisesRegex(rag.RagDbError, "중복 case_id"):
            rag.validate_gold_cases([case, dict(case)])


@unittest.skipUnless(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("RUN_RAG_INTEGRATION") == "1",
    "GEMINI_API_KEY와 RUN_RAG_INTEGRATION=1일 때만 실제 Google API를 호출합니다.",
)
class GoogleApiIntegrationTests(unittest.TestCase):
    def test_real_google_embedding_is_768_dimensions(self) -> None:
        client = rag.google_client()
        embeddings = rag.embed_texts(
            client,
            [
                rag.query_embedding_text("강아지 구토"),
                rag.query_embedding_text("고양이 신장 질환"),
            ],
        )
        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(vector) == rag.DIMENSION for vector in embeddings))


if __name__ == "__main__":
    unittest.main()
