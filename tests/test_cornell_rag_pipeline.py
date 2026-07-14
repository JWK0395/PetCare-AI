from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petcare_rag import (  # noqa: E402
    DEFAULT_DISCLAIMER,
    RagAnswer,
    RagPipelineError,
    RetrievedChunk,
    answer_question,
    build_context,
    generate_answer,
    retrieve,
)
from petcare_rag.pipeline import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    GENERATION_MODEL,
    RAG_ANSWER_SCHEMA,
    _decode_json_object,
    _safe_google_diagnostic,
    validate_generated_answer,
)


VECTOR = [0.1] * 768


def chunk(number: int = 1, species: str = "dog") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{number}",
        document_id=f"document_{number}",
        title=f"Title {number}",
        section_path=[f"Title {number}", "Signs"],
        species=[species],
        canonical_url=f"https://www.vet.cornell.edu/topic-{number}",
        content=f"Medical evidence {number}.",
        distance=0.1 * number,
    )


class FakeCollection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.kwargs: dict[str, object] | None = None
        self.rows = rows or [
            {
                "id": "chunk_1",
                "document": "Medical evidence 1.",
                "metadata": {
                    "document_id": "document_1",
                    "title": "Title 1",
                    "section_path": ["Title 1", "Signs"],
                    "species": ["dog"],
                    "canonical_url": "https://www.vet.cornell.edu/topic-1",
                },
                "distance": 0.1,
            }
        ]

    def query(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return {
            "ids": [[row["id"] for row in self.rows]],
            "documents": [[row["document"] for row in self.rows]],
            "metadatas": [[row["metadata"] for row in self.rows]],
            "distances": [[row["distance"] for row in self.rows]],
        }


class BrokenCollection:
    def query(self, **kwargs: object) -> dict[str, object]:
        raise OSError("internal database detail that must not escape")


class RequestAndRetrievalTests(unittest.TestCase):
    def test_empty_question_fails_before_embedder(self) -> None:
        calls = []
        with self.assertRaisesRegex(RagPipelineError, "질문"):
            retrieve(
                "  ",
                "dog",
                collection=FakeCollection(),
                embedder=lambda prompt: calls.append(prompt) or VECTOR,
            )
        self.assertEqual(calls, [])

    def test_species_filter_and_explicit_query_embedding_are_used(self) -> None:
        collection = FakeCollection()
        results = retrieve(
            "강아지 구토",
            "dog",
            collection=collection,
            embedder=lambda _: VECTOR,
        )
        self.assertEqual(len(results), 1)
        assert collection.kwargs is not None
        self.assertEqual(
            collection.kwargs["where"], {"species": {"$contains": "dog"}}
        )
        self.assertIn("query_embeddings", collection.kwargs)
        self.assertNotIn("query_texts", collection.kwargs)

    def test_non_cornell_and_wrong_species_rows_are_removed(self) -> None:
        rows = [
            {
                "id": "bad-url",
                "document": "body",
                "metadata": {
                    "document_id": "bad",
                    "title": "Bad",
                    "section_path": ["Bad"],
                    "species": ["dog"],
                    "canonical_url": "https://example.com/bad",
                },
                "distance": 0.1,
            },
            {
                "id": "wrong-species",
                "document": "body",
                "metadata": {
                    "document_id": "cat",
                    "title": "Cat",
                    "section_path": ["Cat"],
                    "species": ["cat"],
                    "canonical_url": "https://www.vet.cornell.edu/cat",
                },
                "distance": 0.2,
            },
        ]
        results = retrieve(
            "question",
            "dog",
            collection=FakeCollection(rows),
            embedder=lambda _: VECTOR,
        )
        self.assertEqual(results, [])

    def test_database_error_is_replaced_with_safe_message(self) -> None:
        with self.assertRaisesRegex(RagPipelineError, "ChromaDB 검색에 실패") as caught:
            retrieve(
                "question",
                "dog",
                collection=BrokenCollection(),
                embedder=lambda _: VECTOR,
            )
        self.assertNotIn("internal database detail", str(caught.exception))


class ContextTests(unittest.TestCase):
    def test_context_has_stable_source_numbers_and_metadata(self) -> None:
        context = build_context([chunk(1), chunk(2)])
        self.assertIn("[SOURCE 1]", context)
        self.assertIn("[SOURCE 2]", context)
        self.assertIn("Section: Title 1 > Signs", context)
        self.assertIn("https://www.vet.cornell.edu/topic-2", context)


class GenerationValidationTests(unittest.TestCase):
    def valid_payload(self) -> dict[str, object]:
        return {
            "answer": "Cornell 자료의 설명입니다. [1]",
            "cited_source_numbers": [1],
            "insufficient_evidence": False,
            "disclaimer": "model-created text",
        }

    def test_generate_content_schema_omits_unsupported_additional_properties(self) -> None:
        self.assertNotIn("additionalProperties", RAG_ANSWER_SCHEMA)
        self.assertNotIn("additional_properties", RAG_ANSWER_SCHEMA)

    def test_generation_uses_current_stable_free_tier_model(self) -> None:
        self.assertEqual(GENERATION_MODEL, "gemini-3.5-flash")
        self.assertEqual(DEFAULT_MAX_OUTPUT_TOKENS, 4096)

    def test_empty_evidence_does_not_call_generator(self) -> None:
        calls = []
        result = generate_answer(
            "question",
            "dog",
            [],
            generator=lambda *args: calls.append(args) or self.valid_payload(),
        )
        self.assertTrue(result.insufficient_evidence)
        self.assertEqual(result.cited_source_numbers, [])
        self.assertEqual(calls, [])

    def test_valid_answer_uses_fixed_disclaimer(self) -> None:
        result = validate_generated_answer(self.valid_payload(), 1)
        self.assertFalse(result.insufficient_evidence)
        self.assertEqual(result.disclaimer, DEFAULT_DISCLAIMER)

    def test_nonexistent_source_number_is_blocked(self) -> None:
        payload = self.valid_payload()
        payload["answer"] = "Unsupported. [2]"
        payload["cited_source_numbers"] = [2]
        with self.assertRaisesRegex(RagPipelineError, "존재하지 않는 SOURCE"):
            validate_generated_answer(payload, 1)

    def test_answer_url_is_blocked(self) -> None:
        payload = self.valid_payload()
        payload["answer"] = "See https://example.com [1]"
        with self.assertRaisesRegex(RagPipelineError, "URL"):
            validate_generated_answer(payload, 1)

    def test_visible_markers_normalize_mismatched_citation_list(self) -> None:
        payload = self.valid_payload()
        payload["answer"] = "First fact. [2] Repeated fact. [2] Other fact. [1]"
        payload["cited_source_numbers"] = [1]
        result = validate_generated_answer(payload, 2)
        self.assertEqual(result.cited_source_numbers, [2, 1])

    def test_answer_without_visible_marker_is_blocked(self) -> None:
        payload = self.valid_payload()
        payload["answer"] = "The answer has no marker."
        with self.assertRaisesRegex(RagPipelineError, "본문에.*인용이 없습니다"):
            validate_generated_answer(payload, 1)

    def test_invalid_declared_citation_is_still_blocked(self) -> None:
        payload = self.valid_payload()
        payload["cited_source_numbers"] = [99]
        with self.assertRaisesRegex(RagPipelineError, "존재하지 않는 SOURCE"):
            validate_generated_answer(payload, 1)

    def test_insufficient_output_is_replaced_with_safe_message(self) -> None:
        payload = {
            "answer": "A speculative answer that must be discarded.",
            "cited_source_numbers": [1],
            "insufficient_evidence": True,
            "disclaimer": "anything",
        }
        result = validate_generated_answer(payload, 1)
        self.assertTrue(result.insufficient_evidence)
        self.assertNotIn("speculative", result.answer)
        self.assertEqual(result.cited_source_numbers, [])

    def test_google_diagnostic_redacts_api_key(self) -> None:
        class GoogleError(Exception):
            status_code = 400

        secret = "test-api-key-secret-value"
        diagnostic = _safe_google_diagnostic(
            GoogleError(f"invalid x-goog-api-key={secret}")
        )
        self.assertIn("HTTP 400", diagnostic)
        self.assertNotIn(secret, diagnostic)
        self.assertIn("[REDACTED]", diagnostic)

    def test_fenced_json_response_is_parsed(self) -> None:
        payload = _decode_json_object(
            "```json\n{\"answer\": \"ok [1]\", \"cited_source_numbers\": [1], "
            "\"insufficient_evidence\": false, \"disclaimer\": \"text\"}\n```"
        )
        self.assertEqual(payload["answer"], "ok [1]")

    def test_json_after_leading_explanation_is_parsed(self) -> None:
        payload = _decode_json_object(
            "Here is the result:\n{\"answer\": \"ok [1]\", "
            "\"cited_source_numbers\": [1], \"insufficient_evidence\": false, "
            "\"disclaimer\": \"text\"}"
        )
        self.assertEqual(payload["cited_source_numbers"], [1])

    def test_malformed_response_keeps_safe_debug_preview(self) -> None:
        with self.assertRaises(RagPipelineError) as caught:
            _decode_json_object("not json at all")
        self.assertIn("응답 앞부분", caught.exception.diagnostic or "")


class WholePipelineTests(unittest.TestCase):
    def test_answer_question_returns_only_retrieved_citation_metadata(self) -> None:
        result = answer_question(
            "강아지 일반 건강 질문",
            "dog",
            collection=FakeCollection(),
            embedder=lambda _: VECTOR,
            generator=lambda *_: {
                "answer": "검색된 Cornell 근거에 따른 설명입니다. [1]",
                "cited_source_numbers": [1],
                "insufficient_evidence": False,
                "disclaimer": "ignored",
            },
        )
        self.assertEqual(result.species, "dog")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].url, "https://www.vet.cornell.edu/topic-1")
        self.assertEqual(result.citations[0].chunk_id, "chunk_1")

    def test_same_dependencies_produce_deterministic_response(self) -> None:
        kwargs = {
            "collection": FakeCollection(),
            "embedder": lambda _: VECTOR,
            "generator": lambda *_: RagAnswer(
                answer="Same answer. [1]",
                cited_source_numbers=[1],
                insufficient_evidence=False,
                disclaimer="ignored",
            ),
        }
        first = answer_question("question", "dog", **kwargs).to_dict()
        second = answer_question("question", "dog", **kwargs).to_dict()
        self.assertEqual(first, second)


@unittest.skipUnless(
    os.environ.get("GEMINI_API_KEY") and os.environ.get("RUN_RAG_INTEGRATION") == "1",
    "GEMINI_API_KEY와 RUN_RAG_INTEGRATION=1일 때만 실제 RAG API를 호출합니다.",
)
class GooglePipelineIntegrationTests(unittest.TestCase):
    def test_real_chroma_and_gemini_pipeline_returns_cornell_citation(self) -> None:
        result = answer_question(
            "강아지가 초콜릿을 먹으면 왜 위험한가?",
            "dog",
            top_k=5,
        )
        self.assertFalse(result.insufficient_evidence)
        self.assertTrue(result.citations)
        self.assertTrue(
            all(c.url.startswith("https://www.vet.cornell.edu/") for c in result.citations)
        )
        self.assertTrue(all(c.number >= 1 for c in result.citations))


if __name__ == "__main__":
    unittest.main()
