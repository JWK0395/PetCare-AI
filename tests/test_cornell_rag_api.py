from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from petcare_rag.api import Readiness, create_app
    from petcare_rag.models import Citation, RagResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI 테스트 의존성이 설치되어야 합니다.")
class RagApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[dict[str, object]] = []

        def fake_answerer(**kwargs: object) -> RagResponse:
            self.calls.append(kwargs)
            return RagResponse(
                question=str(kwargs["question"]),
                species=str(kwargs["species"]),
                answer="Cornell 자료에 따른 일반 건강정보입니다. [1]",
                insufficient_evidence=False,
                citations=[
                    Citation(
                        number=1,
                        title="Chocolate toxicity",
                        section_path=["Chocolate toxicity", "Clinical signs"],
                        url="https://www.vet.cornell.edu/example",
                        chunk_id="cornell_dog_example_001",
                    )
                ],
                disclaimer="일반적인 공식 건강정보입니다.",
            )

        ready = lambda: Readiness(
            ready=True,
            checks={"database_exists": True},
            message="ready",
        )
        self.environment = patch.dict(
            os.environ,
            {"PETCARE_RAG_SERVICE_TOKEN": "team-test-token"},
            clear=False,
        )
        self.environment.start()
        self.client = TestClient(
            create_app(answerer=fake_answerer, readiness_checker=ready)
        )

    def tearDown(self) -> None:
        self.environment.stop()

    def test_health_does_not_call_rag(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(self.calls, [])

    def test_ready_reports_dependency_state(self) -> None:
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_answer_requires_service_token(self) -> None:
        response = self.client.post(
            "/v1/rag/answer",
            json={"question": "왜 위험해?", "species": "dog"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "unauthorized")
        self.assertEqual(self.calls, [])

    def test_answer_returns_existing_rag_contract(self) -> None:
        response = self.client.post(
            "/v1/rag/answer",
            headers={"X-PetCare-Token": "team-test-token"},
            json={
                "question": "강아지가 초콜릿을 먹으면 왜 위험해?",
                "species": "dog",
                "top_k": 5,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["species"], "dog")
        self.assertEqual(payload["citations"][0]["number"], 1)
        self.assertTrue(payload["citations"][0]["url"].startswith("https://"))
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("pet_id", self.calls[0])

    def test_invalid_species_is_rejected_before_rag(self) -> None:
        response = self.client.post(
            "/v1/rag/answer",
            headers={"X-PetCare-Token": "team-test-token"},
            json={"question": "질문", "species": "rabbit"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")
        self.assertEqual(self.calls, [])

    def test_personal_record_fields_are_rejected(self) -> None:
        response = self.client.post(
            "/v1/rag/answer",
            headers={"X-PetCare-Token": "team-test-token"},
            json={
                "question": "질문",
                "species": "dog",
                "pet_id": "private-pet-id",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
