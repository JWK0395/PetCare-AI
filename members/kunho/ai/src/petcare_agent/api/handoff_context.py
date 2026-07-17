"""Adapter for the documented handoff-context API contract."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from petcare_agent.config import PetCareSettings, get_settings
from petcare_agent.schemas.graph_state import PetCareContext

HANDOFF_CONTEXT_DAYS = 3
HANDOFF_CONTEXT_PATH_TEMPLATE = "/api/pets/{pet_id}/handoff-context"
HANDOFF_CONTEXT_ENDPOINT_CONTRACT = "GET /api/pets/{pet_id}/handoff-context?days=3"
ALLOWED_EXISTING_API_ENDPOINTS = frozenset({HANDOFF_CONTEXT_PATH_TEMPLATE})


class HandoffContextAPIError(RuntimeError):
    """Raised when the existing handoff-context API cannot return context."""


class HandoffContextClient(Protocol):
    """HTTP client boundary used by the existing API provider."""

    def get_json(self, path: str, *, query: Mapping[str, int]) -> Mapping[str, Any]:
        """Return a JSON object from an existing PetCare API path."""


class UrllibHandoffContextClient:
    """Small stdlib HTTP client for the existing PetCare API.

    Tests should inject a mocked client/provider; this class exists for the
    runtime boundary and performs no work until get_json is called.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 5.0,
        settings: PetCareSettings | None = None,
    ) -> None:
        runtime_settings = settings or get_settings()
        self.base_url = (base_url or runtime_settings.petcare_api_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_json(self, path: str, *, query: Mapping[str, int]) -> Mapping[str, Any]:
        if not _is_allowed_handoff_context_path(path):
            raise HandoffContextAPIError(f"Unsupported existing API path: {path}")

        url = f"{self.base_url}{path}?{urlencode(dict(query))}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", response.getcode())
                if status >= 400:
                    raise HandoffContextAPIError(
                        f"handoff-context API returned HTTP {status}"
                    )
                body = response.read()
        except HandoffContextAPIError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise HandoffContextAPIError("handoff-context API request failed") from exc

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HandoffContextAPIError("handoff-context API returned invalid JSON") from exc

        if not isinstance(payload, Mapping):
            raise HandoffContextAPIError("handoff-context API payload must be an object")
        return payload


class ExistingAPIHandoffContextProvider:
    """DBContextProvider implementation for the documented existing API."""

    def __init__(self, client: HandoffContextClient | None = None) -> None:
        self.client = client or UrllibHandoffContextClient()

    def load_context(self, pet_id: int, *, days: int = HANDOFF_CONTEXT_DAYS) -> PetCareContext:
        if days != HANDOFF_CONTEXT_DAYS:
            raise HandoffContextAPIError(
                f"{HANDOFF_CONTEXT_ENDPOINT_CONTRACT} is the only Phase 10 context API"
            )

        payload = self.client.get_json(
            build_handoff_context_path(pet_id),
            query={"days": HANDOFF_CONTEXT_DAYS},
        )
        return coerce_handoff_context_payload(payload)


def build_handoff_context_path(pet_id: int) -> str:
    """Build the only existing API path allowed for graph DB context."""

    if pet_id <= 0:
        raise HandoffContextAPIError("pet_id must be positive")
    return HANDOFF_CONTEXT_PATH_TEMPLATE.format(pet_id=pet_id)


def coerce_handoff_context_payload(payload: PetCareContext | Mapping[str, Any]) -> PetCareContext:
    """Coerce the documented API response into graph context fields.

    The API response may include transport/runtime fields such as generated_at.
    Graph state does not add new schema fields, so top-level keys outside the
    existing PetCareContext contract are ignored.
    """

    if isinstance(payload, PetCareContext):
        return payload.model_copy(deep=True)

    payload_dict = deepcopy(dict(payload))
    allowed_fields = set(PetCareContext.model_fields)
    context_payload = {
        key: value for key, value in payload_dict.items() if key in allowed_fields
    }
    return PetCareContext.model_validate(context_payload)


def _is_allowed_handoff_context_path(path: str) -> bool:
    if not path.startswith("/api/pets/") or not path.endswith("/handoff-context"):
        return False
    pet_id_text = path.removeprefix("/api/pets/").removesuffix("/handoff-context")
    return pet_id_text.isdigit() and int(pet_id_text) > 0


__all__ = [
    "ALLOWED_EXISTING_API_ENDPOINTS",
    "HANDOFF_CONTEXT_DAYS",
    "HANDOFF_CONTEXT_ENDPOINT_CONTRACT",
    "HANDOFF_CONTEXT_PATH_TEMPLATE",
    "ExistingAPIHandoffContextProvider",
    "HandoffContextAPIError",
    "HandoffContextClient",
    "UrllibHandoffContextClient",
    "build_handoff_context_path",
    "coerce_handoff_context_payload",
]
