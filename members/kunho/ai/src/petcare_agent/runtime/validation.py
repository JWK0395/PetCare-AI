"""Runtime validation helpers for graph request/response contracts."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

from petcare_agent.contracts import load_json_schema
from petcare_agent.schemas.graph_state import GraphRequest, GraphResponse


class ContractValidationError(ValueError):
    """Raised when a runtime payload does not match a graph contract."""


def validate_graph_request_payload(payload: GraphRequest | Mapping[str, Any]) -> GraphRequest:
    """Validate a backend payload against pydantic and JSON schema contracts."""

    try:
        request = (
            payload if isinstance(payload, GraphRequest) else GraphRequest.model_validate(payload)
        )
        _validate_json_schema_payload(
            request.model_dump(mode="json"),
            load_json_schema("agent_graph_request"),
        )
    except (ValidationError, ContractValidationError) as exc:
        raise ContractValidationError("GraphRequest validation failed") from exc
    return request


def validate_graph_response_payload(
    payload: GraphResponse | Mapping[str, Any],
) -> GraphResponse:
    """Validate a graph response against pydantic and JSON schema contracts."""

    try:
        response = (
            payload if isinstance(payload, GraphResponse) else GraphResponse.model_validate(payload)
        )
        _validate_json_schema_payload(
            response.model_dump(mode="json"),
            load_json_schema("agent_graph_response"),
        )
    except (ValidationError, ContractValidationError) as exc:
        raise ContractValidationError("GraphResponse validation failed") from exc
    return response


def _validate_json_schema_payload(
    value: Any,
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any] | None = None,
    path: str = "$",
) -> None:
    root = root_schema or schema

    if "$ref" in schema:
        _validate_json_schema_payload(
            value,
            _resolve_ref(schema["$ref"], root),
            root_schema=root,
            path=path,
        )
        return

    if "oneOf" in schema:
        matched = 0
        for option in schema["oneOf"]:
            try:
                _validate_json_schema_payload(value, option, root_schema=root, path=path)
            except ContractValidationError:
                continue
            matched += 1
        if matched != 1:
            raise ContractValidationError(f"{path} matched {matched} oneOf branches")
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(value, expected_type, path)

    if "enum" in schema and value not in schema["enum"]:
        raise ContractValidationError(f"{path} is not an allowed enum value")

    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise ContractValidationError(f"{path} is shorter than minLength")

    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise ContractValidationError(f"{path} is smaller than minimum")

    if isinstance(value, float):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ContractValidationError(f"{path} is smaller than minimum")
        if maximum is not None and value > maximum:
            raise ContractValidationError(f"{path} is larger than maximum")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = required - set(value)
        if missing:
            raise ContractValidationError(f"{path} missing required fields: {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ContractValidationError(f"{path} has extra fields: {sorted(extra)}")
        for key, child_value in value.items():
            if key in properties:
                _validate_json_schema_payload(
                    child_value,
                    properties[key],
                    root_schema=root,
                    path=f"{path}.{key}",
                )

    if isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_json_schema_payload(
                    item,
                    item_schema,
                    root_schema=root,
                    path=f"{path}[{index}]",
                )


def _resolve_ref(ref: str, root_schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ContractValidationError(f"Only local JSON schema refs are supported: {ref}")

    current: Any = root_schema
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(current, Mapping) or part not in current:
            raise ContractValidationError(f"Unresolvable JSON schema ref: {ref}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise ContractValidationError(f"JSON schema ref is not an object: {ref}")
    return current


def _validate_type(value: Any, expected_type: str | list[str], path: str) -> None:
    expected_types = [expected_type] if isinstance(expected_type, str) else expected_type
    if any(_matches_type(value, type_name) for type_name in expected_types):
        return
    raise ContractValidationError(f"{path} is not of type {expected_types}")


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "object":
        return isinstance(value, Mapping)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    raise ContractValidationError(f"Unsupported JSON schema type: {type_name}")


__all__ = [
    "ContractValidationError",
    "validate_graph_request_payload",
    "validate_graph_response_payload",
]
