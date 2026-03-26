from __future__ import annotations

from typing import Any, TypeVar

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from src.common.errors import ValidationPipelineError

T = TypeVar("T", bound=BaseModel)


def validate_payload(model_cls: type[T], payload: dict[str, Any]) -> T:
    """Validate with both Pydantic and JSON Schema for deterministic contracts."""
    try:
        obj = model_cls.model_validate(payload)
    except ValidationError as exc:
        raise ValidationPipelineError(str(exc)) from exc

    schema = model_cls.model_json_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(obj.model_dump()), key=lambda e: e.path)
    if errors:
        joined = "; ".join(error.message for error in errors)
        raise ValidationPipelineError(f"JSON schema validation failed: {joined}")

    return obj
