"""Shared YAML-to-dataclass decoding and scalar validation for algorithm configs.

Used by experiment envelope parsing and per-algorithm typed configuration decoders.
New algorithms should reuse these helpers instead of copying validation logic.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping, get_origin, get_type_hints


def mapping(value: object, label: str) -> dict[str, object]:
    """Coerce one untrusted mapping into a string-keyed dict."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def positive_integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    """Reject bools, floats, and out-of-range integers at config boundaries."""
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def finite_number(value: object, label: str) -> float:
    """Reject bools and non-finite numeric values at config boundaries."""
    import math

    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def decode_dataclass(cls: type, value: object, label: str):
    """Decode one mapping into a typed configuration dataclass.

    Rejects unknown keys, coerces YAML lists to tuples where annotated, and does
    not call ``validate()`` — callers should validate after decode.
    """
    values = mapping(value, label)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in values:
            continue
        expected = hints[field.name]
        if get_origin(expected) is tuple and isinstance(values[field.name], list):
            values[field.name] = tuple(values[field.name])
    return cls(**values)


__all__ = ["decode_dataclass", "finite_number", "mapping", "positive_integer"]
