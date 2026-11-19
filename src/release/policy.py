"""Publication policy for the local fitness-participation ledger.

The policy file is deliberately small.  It describes the three required
controls:

``min_cell``
    A protected count below this value is a small cell.

``max_interval_width``
    A protected count must remain an interval of at least this many possible
    integer values after every visible release is taken into account.
    Counts are discrete, so the width of ``[lo, hi]`` is ``hi - lo + 1``.
    An interval narrower than this is a disclosure.

``merge_scope``
    Which dimensions may stay fixed while small cells are combined.  The
    fixture value ``district-and-period`` means a merge keeps period and
    district fixed and may combine course types.  It is not permission to
    collapse a district or a period.

Optional keys (all have safe defaults):

``protected_metrics``
    Metric names whose person counts are subject to the small-cell rule.
    Default: ``participants``, ``attended``, ``confirmed``.  ``eligible`` is
    the public denominator, not a protected headcount.

``rate_numerator`` / ``rate_denominator``
    Columns used for the attendance rate.

``rate_tolerance``
    Absolute tolerance for a published rate.  ``0`` means the rate is treated
    as an exact rational; a positive tolerance widens inference accordingly.

``max_count``
    A structural upper bound for any unregistered count.  Used only to keep
    interval arithmetic finite; it is public, not a ledger value.

``definition_version``
    Label recorded on every release sheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PROTECTED_METRICS = ("participants", "attended", "confirmed")
DEFAULT_POLICY = {
    "min_cell": 8,
    "max_interval_width": 4,
    "merge_scope": "district-and-period",
    "protected_metrics": list(DEFAULT_PROTECTED_METRICS),
    "rate_numerator": "attended",
    "rate_denominator": "eligible",
    "rate_tolerance": 0,
    "max_count": 10**6,
    "definition_version": "default",
}


def normalise_scope(value: Any) -> tuple[str, ...]:
    """Return the dimensions that must stay fixed for a merge.

    ``none`` means only whole cells may be released; ``all`` permits every
    grouping.  ``district-and-period`` fixes period and district, so course
    types may be collapsed.
    """

    if isinstance(value, Mapping):
        fixed = value.get("fixed", ())
        if isinstance(fixed, str):
            fixed = [fixed]
        return tuple(str(item) for item in fixed)
    if value is None:
        return ()
    text = str(value).strip().lower().replace("_", "-")
    if text in {"", "none", "cell", "cells", "cell-only"}:
        return ()
    if text in {"all", "*", "any"}:
        return ("*",)
    if "district-and-period" in text or "period-and-district" in text:
        return ("period", "district")
    if text in {"period", "period-only", "by-period"}:
        return ("period",)
    if text in {"district", "district-only", "by-district"}:
        return ("district",)
    if text in {"course-type-and-period", "course-and-period"}:
        return ("period", "course_type")
    parts = [part for part in text.replace("+", "-").split("-") if part]
    return tuple(parts)


def _parse_metrics(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_PROTECTED_METRICS
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, Iterable):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError("policy field 'protected_metrics' must be a list or string")
    items = [item for item in items if item]
    if not items:
        raise ValueError("policy field 'protected_metrics' must not be empty")
    return tuple(items)


@dataclass(frozen=True)
class Policy:
    min_cell: int = 8
    max_interval_width: int = 4
    merge_scope: str = "district-and-period"
    protected_metrics: tuple[str, ...] = DEFAULT_PROTECTED_METRICS
    rate_numerator: str = "attended"
    rate_denominator: str = "eligible"
    rate_tolerance: Fraction = Fraction(0)
    max_count: int = 10**6
    definition_version: str = "default"
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Policy":
        def positive_int(name: str, default: int) -> int:
            raw = value.get(name, default)
            try:
                number = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"policy field {name!r} must be an integer") from exc
            if number < 1:
                raise ValueError(f"policy field {name!r} must be positive")
            return number

        tolerance_raw = value.get("rate_tolerance", 0)
        try:
            tolerance = Fraction(str(tolerance_raw))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("policy field 'rate_tolerance' must be a number") from exc
        if tolerance < 0:
            raise ValueError("policy field 'rate_tolerance' must not be negative")
        return cls(
            min_cell=positive_int("min_cell", 8),
            max_interval_width=positive_int("max_interval_width", 4),
            merge_scope=str(value.get("merge_scope", DEFAULT_POLICY["merge_scope"])),
            protected_metrics=_parse_metrics(value.get("protected_metrics")),
            rate_numerator=str(value.get("rate_numerator", "attended")),
            rate_denominator=str(value.get("rate_denominator", "eligible")),
            rate_tolerance=tolerance,
            max_count=positive_int("max_count", 10**6),
            definition_version=str(value.get("definition_version", "default")),
            raw=dict(value),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: policy must be a JSON object")
        return cls.from_mapping(value)

    @property
    def fixed_merge_dimensions(self) -> tuple[str, ...]:
        return normalise_scope(self.merge_scope)

    @property
    def permit_merge(self) -> bool:
        return bool(self.fixed_merge_dimensions)

    def is_protected_metric(self, metric: str) -> bool:
        from .model import canonical_metric

        return canonical_metric(metric) in {canonical_metric(item) for item in self.protected_metrics}

    def protects_value(self, value: int | float | None) -> bool:
        """A value is protected when it is a small count.

        ``None`` is not zero.  Missing values are handled separately: they
        cannot be shown safe and become pending verification.
        """

        return value is not None and value < self.min_cell

    def interval_is_protected(self, lo: int, hi: int) -> bool:
        if hi < lo:
            raise ValueError("invalid interval")
        return (hi - lo + 1) >= self.max_interval_width

    def interval_width(self, lo: int, hi: int) -> int:
        return hi - lo + 1

    def digest(self) -> str:
        payload = json.dumps(self.raw or self.to_dict(), sort_keys=True,
                             ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_cell": self.min_cell,
            "max_interval_width": self.max_interval_width,
            "merge_scope": self.merge_scope,
            "protected_metrics": list(self.protected_metrics),
            "rate_numerator": self.rate_numerator,
            "rate_denominator": self.rate_denominator,
            "rate_tolerance": str(self.rate_tolerance),
            "max_count": self.max_count,
            "definition_version": self.definition_version,
        }

    def merge_levels(self, dimensions: Iterable[str] = ("period", "district", "course_type")) -> list[tuple[str, ...]]:
        """Group keys permitted by this policy, from coarse to fine."""

        dims = tuple(dimensions)
        fixed = set(self.fixed_merge_dimensions)
        levels: list[tuple[str, ...]] = [()]
        for mask in range(1, 1 << len(dims)):
            key = tuple(dim for index, dim in enumerate(dims) if mask & (1 << index))
            if fixed == {"*"} or fixed.issubset(set(key)):
                levels.append(key)
        result: list[tuple[str, ...]] = []
        for key in levels:
            if key not in result:
                result.append(key)
        return result

    def allows_group_key(self, key: Iterable[str]) -> bool:
        fixed = set(self.fixed_merge_dimensions)
        if fixed == {"*"}:
            return True
        return fixed.issubset(set(key))
