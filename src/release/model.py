"""Ledger and publication records for the offline release workflow.

Nothing in this module performs I/O to an external service.  The only
persisted artefacts are local CSV/JSON files and local delivery receipts.

Important modelling choices, all of which are deliberate:

* An empty CSV field is ``None`` (unregistered), never ``0``.  A missing
  denominator therefore cannot accidentally produce a zero attendance rate.
* A metric is identified by its unit, period, roster and definition, not by
  a report title.  ``participants`` and ``participant_visits`` are different
  quantities even when a spreadsheet heading calls both "参与人数".
* Counts whose rosters may overlap are not treated as additive.  If member
  identifiers are absent we can only bound a union total
  (``max(cells) <= union <= sum(cells)``).  This is recorded on the aggregate
  instead of silently summing the cells.
* A published row is visible if it was sent, if delivery is uncertain, or if
  it was retracted.  A retraction does not erase what a reader may have seen.
  Only a proven failure to send removes a row from visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Metric definitions


_METRIC_ALIASES = {
    "participants": "participants",
    "participant": "participants",
    "participant_count": "participants",
    "participants_count": "participants",
    "residents": "participants",
    "参与居民数": "participants",
    "参与人数": "participants",
    "参与居民": "participants",
    "attended": "attended",
    "attendees": "attended",
    "attendance": "attended",
    "actual_attendees": "attended",
    "实际到场人数": "attended",
    "实际到场": "attended",
    "到场人数": "attended",
    "confirmed": "confirmed",
    "confirmed_total": "confirmed",
    "已确认合计": "confirmed",
    "已确认": "confirmed",
    "确认合计": "confirmed",
    "eligible": "eligible",
    "eligible_count": "eligible",
    "registered": "eligible",
    "报名人数": "eligible",
    "报名": "eligible",
    "attendance_rate": "attendance_rate",
    "rate": "attendance_rate",
    "到场率": "attendance_rate",
    "出勤率": "attendance_rate",
    "participant_visits": "participant_visits",
    "visits": "participant_visits",
    "participation_visits": "participant_visits",
    "参与人次": "participant_visits",
    "人次": "participant_visits",
    "attended_visits": "attended_visits",
    "attendance_visits": "attended_visits",
    "到场人次": "attended_visits",
    "出席人次": "attended_visits",
    "confirmed_visits": "confirmed_visits",
    "确认人次": "confirmed_visits",
}


def canonical_metric(name: str) -> str:
    text = (name or "").strip()
    if text in _METRIC_ALIASES:
        return _METRIC_ALIASES[text]
    slug = text.lower().replace("-", "_").replace(" ", "_")
    return _METRIC_ALIASES.get(slug, slug or "unknown")


@dataclass(frozen=True)
class MetricDefinition:
    """Definition-level identity of a measurement.

    ``additive`` describes whether values may be summed over *roster-disjoint*
    units.  It is not a license to sum overlapping rosters.  ``attendance_rate``
    is a ratio of two count metrics.
    """

    name: str
    additive: bool
    kind: str = "count"
    numerator: str | None = None
    denominator: str | None = None

    @property
    def canonical(self) -> str:
        return canonical_metric(self.name)

    @classmethod
    def for_metric(cls, name: str, definition_version: str = "default") -> "MetricDefinition":
        canonical = canonical_metric(name)
        if canonical == "attendance_rate":
            return cls(name=canonical, additive=False, kind="rate",
                       numerator="attended", denominator="eligible")
        # Person-level counts can overlap between course types and periods.
        # Capacity / confirmation totals count slots and are additive unless a
        # roster check says otherwise.  Visit counts count attendances, so two
        # course types may legitimately sum even without a roster (a person
        # attending twice is two visits).  Unique-resident counts do not.
        additive = canonical in {"eligible", "confirmed", "participant_visits",
                                 "attended_visits", "confirmed_visits"}
        return cls(name=canonical, additive=additive)


DEFAULT_METRICS: dict[str, MetricDefinition] = {
    name: MetricDefinition.for_metric(name)
    for name in ("participants", "attended", "confirmed", "eligible", "attendance_rate",
                 "participant_visits", "attended_visits", "confirmed_visits")
}


UNIQUE_PERSON_METRICS = {"participants", "attended", "confirmed"}
VISIT_METRICS = {"participant_visits", "attended_visits", "confirmed_visits"}
COUNT_METRICS = set(DEFAULT_METRICS) - {"attendance_rate"}


# ---------------------------------------------------------------------------
# Cell identity and ledger


@dataclass(frozen=True, order=True)
class CellKey:
    period: str
    district: str
    course_type: str

    def as_dict(self) -> dict[str, str]:
        return {"period": self.period, "district": self.district, "course_type": self.course_type}

    def label(self) -> str:
        return f"{self.period}/{self.district}/{self.course_type}"


@dataclass
class CountCell:
    key: CellKey
    values: dict[str, int | None] = field(default_factory=dict)
    source_revision: str | None = None
    rosters: dict[str, frozenset[str]] = field(default_factory=dict)
    definition_version: str = "default"
    unit: str | None = None
    row_number: int | None = None

    def value(self, metric: str) -> int | None:
        return self.values.get(canonical_metric(metric))

    def known(self, metric: str) -> bool:
        return self.value(metric) is not None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            **self.key.as_dict(),
            "values": {key: value for key, value in sorted(self.values.items())},
            "source_revision": self.source_revision,
        }


@dataclass
class Aggregate:
    """Result of combining cells.

    ``exact`` is set only when a roster proof or the metric definition makes
    the union additive.  ``lower``/``upper`` are always the safe union bounds.
    ``visit_sum`` is the arithmetic sum and is a different metric from a
    unique-resident total when rosters may overlap.
    """

    cells: tuple[CellKey, ...]
    metric: str
    lower: int
    upper: int
    exact: int | None
    visit_sum: int
    roster_relation: str
    missing: tuple[CellKey, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def overlap_warning(self) -> bool:
        return self.roster_relation != "disjoint"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "cells": [cell.label() for cell in self.cells],
            "lower": self.lower,
            "upper": self.upper,
            "exact": self.exact,
            "visit_sum": self.visit_sum,
            "roster_relation": self.roster_relation,
            "missing": [cell.label() for cell in self.missing],
        }


class Ledger:
    """The current synthetic counts ledger, plus revision fingerprinting."""

    def __init__(self, cells: Iterable[CountCell], *, source: str | None = None,
                 raw_bytes: bytes | None = None, definition_version: str = "default"):
        self.cells: dict[CellKey, CountCell] = {cell.key: cell for cell in cells}
        self.source = source
        self.raw_bytes = raw_bytes
        self.definition_version = definition_version

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        path = Path(path)
        raw = path.read_bytes()
        cells: list[CountCell] = []
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: missing header")
            definitions = {}
            for row_number, row in enumerate(reader, start=2):
                key = CellKey(
                    period=(row.get("period") or "").strip(),
                    district=(row.get("district") or "").strip(),
                    course_type=(row.get("course_type") or "").strip(),
                )
                if not all((key.period, key.district, key.course_type)):
                    raise ValueError(f"{path}:{row_number}: incomplete cell key")
                values: dict[str, int | None] = {}
                rosters: dict[str, frozenset[str]] = {}
                for column, raw_value in row.items():
                    if not isinstance(column, str):
                        # csv.DictReader puts overflow fields under a ``None``
                        # key (or ``restkey``); they carry no metric name.
                        continue
                    if column in {"period", "district", "course_type", "source_revision",
                                  "definition_version", "roster_id", "member_ids",
                                  "unit", "roster", "covers_metric"}:
                        continue
                    metric = canonical_metric(column)
                    text = (raw_value or "").strip() if isinstance(raw_value, str) else ""
                    if text == "":
                        values[metric] = None
                    else:
                        try:
                            number = int(Fraction(text))
                        except (ValueError, ZeroDivisionError) as exc:
                            raise ValueError(
                                f"{path}:{row_number}: invalid {column} value {text!r}"
                            ) from exc
                        if number < 0:
                            raise ValueError(f"{path}:{row_number}: negative {column}")
                        values[metric] = number
                    definitions[metric] = MetricDefinition.for_metric(metric)
                members = (row.get("member_ids") or row.get("roster") or "")
                if isinstance(members, list):
                    members = ",".join(members)
                members = members.strip()
                if members:
                    roster = frozenset(part.strip() for part in members.replace(";", ",").split(",") if part.strip())
                    for metric in values:
                        rosters[metric] = roster
                elif (row.get("roster_id") or "").strip():
                    for metric in values:
                        rosters[metric] = frozenset({(row.get("roster_id") or "").strip()})
                cells.append(CountCell(
                    key=key,
                    values=values,
                    source_revision=(row.get("source_revision") or "").strip() or None,
                    rosters=rosters,
                    definition_version=(row.get("definition_version") or "default").strip(),
                    unit=(row.get("unit") or "").strip() or None,
                    row_number=row_number,
                ))
        return cls(cells, source=str(path), raw_bytes=raw,
                   definition_version=next((cell.definition_version for cell in cells), "default"))

    def keys(self) -> tuple[CellKey, ...]:
        return tuple(sorted(self.cells))

    def periods(self) -> tuple[str, ...]:
        return tuple(sorted({key.period for key in self.cells}))

    def districts(self) -> tuple[str, ...]:
        return tuple(sorted({key.district for key in self.cells}))

    def course_types(self) -> tuple[str, ...]:
        return tuple(sorted({key.course_type for key in self.cells}))

    def metrics(self) -> tuple[str, ...]:
        found: set[str] = set()
        for cell in self.cells.values():
            found.update(cell.values)
        return tuple(sorted(found))

    def cell(self, key: CellKey) -> CountCell:
        return self.cells[key]

    def value(self, key: CellKey, metric: str) -> int | None:
        cell = self.cells.get(key)
        return None if cell is None else cell.value(metric)

    def revision(self, key: CellKey) -> str | None:
        cell = self.cells.get(key)
        return None if cell is None else cell.source_revision

    def values_fingerprint(self) -> str:
        payload = []
        for key in self.keys():
            cell = self.cells[key]
            payload.append({
                "cell": key.label(),
                "values": sorted((metric, value) for metric, value in cell.values.items()),
                "revision": cell.source_revision,
            })
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        if self.raw_bytes is not None:
            return hashlib.sha256(self.raw_bytes).hexdigest()
        return self.values_fingerprint()

    def roster_relation(self, cells: Sequence[CellKey], metric: str) -> str:
        """Return ``disjoint`` only when member lists prove it.

        A roster identifier is a label, not a proof of disjointness.  Two
        different roster labels may still contain the same person.
        """

        metric = canonical_metric(metric)
        rosters = []
        for key in cells:
            cell = self.cells.get(key)
            if cell is None or metric not in cell.rosters:
                return "unknown"
            rosters.append(cell.rosters[metric])
        if len(rosters) <= 1:
            return "same"
        combined: set[str] = set()
        for roster in rosters:
            if not roster:
                return "unknown"
            if combined & set(roster):
                return "overlapping"
            combined |= set(roster)
        return "disjoint"

    def aggregate(self, cells: Sequence[CellKey], metric: str) -> Aggregate:
        metric = canonical_metric(metric)
        definition = MetricDefinition.for_metric(metric)
        values: list[int] = []
        missing: list[CellKey] = []
        for key in cells:
            value = self.value(key, metric)
            if value is None:
                missing.append(key)
            else:
                values.append(value)
        relation = self.roster_relation(cells, metric)
        if missing:
            return Aggregate(tuple(cells), metric, 0, 0, None, 0, relation, tuple(missing))
        total = sum(values)
        if values:
            lower = max(values)
        else:
            lower = 0
        # A single cell is its own roster, so it is exact regardless of the
        # metric's cross-cell additivity.  Across cells, only a definition or
        # roster proof of additivity gives an exact union.
        exact = total if (len(cells) <= 1 or definition.additive
                          or relation in {"disjoint", "same"}) else None
        return Aggregate(tuple(cells), metric, lower, total, exact, total, relation)

    def where(self, *, period: str | None = None, district: str | None = None,
              course_type: str | None = None) -> tuple[CellKey, ...]:
        result = []
        for key in self.keys():
            if period not in (None, "*") and key.period != period:
                continue
            if district not in (None, "*") and key.district != district:
                continue
            if course_type not in (None, "*") and key.course_type != course_type:
                continue
            result.append(key)
        return tuple(result)


# ---------------------------------------------------------------------------
# Facts and visibility


VISIBLE_STATUSES = {"sent", "delivered", "uncertain", "retracted", "withdrawn", "recalled"}
PENDING_STATUSES = {"draft", "pending", "approved", "scheduled", "queued", "review"}
INVISIBLE_STATUSES = {"failed", "not_sent", "cancelled", "canceled", "void"}


def _parse_boolish(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "delivered"}:
        return "yes"
    if text in {"0", "false", "no", "n", "undelivered"}:
        return "no"
    if text in {"", "unknown", "?"}:
        return "unknown"
    return text


@dataclass(frozen=True)
class Fact:
    """A statement that may be, or may become, visible to a reader."""

    source_id: str
    metric: str
    value: Fraction | None
    cells: tuple[CellKey, ...]
    kind: str = "cell"                 # cell | sum | union | rate
    status: str = "draft"
    delivered: str = "unknown"
    visible: bool = False
    seen_risk: bool = False
    revision: str | None = None
    definition_version: str = "default"
    unit: str | None = None
    roster: str | None = None
    numerator_metric: str | None = None
    denominator_metric: str | None = None
    denominator_cells: tuple[CellKey, ...] = ()
    destination: str | None = None
    evidence: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_metric(self) -> str:
        return canonical_metric(self.metric)

    @property
    def is_count(self) -> bool:
        return self.canonical_metric in COUNT_METRICS and self.kind != "rate"

    @property
    def is_rate(self) -> bool:
        return self.kind == "rate" or self.canonical_metric == "attendance_rate"

    def redacted(self, min_cell: int = 8) -> dict[str, Any]:
        """A copy without a small count.  Used for logs and explanations."""

        protected = bool(self.is_count and self.value is not None
                         and self.value < min_cell)
        return {
            "source": self.source_id,
            "metric": self.canonical_metric,
            "cells": [cell.label() for cell in self.cells],
            "value": None if protected else (
                float(self.value) if isinstance(self.value, Fraction) else self.value),
            "protected": protected,
            "kind": self.kind,
            "status": self.status,
            "visible": self.visible,
        }


def make_fact(*, source_id: str, metric: str, value: Any, cells: Sequence[CellKey],
              kind: str = "cell", status: str = "draft", delivered: str = "unknown",
              revision: str | None = None, definition_version: str = "default",
              unit: str | None = None, roster: str | None = None,
              numerator_metric: str | None = None, denominator_metric: str | None = None,
              denominator_cells: Sequence[CellKey] = (), destination: str | None = None,
              evidence: str | None = None, proof_unsent: bool = False,
              raw: Mapping[str, Any] | None = None) -> Fact:
    canonical_status = (status or "draft").strip().lower()
    delivery = _parse_boolish(delivered)
    if canonical_status in VISIBLE_STATUSES:
        visible = True
    elif canonical_status in PENDING_STATUSES:
        visible = False
    elif canonical_status in INVISIBLE_STATUSES:
        # A failure is only treated as non-delivery with an explicit proof.
        visible = not proof_unsent
    else:
        visible = False
    if delivery == "yes":
        visible = True
    # Anything a reader may have seen is a seen risk, including uncertain
    # delivery and retracted material.  Only proven non-delivery is excluded.
    seen_risk = visible or canonical_status in {"retracted", "withdrawn", "recalled"}
    numeric_value: Fraction | None
    if value is None or (isinstance(value, str) and not value.strip()):
        numeric_value = None
    else:
        try:
            numeric_value = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            numeric_value = None
    return Fact(
        source_id=source_id,
        metric=canonical_metric(metric),
        value=numeric_value,
        cells=tuple(cells),
        kind=kind,
        status=canonical_status,
        delivered=delivery,
        visible=visible,
        seen_risk=seen_risk,
        revision=revision,
        definition_version=definition_version,
        unit=unit,
        roster=roster,
        numerator_metric=(canonical_metric(numerator_metric) if numerator_metric else None),
        denominator_metric=(canonical_metric(denominator_metric) if denominator_metric else None),
        denominator_cells=tuple(denominator_cells),
        destination=destination,
        evidence=evidence,
        raw=dict(raw or {}),
    )


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    release_id: str
    material_id: str
    version_hash: str
    audience: str
    recipient: str
    status: str
    delivered: str
    sent_at: str | None
    received_at: str | None
    proof_unsent: bool = False
    note: str | None = None

    @property
    def visible(self) -> bool:
        status = (self.status or "").lower()
        delivered = _parse_boolish(self.delivered)
        if status in VISIBLE_STATUSES:
            return True
        if status in INVISIBLE_STATUSES:
            return self.proof_unsent is False
        if delivered == "yes":
            return True
        return False

    @property
    def uncertain(self) -> bool:
        return bool(self.visible and _parse_boolish(self.delivered) != "yes")


def load_published(path: str | Path, *, definition_version: str = "default") -> list[Fact]:
    """Load a publication log.

    The file has one row per cell in the fixture, but aggregate rows are
    accepted when any scope column contains ``*`` or ``all``.  A blank value
    is kept as ``None``; it is not interpreted as zero.
    """

    path = Path(path)
    facts: list[Fact] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        for row_number, row in enumerate(reader, start=2):
            release = (row.get("release") or row.get("release_id") or f"row-{row_number}").strip()
            metric = canonical_metric(row.get("metric") or "")
            period = (row.get("period") or "*").strip() or "*"
            district = (row.get("district") or "*").strip() or "*"
            course_type = (row.get("course_type") or row.get("course") or "*").strip() or "*"
            scope = (row.get("scope") or "").strip().lower()
            cells: list[CellKey] = []
            if scope in {"all", "total", "month-total", "period-total"}:
                cells = []
            elif "*" in {period, district, course_type} or scope in {"aggregate", "group"}:
                # Caller must attach the matching ledger to expand wildcards.
                cells = []
            else:
                cells = [CellKey(period, district, course_type)]
            numerator_metric = row.get("numerator_metric")
            denominator_metric = row.get("denominator_metric")
            denominator_cells: list[CellKey] = []
            if row.get("denominator_period") or row.get("denominator_district"):
                denominator_cells = [CellKey(
                    (row.get("denominator_period") or period).strip(),
                    (row.get("denominator_district") or district).strip(),
                    (row.get("denominator_course_type") or course_type).strip(),
                )]
            fact = make_fact(
                source_id=release,
                metric=metric,
                value=row.get("value"),
                cells=cells,
                kind=(row.get("kind") or ("cell" if len(cells) == 1 else
                                          ("range" if row.get("lower") or row.get("upper")
                                           else "sum"))),
                status=row.get("status") or "draft",
                delivered=row.get("delivered"),
                revision=(row.get("source_revision") or "").strip() or None,
                definition_version=(row.get("definition_version") or definition_version).strip(),
                unit=(row.get("unit") or "").strip() or None,
                roster=(row.get("roster") or row.get("roster_id") or "").strip() or None,
                numerator_metric=numerator_metric,
                denominator_metric=denominator_metric,
                denominator_cells=denominator_cells,
                destination=(row.get("destination") or "").strip() or None,
                evidence=(row.get("evidence") or "").strip() or None,
                proof_unsent=_parse_boolish(row.get("proof_unsent")) == "yes",
                raw={
                    key: value for key, value in row.items() if value is not None
                },
            )
            facts.append(fact)
    return facts


def scope_cells(period: str, district: str, course_type: str, ledger: Ledger) -> tuple[CellKey, ...]:
    return ledger.where(period=period, district=district, course_type=course_type)


def expand_fact(fact: Fact, ledger: Ledger) -> Fact:
    """Expand wildcard scopes against a ledger, preserving all other fields."""

    if fact.cells:
        return fact
    scope = fact.raw.get("scope", "")
    if isinstance(scope, str) and scope.lower() in {"all", "total", "month-total", "period-total"}:
        period = fact.raw.get("period", "*") or "*"
        cells = (ledger.where(period=period)
                 if period != "*" else ledger.keys())
    else:
        period = fact.raw.get("period", "*")
        district = fact.raw.get("district", "*")
        course_type = fact.raw.get("course_type", fact.raw.get("course", "*"))
        cells = scope_cells(period or "*", district or "*", course_type or "*", ledger)
    return Fact(
        source_id=fact.source_id,
        metric=fact.metric,
        value=fact.value,
        cells=tuple(cells),
        kind=fact.kind if fact.kind != "cell" or len(cells) != 1 else "cell",
        status=fact.status,
        delivered=fact.delivered,
        visible=fact.visible,
        seen_risk=fact.seen_risk,
        revision=fact.revision,
        definition_version=fact.definition_version,
        unit=fact.unit,
        roster=fact.roster,
        numerator_metric=fact.numerator_metric,
        denominator_metric=fact.denominator_metric,
        denominator_cells=fact.denominator_cells,
        destination=fact.destination,
        evidence=fact.evidence,
        raw=fact.raw,
    )


@dataclass
class VisibleState:
    facts: tuple[Fact, ...]
    pending: tuple[Fact, ...]
    conflicts: tuple[str, ...]

    @property
    def visible_facts(self) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.visible)

    def fingerprint(self) -> str:
        """Hash of the reader-visible statements, independent of duplicates.

        Sending the same material twice does not change what a reader can
        infer, so the fingerprint must not change either; otherwise an
        approval would be withdrawn for an event that reveals nothing.
        """

        entries = {
            (
                fact.source_id,
                fact.metric,
                tuple(cell.label() for cell in fact.cells),
                str(fact.value) if fact.value is not None else None,
                fact.kind,
                fact.status,
                fact.delivered,
            )
            for fact in self.visible_facts
        }
        payload = [
            {
                "source": source,
                "metric": metric,
                "cells": list(cells),
                "value": value,
                "kind": kind,
                "status": status,
                "delivered": delivered,
            }
            for source, metric, cells, value, kind, status, delivered
            in sorted(entries)
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_visible_state(facts: Iterable[Fact], ledger: Ledger | None = None) -> VisibleState:
    expanded: list[Fact] = []
    for fact in facts:
        expanded.append(expand_fact(fact, ledger) if ledger is not None else fact)
    conflicts: list[str] = []
    exact: dict[tuple[CellKey, str], Fraction] = {}
    for fact in expanded:
        if not fact.visible or fact.kind != "cell" or len(fact.cells) != 1 or fact.value is None:
            continue
        key = (fact.cells[0], fact.canonical_metric)
        previous = exact.get(key)
        if previous is not None and previous != fact.value:
            conflicts.append(
                f"{key[0].label()} {fact.canonical_metric}: visible values {previous} and {fact.value}"
            )
        exact[key] = fact.value
    return VisibleState(
        facts=tuple(expanded),
        pending=tuple(fact for fact in expanded if not fact.visible),
        conflicts=tuple(conflicts),
    )
