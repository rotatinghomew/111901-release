"""Offline release workflow.

This module ties the ledger, policy and planner to the practical steps an
editor performs:

1. prepare a per-period release sheet (facts + definition version + scopes +
   destinations);
2. get it approved against the material that is *actually visible* at that
   moment;
3. schedule and send it through a local outbox only -- no mail service is
   called;
4. record a local delivery receipt for every recipient, so the manifest can
   say which content version a partner received, not merely how many files
   were produced.

The guards that matter for re-identification are implemented here rather than
left to operator discipline:

* An approval records the visible-state fingerprint it was checked against.
  If anything else becomes visible before send time (revision, resend, another
  brief), the approval is re-checked against the new state.  It is withdrawn
  if the send would now be unsafe; if it is still safe, the approval is
  re-pinned to the new state so the next change is measured against what the
  editor could actually rely on.
* Scheduling commits the content, so a schedule is judged jointly with
  anything already scheduled for an overlapping reader.  An approval alone is
  not a commitment: each editor may review a sheet that is safe on its own.
  Once one material is actually sent, every other approval is re-checked
  against the enlarged visible state.
* Retraction never removes a fact from the visible set.  Only a proven failed
  delivery does, and only once every receipt for that material is proven
  undelivered.
* Uncertain local delivery counts as visible until proven otherwise.
* The internal report is evidence, not distribution.  It may keep every
  value and never enters the reader-visible set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fractions import Fraction
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .model import (
    CellKey,
    Fact,
    Ledger,
    Receipt,
    build_visible_state,
    canonical_metric,
    expand_fact,
    load_published,
    make_fact,
)
from .planner import (
    INTERNAL,
    BatchVerdict,
    Planner,
    metric_values_missing,
)
from .policy import Policy


DRAFT = "draft"
APPROVED = "approved"
SCHEDULED = "scheduled"
SENT = "sent"
WITHDRAWN = "withdrawn"
SUPERSEDED = "superseded"
RETRACTED = "retracted"
FAILED = "failed"

INTERNAL_AUDIENCE = "internal"
PUBLIC_AUDIENCE = "external-public"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LineItem:
    """A single line on a release sheet.

    A line is one of:

    * an exact value (``value`` set);
    * a published range for a unique-person total whose rosters are not
      proven disjoint (``lower``/``upper`` set, ``value`` absent);
    * a suppressed line: the cell or group is named, but no value is carried.

    An external public sheet never stores a protected value, so its content
    hash cannot leak one.
    """

    source_id: str
    metric: str
    cells: tuple[CellKey, ...]
    value: Fraction | None
    kind: str = "cell"
    numerator_metric: str | None = None
    denominator_metric: str | None = None
    denominator_cells: tuple[CellKey, ...] = ()
    suppressed: bool = False
    lower: int | None = None
    upper: int | None = None

    @property
    def canonical_metric(self) -> str:
        return canonical_metric(self.metric)

    def payload(self) -> dict[str, Any]:
        data = {
            "source": self.source_id,
            "metric": self.canonical_metric,
            "cells": [cell.label() for cell in self.cells],
            "kind": self.kind,
            "suppressed": self.suppressed,
        }
        if self.value is not None and not self.suppressed:
            data["value"] = str(self.value)
        if self.lower is not None and not self.suppressed:
            data["lower"] = self.lower
            data["upper"] = self.upper
        if self.numerator_metric:
            data["numerator"] = self.numerator_metric
            data["denominator"] = self.denominator_metric
            data["denominator_cells"] = [cell.label() for cell in self.denominator_cells]
        return data

    def to_fact(self, *, status: str = "draft", delivered: str = "unknown",
                destination: str | None = None,
                definition_version: str = "default") -> Fact | None:
        if self.suppressed:
            return None
        if self.kind == "range" and self.lower is not None and self.upper is not None:
            return Fact(
                source_id=self.source_id, metric=canonical_metric(self.metric),
                value=None, cells=self.cells, kind="range", status=status,
                delivered=delivered, destination=destination,
                definition_version=definition_version,
                raw={"lower": self.lower, "upper": self.upper},
            )
        if self.value is None:
            return None
        return make_fact(
            source_id=self.source_id, metric=self.metric, value=self.value,
            cells=self.cells, kind=self.kind, status=status, delivered=delivered,
            numerator_metric=self.numerator_metric,
            denominator_metric=self.denominator_metric,
            denominator_cells=self.denominator_cells,
            destination=destination, definition_version=definition_version,
            raw={"rate_exact": "yes"} if self.kind == "rate" else {},
        )


@dataclass
class Material:
    material_id: str
    release_id: str
    period: str
    audience: str
    recipients: tuple[str, ...]
    items: tuple[LineItem, ...]
    definition_version: str
    policy_digest: str
    ledger_fingerprint: str
    visible_fingerprint: str
    version: int = 1
    status: str = DRAFT
    created_at: str = field(default_factory=_now)
    released_at: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def content_hash(self) -> str:
        return _hash({
            "material": self.material_id,
            "release": self.release_id,
            "period": self.period,
            "audience": self.audience,
            "recipients": sorted(self.recipients),
            "definition_version": self.definition_version,
            "policy_digest": self.policy_digest,
            "ledger": self.ledger_fingerprint,
            "visible": self.visible_fingerprint,
            "version": self.version,
            "items": [item.payload() for item in self.items],
        })

    @property
    def is_external(self) -> bool:
        return self.audience != INTERNAL_AUDIENCE

    def facts(self) -> list[Fact]:
        """Facts as a reader would see them, keyed by the material.

        ``source_id`` is the material id so that visibility, receipts and
        retraction are all joined at the same unit.  The line's own label is
        preserved in ``raw['line_source']``.
        """

        result = []
        for item in self.items:
            fact = item.to_fact(definition_version=self.definition_version,
                                destination=self.audience)
            if fact is None:
                continue
            raw = dict(fact.raw)
            raw["line_source"] = item.source_id
            raw["material_id"] = self.material_id
            result.append(replace(fact, source_id=self.material_id, raw=raw))
        return result

    def suppressed_cells(self) -> list[str]:
        return sorted({cell.label() for item in self.items if item.suppressed
                       for cell in item.cells})

    def manifest(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "release_id": self.release_id,
            "period": self.period,
            "audience": self.audience,
            "recipients": list(self.recipients),
            "version": self.version,
            "status": self.status,
            "content_hash": self.content_hash,
            "definition_version": self.definition_version,
            "policy_digest": self.policy_digest,
            "ledger_fingerprint": self.ledger_fingerprint,
            "visible_fingerprint_at_creation": self.visible_fingerprint,
            "items": [item.payload() for item in self.items],
            "suppressed_cells": self.suppressed_cells(),
            "notes": list(self.notes),
        }


@dataclass
class Approval:
    approval_id: str
    material_id: str
    material_hash: str
    visible_fingerprint: str
    approved_at: str
    editor: str
    status: str = APPROVED
    reasons: tuple[str, ...] = ()
    sent_material_hash: str | None = None
    withdrawn_at: str | None = None
    history: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "material_id": self.material_id,
            "material_hash": self.material_hash,
            "visible_fingerprint": self.visible_fingerprint,
            "approved_at": self.approved_at,
            "editor": self.editor,
            "status": self.status,
            "reasons": list(self.reasons),
            "sent_material_hash": self.sent_material_hash,
            "withdrawn_at": self.withdrawn_at,
            "history": list(self.history),
        }


@dataclass
class ScheduledSend:
    schedule_id: str
    material_id: str
    material_hash: str
    due_at: str
    recipients: tuple[str, ...]
    status: str = SCHEDULED
    reason: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "material_id": self.material_id,
            "material_hash": self.material_hash,
            "due_at": self.due_at,
            "recipients": list(self.recipients),
            "status": self.status,
            "reason": self.reason,
        }


def _material_from_manifest(data: Mapping[str, Any]) -> Material:
    """Reconstruct a material from an outbox manifest (values included).

    The manifest was written locally at send time.  Reading it back lets a
    later run re-derive exactly what a recipient holds, including the
    suppressed-cell list, without holding the live Workflow object.
    """

    items = []
    for raw in data.get("items", []):
        cells = []
        for label in raw.get("cells", []):
            parts = str(label).split("/")
            if len(parts) == 3:
                cells.append(CellKey(*parts))
        if not cells:
            continue
        value = raw.get("value")
        lower = raw.get("lower")
        upper = raw.get("upper")
        items.append(LineItem(
            source_id=str(raw.get("source", "")),
            metric=canonical_metric(str(raw.get("metric", ""))),
            cells=tuple(cells),
            value=Fraction(str(value)) if value not in (None, "") else None,
            kind=str(raw.get("kind", "cell")),
            suppressed=bool(raw.get("suppressed", False)),
            lower=int(lower) if lower not in (None, "") else None,
            upper=int(upper) if upper not in (None, "") else None,
        ))
    return Material(
        material_id=str(data.get("material_id", "")),
        release_id=str(data.get("release_id", "")),
        period=str(data.get("period", "")),
        audience=str(data.get("audience", "")),
        recipients=tuple(data.get("recipients", ())),
        items=tuple(items),
        definition_version=str(data.get("definition_version", "default")),
        policy_digest=str(data.get("policy_digest", "")),
        ledger_fingerprint=str(data.get("ledger_fingerprint", "")),
        visible_fingerprint=str(data.get("visible_fingerprint_at_creation", "")),
        version=int(data.get("version", 1)),
        status=str(data.get("status", "sent")),
        notes=tuple(data.get("notes", ())),
    )


class Workflow:
    """Local-only release book.

    ``outbox`` receives one JSON file per delivery attempt.  Nothing here
    imports a mail library, opens a network connection or shells out.
    """

    def __init__(self, ledger: Ledger, policy: Policy, visible: Sequence[Fact],
                 outbox: str | Path):
        self.ledger = ledger
        self.policy = policy
        # Expand wildcard scopes (month totals, cross-district totals) against
        # the ledger once, so every later interval calculation sees cells.
        self.visible = [expand_fact(fact, ledger) for fact in visible]
        self.outbox = Path(outbox)
        self.planner = Planner(ledger, policy)
        self.materials: dict[str, Material] = {}
        self.approvals: dict[str, Approval] = {}
        self.schedules: dict[str, ScheduledSend] = {}
        self.receipts: list[Receipt] = []
        self.visible_at_start = build_visible_state(self.visible, ledger).fingerprint()

    # -- construction ----------------------------------------------------

    @classmethod
    def from_files(cls, *, counts: str | Path, policy: str | Path,
                   published: str | Path | None = None,
                   receipts: str | Path | None = None,
                   outbox: str | Path) -> "Workflow":
        ledger = Ledger.load(counts)
        policy_obj = Policy.load(policy)
        facts: list[Fact] = []
        if published is not None and Path(published).exists():
            facts.extend(load_published(published,
                                        definition_version=policy_obj.definition_version))
        workflow = cls(ledger, policy_obj, facts, outbox)
        if receipts is not None and Path(receipts).exists():
            workflow.load_receipts(receipts)
        return workflow

    @classmethod
    def from_outbox(cls, *, counts: str | Path, policy: str | Path,
                    published: str | Path | None,
                    outbox: str | Path) -> "Workflow":
        """Rebuild the visible state from local delivery receipts.

        Each ``*.json`` file in the outbox was written at send time and carries
        both the receipt and the exact material manifest (content hash,
        suppressed cells, definition version).  Reading them back means the
        manifest can be checked against what the partner actually received,
        not merely against how many files this process produced.
        """

        workflow = cls.from_files(counts=counts, policy=policy,
                                  published=published, outbox=outbox)
        outbox_path = Path(outbox)
        if not outbox_path.exists():
            return workflow
        for path in sorted(outbox_path.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            receipt_data = record.get("receipt", {})
            material_data = record.get("material", {})
            receipt = Receipt(
                receipt_id=receipt_data.get("receipt_id", path.stem),
                release_id=receipt_data.get("release_id", ""),
                material_id=receipt_data.get("material_id", ""),
                version_hash=receipt_data.get("version_hash", ""),
                audience=receipt_data.get("audience", ""),
                recipient=receipt_data.get("recipient", ""),
                status=receipt_data.get("status", ""),
                delivered=receipt_data.get("delivered", "unknown"),
                sent_at=receipt_data.get("sent_at"),
                received_at=receipt_data.get("received_at"),
                proof_unsent=bool(receipt_data.get("proof_unsent", False)),
                note=receipt_data.get("note"),
            )
            if any(existing.receipt_id == receipt.receipt_id
                   for existing in workflow.receipts):
                continue
            material = _material_from_manifest(material_data)
            if material.material_id and material.material_id not in workflow.materials:
                # The outbox record is written after a send, so the material
                # it describes has been dispatched even if a crash prevented
                # the live object from being saved.
                material.status = SENT
                workflow.materials[material.material_id] = material
            workflow.receipts.append(receipt)
            if receipt.visible:
                for fact in workflow._facts_for_receipt(receipt):
                    workflow.visible.append(fact)
        return workflow

    def visible_fingerprint(self) -> str:
        return build_visible_state(self.visible, self.ledger).fingerprint()

    def visible_state(self):
        return build_visible_state(self.visible, self.ledger)

    def conflicts(self) -> tuple[str, ...]:
        from .planner import detect_conflicts

        return detect_conflicts(self.visible, self.ledger)

    def _next_id(self, prefix: str, existing: Iterable[str]) -> str:
        used = set(existing)
        index = 1
        while f"{prefix}{index}" in used:
            index += 1
        return f"{prefix}{index}"

    # -- sheet preparation ----------------------------------------------

    def prepare_period_material(self, *, period: str, metrics: Sequence[str],
                                audience: str = PUBLIC_AUDIENCE,
                                recipients: Sequence[str] = (),
                                release_id: str | None = None,
                                course_groups: Sequence[Sequence[str]] | None = None,
                                auto_merge: bool = False,
                                notes: Sequence[str] = ()) -> Material:
        """Build an internal report or an external public material.

        The internal report keeps every registered value: it is the evidence
        trail.  An external material never carries a protected value.  A cell
        below ``min_cell`` is suppressed on the public sheet unless the editor
        explicitly asks for a permitted merge (or supplies larger
        ``course_groups``), and the suppression is recorded by cell name.

        ``auto_merge`` is off by default so that the sheet matches the
        editor's chosen grouping; merge alternatives are surfaced by
        :meth:`explain_material`.
        """

        is_internal = audience == INTERNAL_AUDIENCE
        release = release_id or f"{period}-{audience}"
        course_groups = course_groups or [[course] for course in self.ledger.course_types()]
        items: list[LineItem] = []
        merged: set[CellKey] = set()
        for metric in metrics:
            metric = canonical_metric(metric)
            for district in self.ledger.districts():
                for courses in course_groups:
                    cells = tuple(sorted(
                        CellKey(period, district, course) for course in courses
                        if CellKey(period, district, course) in self.ledger.cells))
                    if not cells:
                        continue
                    if cells and all(cell in merged for cell in cells):
                        continue
                    source = f"{release}/{district}"
                    if len(cells) == 1:
                        source = f"{source}/{cells[0].course_type}"
                    if is_internal:
                        self._append_internal(items, source, cells, metric)
                        continue
                    group = self._append_external(items, source, cells, metric,
                                                  auto_merge=auto_merge)
                    if group is not None:
                        # Do not repeat a merged group once per member cell.
                        merged.update(group)
        return self.register_material(Material(
            material_id=self._next_id("m", self.materials),
            release_id=release,
            period=period,
            audience=audience,
            recipients=tuple(recipients),
            items=tuple(items),
            definition_version=self.policy.definition_version,
            policy_digest=self.policy.digest(),
            ledger_fingerprint=self.ledger.fingerprint(),
            visible_fingerprint=self.visible_fingerprint(),
            notes=tuple(notes),
        ))

    def _append_internal(self, items: list[LineItem], source: str,
                         cells: tuple[CellKey, ...], metric: str) -> None:
        values = [self.ledger.value(cell, metric) for cell in cells]
        kind = "cell" if len(cells) == 1 else "sum"
        if any(value is None for value in values):
            items.append(LineItem(source_id=source, metric=metric, cells=cells,
                                  value=None, kind=kind, suppressed=True))
            return
        total = sum(value for value in values if value is not None)
        items.append(LineItem(source_id=source, metric=metric, cells=cells,
                              value=Fraction(total), kind=kind))

    def _append_external(self, items: list[LineItem], source: str,
                         cells: tuple[CellKey, ...], metric: str, *,
                         auto_merge: bool) -> tuple[CellKey, ...] | None:
        """Append a public line; return the merged group used, if any."""
        fact, option = self.planner.aggregate_fact(
            cells, metric, source_id=source, destination=PUBLIC_AUDIENCE)
        single = len(cells) == 1
        if single:
            value = self.ledger.value(cells[0], metric)
            protected = (self.policy.is_protected_metric(metric)
                         and value is not None and value < self.policy.min_cell)
            if not protected:
                if fact is not None and fact.value is not None \
                        and fact.value >= self.policy.min_cell:
                    items.append(LineItem(source_id=source, metric=metric,
                                          cells=cells, value=Fraction(fact.value),
                                          kind="cell"))
                else:
                    items.append(LineItem(source_id=source, metric=metric,
                                          cells=cells, value=None, kind="cell",
                                          suppressed=True))
                return None
            if auto_merge:
                merge = self._first_safe_merge(cells[0], metric)
                if merge is not None:
                    group_fact, _ = self.planner.aggregate_fact(
                        merge, metric, source_id=source,
                        destination=PUBLIC_AUDIENCE)
                    if group_fact is not None and group_fact.value is not None \
                            and group_fact.value >= self.policy.min_cell:
                        items.append(LineItem(
                            source_id=source, metric=metric, cells=merge,
                            value=Fraction(group_fact.value), kind="sum"))
                        return merge
            items.append(LineItem(source_id=source, metric=metric, cells=cells,
                                  value=None, kind="cell", suppressed=True))
            return None

        if option.kind == "range_requires_roster":
            # A unique-person total over unproven rosters is neither a sum nor
            # a safe range (the range lower bound is the largest member).  Name
            # the group and suppress it rather than publish a misleading or
            # disclosive figure.
            items.append(LineItem(source_id=source, metric=metric, cells=cells,
                                  value=None, kind="sum", suppressed=True))
            return None
        if fact is not None and fact.value is not None \
                and fact.value >= self.policy.min_cell:
            items.append(LineItem(source_id=source, metric=metric, cells=cells,
                                  value=Fraction(fact.value), kind="sum"))
            return None
        items.append(LineItem(source_id=source, metric=metric, cells=cells,
                              value=None, kind="sum", suppressed=True))
        return None

    def _first_safe_merge(self, key: CellKey, metric: str) -> tuple[CellKey, ...] | None:
        options = self.planner.merge_options(key, metric, self.visible, limit=1)
        for option in options:
            if option.safe and option.kind == "exact" and option.value is not None \
                    and option.value >= self.policy.min_cell:
                return option.cells
        return None

    def _merge_is_public(self, fact: Fact) -> bool:
        return fact.value is not None and fact.value >= self.policy.min_cell

    def register_material(self, material: Material) -> Material:
        self.materials[material.material_id] = material
        return material

    # -- joint appraisal -------------------------------------------------

    def _reader_communities(self) -> dict[str, set[str]]:
        """The fixture treats every external-public material as one community."""

        communities: dict[str, set[str]] = {}
        for material in self.materials.values():
            if not material.is_external:
                continue
            key = PUBLIC_AUDIENCE if material.audience == PUBLIC_AUDIENCE else material.audience
            communities.setdefault(key, set()).update(material.recipients)
        return communities

    def overlapping_materials(self, material: Material) -> list[Material]:
        """Other non-internal materials that may reach the same reader.

        Two external materials overlap if either their audience labels or their
        recipient sets intersect.  Internal reports are excluded: they are the
        evidence trail and are not distributed outside.
        """

        if not material.is_external:
            return []
        result = []
        for other in self.materials.values():
            if other is material or not other.is_external:
                continue
            if other.audience == material.audience:
                result.append(other)
                continue
            if set(other.audience.split("+")) & set(material.audience.split("+")):
                result.append(other)
                continue
            if set(other.recipients) & set(material.recipients):
                result.append(other)
        return result

    def joint_facts(self, material: Material, *, include_scheduled: bool = True,
                    include_approved: bool = False) -> list[Fact]:
        """Material that could become visible to the same reader as ``material``.

        The starting point is what is *actually* visible (sent, uncertain,
        retracted and proven-undelivered is excluded by the model).  Material
        that is merely approved is not yet committed and is therefore not a
        visibility fact; each editor is entitled to review a sheet that is safe
        on its own.  A *scheduled* send is committed and is included, so two
        queued materials aimed at the same reader are judged jointly.  Once
        one material is actually sent, the visible log contains it, and every
        other approval is re-checked against that new state.
        """

        facts = list(self.visible)
        if include_scheduled:
            for other in self.overlapping_materials(material):
                if other.status == SCHEDULED:
                    facts.extend(other.facts())
        if include_approved:
            for other in self.overlapping_materials(material):
                if other.status == APPROVED:
                    facts.extend(other.facts())
        return facts

    def appraise_material(self, material: Material, *,
                          include_merge_options: bool = False) -> BatchVerdict:
        """Appraise an outward send.

        Internal reports are the evidence trail, not a distribution: they are
        never held to the small-cell rule and may keep every value.  External
        material is appraised against everything a reader can already see.
        """

        if not material.is_external:
            return BatchVerdict(safe=True, blocked_cells=(), reasons=(),
                                pending_cells=(), conflicts=(),
                                held_facts=(),
                                releasable_facts=tuple(material.facts()))
        facts = [fact for fact in self.joint_facts(material)
                 if fact.source_id != material.material_id]
        proposed = material.facts()
        metrics = tuple(sorted({fact.canonical_metric for fact in (*facts, *proposed)
                                if fact.canonical_metric != "attendance_rate"}))
        return self.planner.appraise(
            facts, proposed, metrics=metrics or ("participants",),
            collect_assessments=include_merge_options,
            include_merge_options=include_merge_options)

    def explain_material(self, material: Material) -> dict[str, Any]:
        """Value-free explanation of what is publishable, merge-only, or held.

        Even for an internal material with every value present, this report
        names cells and widths but never repeats a protected count.
        """

        facts = [fact for fact in self.joint_facts(material)
                 if fact.source_id != material.material_id]
        proposed = material.facts()
        metrics = tuple(sorted({fact.canonical_metric for fact in (*facts, *proposed)
                                if fact.canonical_metric != "attendance_rate"}))
        if material.is_external:
            verdict = self.planner.appraise(
                facts, proposed, metrics=metrics or ("participants",),
                collect_assessments=True, include_merge_options=True)
        else:
            # An internal report keeps values in its own right; the editor
            # explanation still shows the public-facing appraisal for the same
            # period, so it is computed from an empty external proposal and
            # names the merge alternatives available to the public sheet.
            verdict = self.planner.appraise(
                facts, [], metrics=metrics or ("participants",),
                collect_assessments=False)
        return {
            "material_id": material.material_id,
            "release_id": material.release_id,
            "period": material.period,
            "audience": material.audience,
            "definition_version": material.definition_version,
            "policy_digest": material.policy_digest,
            "safe": verdict.safe,
            "releasable": verdict.releasable,
            "blocked_cells": [cell.label() for cell in verdict.blocked_cells],
            "pending_cells": [cell.label() for cell in verdict.pending_cells],
            "conflicts": list(verdict.conflicts),
            "reasons": [reason.as_dict() for reason in verdict.reasons],
            "suppressed_cells": material.suppressed_cells(),
            "assessments": [assessment.to_public_dict()
                            for assessment in verdict.assessments],
        }

    def alternatives(self, material: Material, metric: str, *,
                     include_values: bool = False) -> list[dict]:
        if include_values and material.is_external:
            raise ValueError(
                "external materials must not include protected values in "
                "editor-facing alternative tables; use the internal report")
        facts = [fact for fact in self.joint_facts(material)
                 if fact.source_id != material.material_id]
        return self.planner.alternatives(
            metric, visible=facts, proposed=material.facts(),
            period=material.period, include_values=include_values)

    # -- approval and scheduling -----------------------------------------

    def approve(self, material: Material, *, editor: str,
                approval_id: str | None = None) -> Approval:
        verdict = self.appraise_material(material)
        if not verdict.releasable:
            raise ValueError(
                "material is not releasable: "
                + "; ".join(reason.message for reason in verdict.reasons[:3])
                or "blocked or pending cells")
        if not material.is_external and material.status == DRAFT:
            material.status = DRAFT  # internal report may be exported as evidence
        approval = Approval(
            approval_id=approval_id or self._next_id("a", self.approvals),
            material_id=material.material_id,
            material_hash=material.content_hash,
            visible_fingerprint=self.visible_fingerprint(),
            approved_at=_now(),
            editor=editor,
        )
        self.approvals[approval.approval_id] = approval
        material.status = APPROVED
        return approval

    def revalidate(self, approval: Approval, *,
                   material: Material | None = None) -> Approval:
        """Re-check an approval against the material now visible.

        Withdraws the approval if the visible set changed and the send is no
        longer safe under the current state, or if the approved material was
        edited after approval.
        """

        material = material or self.materials.get(approval.material_id)
        if material is None:
            approval.status = WITHDRAWN
            approval.reasons = ("material missing",)
            approval.withdrawn_at = _now()
            return approval
        if approval.status != APPROVED:
            return approval
        current = self.visible_fingerprint()
        reasons: list[str] = []
        if material.content_hash != approval.material_hash:
            reasons.append("material changed after approval; re-approval required")
        if current != approval.visible_fingerprint:
            verdict = self.appraise_material(material)
            if not verdict.releasable:
                reasons.append(
                    "visible material changed since approval and the send is no "
                    "longer safe")
                reasons.extend(reason.message for reason in verdict.reasons[:3])
                reasons.extend(f"conflict: {item}" for item in verdict.conflicts)
        if reasons:
            approval.status = WITHDRAWN
            approval.reasons = tuple(reasons)
            approval.withdrawn_at = _now()
            approval.history = (*approval.history, f"withdrawn {_now()}: {reasons[0]}")
        elif current != approval.visible_fingerprint:
            # The visible state moved but the send is still safe.  Re-pin the
            # approval to the new state so a later change is compared against
            # what the editor was actually able to rely on.
            approval.visible_fingerprint = current
            approval.history = (*approval.history,
                                f"revalidated {_now()}: still safe")
        else:
            approval.history = (*approval.history,
                                f"revalidated {_now()}: still safe")
        return approval

    def revalidate_all(self, *, exclude: Sequence[str] = ()) -> list[Approval]:
        withdrawn = []
        for approval in list(self.approvals.values()):
            if approval.material_id in exclude or approval.status != APPROVED:
                continue
            before = approval.status
            self.revalidate(approval)
            if approval.status == WITHDRAWN and before == APPROVED:
                material = self.materials.get(approval.material_id)
                if material is not None:
                    material.status = WITHDRAWN
                    for schedule in self.schedules.values():
                        if schedule.material_id == material.material_id \
                                and schedule.status == SCHEDULED:
                            schedule.status = WITHDRAWN
                            schedule.reason = "approval withdrawn after a sibling send"
                withdrawn.append(approval)
        return withdrawn

    def schedule(self, approval: Approval, *, due_at: str,
                 recipients: Sequence[str] | None = None) -> ScheduledSend:
        approval = self.revalidate(approval)
        if approval.status != APPROVED:
            raise ValueError("approval is not valid: " + "; ".join(approval.reasons))
        material = self.materials[approval.material_id]
        # Scheduling commits the content.  Judge it together with anything
        # already queued for the same reader; two individually safe sheets may
        # be unsafe when both are committed.
        verdict = self.appraise_material(material)
        if not verdict.releasable:
            raise ValueError(
                "material is not safe to schedule together with committed "
                "sends for the same reader: "
                + "; ".join(reason.message for reason in verdict.reasons[:3]))
        schedule = ScheduledSend(
            schedule_id=self._next_id("s", self.schedules),
            material_id=material.material_id,
            material_hash=approval.material_hash,
            due_at=due_at,
            recipients=tuple(recipients or material.recipients),
        )
        material.status = SCHEDULED
        self.schedules[schedule.schedule_id] = schedule
        return schedule

    def withdraw_schedule(self, schedule: ScheduledSend, *, reason: str) -> None:
        schedule.status = WITHDRAWN
        schedule.reason = reason
        material = self.materials.get(schedule.material_id)
        if material is not None and material.status == SCHEDULED:
            material.status = WITHDRAWN

    # -- local send ------------------------------------------------------

    def send(self, approval: Approval, *,
             recipients: Sequence[str] | None = None,
             delivered: str = "unknown",
             proof_unsent: bool = False,
             schedule: ScheduledSend | None = None) -> Receipt:
        """Write a local receipt and mark the facts visible.

        There is no network call.  ``delivered`` defaults to ``unknown`` and
        is treated as visible.  Passing ``proof_unsent=True`` is the only way
        to keep the material out of the visible set.
        """

        approval = self.revalidate(approval)
        if approval.status != APPROVED:
            raise ValueError("approval is not valid: " + "; ".join(approval.reasons))
        material = self.materials[approval.material_id]
        if material.content_hash != approval.material_hash:
            raise ValueError("material changed after approval; re-approval required")
        # Re-appraise at the exact send instant, including scheduled siblings.
        verdict = self.appraise_material(material)
        if not verdict.releasable:
            approval.status = WITHDRAWN
            approval.withdrawn_at = _now()
            approval.reasons = tuple(reason.message for reason in verdict.reasons[:3])
            material.status = WITHDRAWN
            raise ValueError("send blocked at dispatch: "
                             + "; ".join(approval.reasons))

        recipients = tuple(recipients or material.recipients)
        if material.is_external and not recipients:
            raise ValueError("an external material needs at least one recipient")
        sent_at = _now()
        outbox = self.outbox
        outbox.mkdir(parents=True, exist_ok=True)
        facts = material.facts()
        receipt_ids: list[str] = []
        for recipient in recipients or ("unaddressed",):
            receipt_id = self._next_id("r", (receipt.receipt_id for receipt in self.receipts))
            status = SENT if not proof_unsent else FAILED
            receipt = Receipt(
                receipt_id=receipt_id,
                release_id=material.release_id,
                material_id=material.material_id,
                version_hash=material.content_hash,
                audience=material.audience,
                recipient=recipient,
                status=status,
                delivered=delivered,
                sent_at=sent_at,
                received_at=None,
                proof_unsent=proof_unsent,
                note=(
                    f"payload {material.content_hash[:12]}; "
                    f"version {material.version}; suppressed "
                    f"{','.join(material.suppressed_cells()) or 'none'}"
                ),
            )
            self.receipts.append(receipt)
            receipt_ids.append(receipt_id)
            record = {
                "receipt": receipt.__dict__,
                "material": material.manifest(),
            }
            (outbox / f"{receipt_id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            if not proof_unsent and material.is_external:
                # Only material that leaves the organisation changes what a
                # reader can infer.  The internal report is evidence for the
                # audit trail and never enters the visible set.
                for fact in facts:
                    self.visible.append(replace(
                        fact, status=SENT, delivered=delivered, visible=True,
                        seen_risk=True, destination=material.audience))
        if not proof_unsent:
            material.status = SENT
            material.released_at = sent_at
            approval.status = SENT
            approval.sent_material_hash = material.content_hash
            if schedule is not None:
                schedule.status = SENT
            # Any other approved material may now be unsafe: withdraw it.
            self.revalidate_all(exclude=(material.material_id,))
        return self.receipts[-1]

    def resend(self, material: Material, *, editor: str,
               reason: str = "resend") -> tuple[Material, Approval]:
        """Create a new version of an already-sent material.

        The old version remains visible (a resend does not unsend it).  The
        new version is approved only against the post-send visible state.
        """

        clone = Material(
            material_id=self._next_id("m", self.materials),
            release_id=material.release_id,
            period=material.period,
            audience=material.audience,
            recipients=material.recipients,
            items=material.items,
            definition_version=material.definition_version,
            policy_digest=material.policy_digest,
            ledger_fingerprint=self.ledger.fingerprint(),
            visible_fingerprint=self.visible_fingerprint(),
            version=material.version + 1,
            notes=(*material.notes, reason),
        )
        self.register_material(clone)
        approval = self.approve(clone, editor=editor)
        return clone, approval

    def revise(self, approval: Approval, *, material: Material, editor: str) -> Approval:
        """Replace the approved aggregation with an edited material.

        The old approval is withdrawn and a fresh one is recorded.  This is
        the editor path for changing the merge scope after a sibling send made
        the original unsafe.
        """

        self.revalidate(approval)
        if approval.status == APPROVED:
            approval.status = WITHDRAWN
            approval.withdrawn_at = _now()
            approval.reasons = ("replaced by revised material",)
            approval.history = (*approval.history, f"revised {_now()}")
        material.visible_fingerprint = self.visible_fingerprint()
        material.policy_digest = self.policy.digest()
        material.ledger_fingerprint = self.ledger.fingerprint()
        self.register_material(material)
        return self.approve(material, editor=editor)

    # -- retraction and uncertain delivery -------------------------------

    def retract(self, material: Material, *, reason: str) -> list[Fact]:
        """Record a retraction without pretending readers did not see it.

        Retraction facts are marked ``retracted``, which stays visible in the
        model.  The original facts are not removed.  Retracting material that
        was never sent records the decision but creates no reader-visible
        fact, because no reader could have seen it.
        """

        was_sent = material.status in {SENT, RETRACTED} or any(
            receipt.visible and receipt.material_id == material.material_id
            for receipt in self.receipts)
        material.status = RETRACTED
        material.notes = (*material.notes, f"retracted: {reason}")
        if not was_sent or not material.is_external:
            return []
        retraction: list[Fact] = []
        for item in material.items:
            if item.suppressed:
                continue
            if item.kind == "range":
                if item.lower is None or item.upper is None:
                    continue
                retraction.append(Fact(
                    source_id=material.material_id, metric=canonical_metric(item.metric),
                    value=None, cells=item.cells, kind="range", status=RETRACTED,
                    delivered="yes", destination=material.audience,
                    definition_version=material.definition_version,
                    raw={"lower": item.lower, "upper": item.upper,
                         "evidence": f"retracted: {reason}"},
                ))
                continue
            if item.value is None:
                continue
            retraction.append(make_fact(
                source_id=material.material_id, metric=item.metric,
                value=item.value, cells=item.cells, kind=item.kind,
                status=RETRACTED, delivered="yes",
                definition_version=material.definition_version,
                destination=material.audience,
                evidence=f"retracted: {reason}",
            ))
        self.visible.extend(retraction)
        return retraction

    def prove_undelivered(self, receipt_id: str, *, evidence: str) -> Receipt:
        """Only an explicit proof can remove an uncertain delivery.

        A material with several recipients stays visible while any recipient
        may still have received it.  Facts are withdrawn only when every
        receipt for that material is proven undelivered.
        """

        for index, receipt in enumerate(self.receipts):
            if receipt.receipt_id != receipt_id:
                continue
            updated = replace(receipt, status=FAILED, delivered="no",
                              proof_unsent=True,
                              note=(receipt.note or "") + f"; proof: {evidence}")
            self.receipts[index] = updated
            still_possible = any(
                other.material_id == receipt.material_id and other.visible
                for other in self.receipts)
            if not still_possible:
                self.visible = [
                    fact for fact in self.visible
                    if not (fact.source_id == receipt.material_id
                            and fact.destination == receipt.audience)
                ]
            return updated
        raise KeyError(receipt_id)

    # -- receipts and manifest ------------------------------------------

    def load_receipts(self, path: str | Path) -> list[Receipt]:
        path = Path(path)
        loaded: list[Receipt] = []
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                proof = str(row.get("proof_unsent") or "").strip().lower() in {"1", "yes", "true"}
                receipt = Receipt(
                    receipt_id=(row.get("receipt_id") or "").strip(),
                    release_id=(row.get("release_id") or "").strip(),
                    material_id=(row.get("material_id") or "").strip(),
                    version_hash=(row.get("version_hash") or "").strip(),
                    audience=(row.get("audience") or "").strip(),
                    recipient=(row.get("recipient") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    delivered=(row.get("delivered") or "unknown").strip(),
                    sent_at=(row.get("sent_at") or "").strip() or None,
                    received_at=(row.get("received_at") or "").strip() or None,
                    proof_unsent=proof,
                    note=(row.get("note") or "").strip() or None,
                )
                loaded.append(receipt)
                if receipt.visible:
                    for fact in self._facts_for_receipt(receipt):
                        self.visible.append(fact)
        self.receipts.extend(loaded)
        return loaded

    def _facts_for_receipt(self, receipt: Receipt) -> list[Fact]:
        material = self.materials.get(receipt.material_id)
        if material is None:
            return []
        return [replace(fact, status="sent", delivered=receipt.delivered,
                        visible=True, seen_risk=True,
                        destination=receipt.audience)
                for fact in material.facts()]

    def delivery_manifest(self) -> dict[str, Any]:
        """Join approvals, schedules, materials and receipts.

        The point is to show which content hash a recipient actually received
        and which version was only approved, scheduled, withdrawn or never
        proven delivered.
        """

        by_material: dict[str, list[dict[str, Any]]] = {}
        for receipt in self.receipts:
            by_material.setdefault(receipt.material_id, []).append({
                "receipt_id": receipt.receipt_id,
                "recipient": receipt.recipient,
                "status": receipt.status,
                "delivered": receipt.delivered,
                "visible": receipt.visible,
                "content_hash": receipt.version_hash,
                "audience": receipt.audience,
                "note": receipt.note,
            })
        rows: list[dict[str, Any]] = []
        for material in self.materials.values():
            approvals = [approval.manifest() for approval in self.approvals.values()
                         if approval.material_id == material.material_id]
            schedules = [schedule.manifest() for schedule in self.schedules.values()
                         if schedule.material_id == material.material_id]
            rows.append({
                "material_id": material.material_id,
                "release_id": material.release_id,
                "period": material.period,
                "audience": material.audience,
                "version": material.version,
                "status": material.status,
                "content_hash": material.content_hash,
                "definition_version": material.definition_version,
                "policy_digest": material.policy_digest,
                "ledger_fingerprint": material.ledger_fingerprint,
                "items": [item.payload() for item in material.items],
                "suppressed_cells": material.suppressed_cells(),
                "approvals": approvals,
                "schedules": schedules,
                "receipts": by_material.get(material.material_id, []),
            })
        return {
            "visible_fingerprint": self.visible_fingerprint(),
            "conflicts": list(self.conflicts()),
            "materials": rows,
        }

    def export_manifest(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.delivery_manifest(), ensure_ascii=False,
                                   indent=2) + "\n", encoding="utf-8")
        return path
