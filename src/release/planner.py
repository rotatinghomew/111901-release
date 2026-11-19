"""Release appraisal: what may be published, merged, or only kept internal.

The planner never decides safety from the ledger value alone.  It rebuilds
what a reader of all *visible* material can infer about each cell, then asks
what one more release would add.  Editor-facing statuses:

``publishable``
    The cell may appear on the public sheet (its count is at or above
    ``min_cell`` and nothing unverified is involved).
``merge_only``
    The cell is protected, but a grouping allowed by ``merge_scope`` gives a
    safe public total.
``internal``
    Protected or unaffected by this batch; keep it in the internal report.
``pending_verification``
    A value is missing (an empty field is not zero) or visible records
    conflict with the current revision, so no release can be proven safe.
``already_exposed``
    The cell was already narrowable below the interval floor before this
    release.  Retracting material does not undo what a reader may have seen.
``blocked``
    The proposed material would push a still-protected cell below the floor.

Explanations are value-free: cells, reason codes and interval widths only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .bounds import BoundsReport, BoundsSolver, width
from .model import (
    CellKey,
    COUNT_METRICS,
    Fact,
    Ledger,
    MetricDefinition,
    canonical_metric,
    make_fact,
)
from .policy import Policy


PUBLISHABLE = "publishable"
MERGE_ONLY = "merge_only"
MERGE_REQUIRES_ROSTER = "merge_requires_roster"
INTERNAL = "internal"
PENDING_VERIFICATION = "pending_verification"
ALREADY_EXPOSED = "already_exposed"
BLOCKED = "blocked"
NOT_PROTECTED = "not_protected"


@dataclass(frozen=True)
class Reason:
    """A value-free explanation.

    ``width_before``/``width_after`` are interval widths (public
    information), never counts.
    """

    code: str
    message: str
    cells: tuple[str, ...] = ()
    width_before: int | None = None
    width_after: int | None = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "cells": list(self.cells),
            "width_before": self.width_before,
            "width_after": self.width_after,
        }


@dataclass
class MergeOption:
    cells: tuple[CellKey, ...]
    metric: str
    kind: str                    # exact | range_requires_roster
    value: int | None = None
    lower: int | None = None
    upper: int | None = None
    safe: bool = False
    status: str = PENDING_VERIFICATION
    reasons: tuple[Reason, ...] = ()
    per_cell_post_width: Mapping[str, int] = field(default_factory=dict)
    redact_below: int | None = None

    @property
    def size(self) -> int:
        return len(self.cells)

    def label(self) -> str:
        return "+".join(cell.label() for cell in self.cells)

    def _echoable(self, number: int | None) -> bool:
        """Whether a number may appear in a value-free explanation.

        A group total at or above the minimum cell is itself publishable.  A
        number below it must not be repeated here: an exact small total, or a
        range whose lower bound equals a protected member's value, would
        re-disclose the very cell the explanation is protecting.
        """

        if number is None:
            return False
        if self.redact_below is None:
            return True
        return number >= self.redact_below

    def to_public_dict(self) -> dict:
        data = {
            "cells": [cell.label() for cell in self.cells],
            "metric": self.metric,
            "kind": self.kind,
            "safe": self.safe,
            "status": self.status,
            "reasons": [reason.as_dict() for reason in self.reasons],
            "per_cell_post_width": dict(self.per_cell_post_width),
        }
        if self.value is not None and self._echoable(self.value):
            data["value"] = self.value
        if self.lower is not None and self._echoable(self.lower) \
                and self._echoable(self.upper):
            data["lower"] = self.lower
            data["upper"] = self.upper
        return data


@dataclass
class CellAssessment:
    cell: CellKey
    metric: str
    status: str
    is_protected: bool
    pre_width: int | None
    post_width: int | None
    already_exposed: bool
    affected: bool
    reasons: tuple[Reason, ...]
    merge_options: tuple[MergeOption, ...] = ()

    def to_public_dict(self) -> dict:
        return {
            "cell": self.cell.label(),
            "metric": self.metric,
            "status": self.status,
            "protected": self.is_protected,
            "pre_width": self.pre_width,
            "post_width": self.post_width,
            "already_exposed": self.already_exposed,
            "affected": self.affected,
            "reasons": [reason.as_dict() for reason in self.reasons],
            "merge_options": [option.to_public_dict() for option in self.merge_options],
        }


@dataclass
class BatchVerdict:
    safe: bool
    blocked_cells: tuple[CellKey, ...]
    reasons: tuple[Reason, ...]
    pending_cells: tuple[CellKey, ...]
    conflicts: tuple[str, ...]
    held_facts: tuple[Fact, ...] = ()
    releasable_facts: tuple[Fact, ...] = ()
    assessments: tuple[CellAssessment, ...] = ()

    @property
    def releasable(self) -> bool:
        return self.safe and not self.held_facts and not self.conflicts

    def to_public_dict(self) -> dict:
        return {
            "safe": self.safe,
            "releasable": self.releasable,
            "blocked_cells": [cell.label() for cell in self.blocked_cells],
            "pending_cells": [cell.label() for cell in self.pending_cells],
            "held_facts": [
                {"source": fact.source_id, "metric": fact.canonical_metric,
                 "cells": [cell.label() for cell in fact.cells]}
                for fact in self.held_facts
            ],
            "reasons": [reason.as_dict() for reason in self.reasons],
            "conflicts": list(self.conflicts),
        }


def metric_values_missing(ledger: Ledger, cells: Iterable[CellKey], metric: str) -> list[CellKey]:
    metric = canonical_metric(metric)
    return [cell for cell in cells if ledger.value(cell, metric) is None]


def _fact_conflict_with_ledger(fact: Fact, ledger: Ledger) -> bool:
    if fact.value is None or fact.kind != "cell" or len(fact.cells) != 1:
        return False
    cell = fact.cells[0]
    if cell not in ledger.cells:
        return False
    true_value = ledger.value(cell, fact.canonical_metric)
    if true_value is None:
        return False
    return Fraction(true_value) != fact.value


def _conflict_keys(visible: Sequence[Fact], ledger: Ledger) -> set[tuple[CellKey, str]]:
    result: set[tuple[CellKey, str]] = set()
    by_cell: dict[tuple[CellKey, str], set[Fraction]] = {}
    for fact in visible:
        if not fact.visible:
            continue
        if fact.kind == "cell" and len(fact.cells) == 1 and _fact_conflict_with_ledger(fact, ledger):
            result.add((fact.cells[0], fact.canonical_metric))
        if fact.kind == "cell" and len(fact.cells) == 1 and fact.value is not None:
            by_cell.setdefault((fact.cells[0], fact.canonical_metric), set()).add(fact.value)
    for key, values in by_cell.items():
        if len(values) > 1:
            result.add(key)
    return result


def conflict_messages(keys: Iterable[tuple[CellKey, str]]) -> tuple[str, ...]:
    """Reader-facing conflict messages for a set of (cell, metric) keys."""

    messages = []
    for key, metric in sorted(set(keys)):
        messages.append(
            f"visible record for {key.label()} {metric} does not match the current "
            "ledger revision or another visible record; the earlier value may "
            "already be known to a reader"
        )
    return tuple(messages)


def detect_conflicts(visible: Sequence[Fact], ledger: Ledger) -> tuple[str, ...]:
    problems: list[str] = []
    statements: dict[tuple[CellKey, str], Fact] = {}
    for fact in visible:
        if not fact.visible or fact.kind != "cell" or len(fact.cells) != 1 or fact.value is None:
            continue
        key = (fact.cells[0], fact.canonical_metric)
        previous = statements.get(key)
        if previous is not None and previous.value != fact.value:
            problems.append(
                f"visible records disagree for {key[0].label()} {key[1]} "
                f"({previous.source_id} vs {fact.source_id})"
            )
        statements[key] = fact
    for key, _metric in sorted(_conflict_keys(visible, ledger)):
        problems.append(
            f"visible record for {key.label()} does not match the current ledger "
            "revision or another visible record; the earlier value may already be "
            "known to a reader")
    return tuple(dict.fromkeys(problems))


class Planner:
    def __init__(self, ledger: Ledger, policy: Policy):
        self.ledger = ledger
        self.policy = policy

    # -- interval helpers -------------------------------------------------

    def bounds(self, visible: Sequence[Fact], metrics: Iterable[str]) -> BoundsReport:
        """Bounds from facts the caller has already established as visible.

        The ``visible`` argument is a statement about visibility, so every
        fact in it is counted even if its recorded status is stale.  Use
        :class:`~release.bounds.BoundsSolver` directly to filter by status.
        """

        metrics = tuple(dict.fromkeys(canonical_metric(metric) for metric in metrics))
        solver = BoundsSolver(self.ledger, self.policy, visible, include_pending=True)
        return solver.propagate(metrics)

    def bounds_including(self, visible: Sequence[Fact], proposed: Sequence[Fact],
                         metrics: Iterable[str]) -> BoundsReport:
        metrics = tuple(dict.fromkeys(canonical_metric(metric) for metric in metrics))
        solver = BoundsSolver.including_proposed(self.ledger, self.policy, visible, proposed,
                                                 max_iterations=200)
        # ``including_proposed`` filters the visible list by the fact's own
        # status; the caller's list is authoritative here, so add any fact it
        # supplied but which is not marked visible.
        combined = list(solver.visible)
        for fact in visible:
            if all(existing is not fact and existing != fact for existing in combined):
                combined.append(fact)
        solver.visible = tuple(combined)
        return solver.propagate(metrics)

    def _metrics_in(self, facts: Iterable[Fact]) -> tuple[str, ...]:
        found: set[str] = set()
        for fact in facts:
            if fact.canonical_metric in COUNT_METRICS:
                found.add(fact.canonical_metric)
            if fact.is_rate:
                if fact.numerator_metric:
                    found.add(fact.numerator_metric)
                if fact.denominator_metric:
                    found.add(fact.denominator_metric)
        return tuple(sorted(found))

    def is_protected_value(self, key: CellKey, metric: str) -> bool:
        if not self.policy.is_protected_metric(metric):
            return False
        value = self.ledger.value(key, metric)
        return value is not None and value < self.policy.min_cell

    # -- aggregate construction ------------------------------------------

    def aggregate_fact(self, cells: Sequence[CellKey], metric: str, *, source_id: str,
                       status: str = "draft", delivered: str = "unknown",
                       destination: str | None = None,
                       definition_version: str | None = None,
                       audience: str = "public") -> tuple[Fact | None, MergeOption]:
        """Build an aggregate or explain why it cannot be published.

        A unique-person total is exact only under a definition or roster proof
        of additivity.  Otherwise the returned option carries the safe
        interval (``lower``/``upper``) and instructs the editor to ask for a
        roster proof or switch to a visit metric.
        """

        metric = canonical_metric(metric)
        cells = tuple(sorted(cells))
        if not cells:
            return None, MergeOption(cells=(), metric=metric, kind="exact",
                                     status=PENDING_VERIFICATION,
                                     reasons=(Reason("empty_group", "合并范围为空"),))
        missing = metric_values_missing(self.ledger, cells, metric)
        if missing:
            return None, MergeOption(
                cells=cells, metric=metric, kind="exact",
                status=PENDING_VERIFICATION,
                reasons=(Reason(
                    "missing_value",
                    "该组有成员没有登记值；空值不是零，无法计算合并数",
                    tuple(cell.label() for cell in missing)),),
            )
        aggregate = self.ledger.aggregate(cells, metric)
        definition = MetricDefinition.for_metric(metric)
        relation = aggregate.roster_relation
        additive = definition.additive or relation in {"disjoint", "same"}
        if not additive and len(cells) > 1:
            # Unique-person total over rosters not proven disjoint.  Neither
            # the sum nor a range may be published: the sum assumes disjoint
            # rosters, and the range's lower bound equals the largest member,
            # which would disclose that member (often a protected cell).  The
            # editor must supply a roster proof, merge differently, or use a
            # visit metric.
            return None, MergeOption(
                cells=cells, metric=metric, kind="range_requires_roster",
                lower=aggregate.lower, upper=aggregate.upper,
                status=MERGE_REQUIRES_ROSTER,
                reasons=(Reason(
                    "overlapping_roster",
                    "成员名单未证明互不重叠，参与人数不能按分项相加，也不宜用区间代替"
                    "（区间下界等于最大成员值）；请补名单证明、改合并范围或改用人次口径",
                    tuple(cell.label() for cell in cells)),),
            )
        fact = make_fact(
            source_id=source_id, metric=metric, value=aggregate.exact,
            cells=cells,
            kind="cell" if len(cells) == 1 else "sum",
            status=status, delivered=delivered,
            destination=destination,
            definition_version=definition_version or self.ledger.definition_version,
            raw={"audience": audience},
        )
        return fact, MergeOption(cells=cells, metric=metric, kind="exact",
                                 value=aggregate.exact, status=PUBLISHABLE)

    # -- relevance -------------------------------------------------------

    def relevant_cells(self, proposed: Sequence[Fact]) -> set[CellKey]:
        """Cells a batch could affect by direct or differencing inference.

        Relevance is the union of the batch's own cells and every cell in the
        same periods (month totals are the main differencing lever), plus all
        cells named by any aggregate in the batch.  Cells outside these
        closures cannot be narrowed by the batch and must not block it.
        """

        if not proposed:
            return set(self.ledger.keys())
        proposed_cells: set[CellKey] = set()
        for fact in proposed:
            proposed_cells.update(fact.cells)
        relevant = set(proposed_cells)
        periods = {cell.period for cell in proposed_cells}
        for key in self.ledger.keys():
            if key.period in periods:
                relevant.add(key)
        for fact in proposed:
            if len(fact.cells) > 1:
                relevant.update(fact.cells)
        return relevant

    # -- appraisal -------------------------------------------------------

    def appraise(self, visible: Sequence[Fact], proposed: Sequence[Fact], *,
                 metrics: Iterable[str] | None = None,
                 collect_assessments: bool = False,
                 include_merge_options: bool = False) -> BatchVerdict:
        proposed = tuple(proposed)
        metrics = tuple(metrics) if metrics is not None else self._metrics_in((*visible, *proposed))
        if not metrics:
            metrics = self._metrics_in(proposed) or ("participants",)
        if not proposed:
            # Nothing would be released, so the sheet cannot narrow any
            # interval and is safe to file even if unrelated visible records
            # conflict.  Those conflicts remain available from
            # ``detect_conflicts`` and the workflow manifest.
            return BatchVerdict(safe=True, blocked_cells=(), reasons=(),
                                pending_cells=(), conflicts=(), held_facts=(),
                                releasable_facts=())
        pre = self.bounds(visible, metrics)
        post = self.bounds_including(visible, proposed, metrics)
        conflict_keys = _conflict_keys((*visible, *proposed), self.ledger)
        relevant = self.relevant_cells(proposed)
        # A conflict blocks only the cells it can actually affect.  Conflicts
        # outside this batch's periods are still reported by the workflow
        # manifest, but they cannot change what a 2026-03 sheet reveals.
        affected_conflicts = {key for key in conflict_keys if key[0] in relevant}
        conflicts = conflict_messages(affected_conflicts)

        blocked: list[CellKey] = []
        pending: list[CellKey] = []
        reasons: list[Reason] = []
        assessments: list[CellAssessment] = []

        proposed_exact = {
            (fact.cells[0], fact.canonical_metric)
            for fact in proposed
            if fact.kind == "cell" and len(fact.cells) == 1
        }
        proposed_small: set[tuple[CellKey, str]] = set()
        for fact in proposed:
            if fact.kind == "cell" and len(fact.cells) == 1 and fact.value is not None \
                    and self.policy.is_protected_metric(fact.canonical_metric) \
                    and fact.value < self.policy.min_cell:
                key = (fact.cells[0], fact.canonical_metric)
                proposed_small.add(key)
                blocked.append(fact.cells[0])
                post_w = post.width(*key) if key in post.bounds else None
                reasons.append(Reason(
                    "protected_cell_direct",
                    "该组人数低于最小保护单元，不能单独公开；请改用允许范围内的合并口径",
                    (fact.cells[0].label(),), post_w, post_w))

        for metric in metrics:
            for key in self.ledger.keys():
                true_value = self.ledger.value(key, metric)
                pre_width = width(pre.bounds[(key, metric)]) if (key, metric) in pre.bounds else None
                post_width = width(post.bounds[(key, metric)]) if (key, metric) in post.bounds else None
                protected = self.policy.is_protected_metric(metric)
                tiny = (
                    true_value is not None and true_value < self.policy.min_cell
                )
                is_protected = protected and tiny
                already_exposed = (
                    is_protected and pre_width is not None
                    and pre_width < self.policy.max_interval_width
                )
                affects = key in relevant
                cell_reasons: list[Reason] = []

                if true_value is None:
                    status = PENDING_VERIFICATION
                    if affects:
                        pending.append(key)
                    cell_reasons.append(Reason(
                        "missing_value", "该格没有登记值；空值不是零", (key.label(),)))
                elif (key, metric) in conflict_keys:
                    status = PENDING_VERIFICATION
                    if affects:
                        pending.append(key)
                    cell_reasons.append(Reason(
                        "visible_conflict",
                        "已有的对外记录与当前台账修订不一致，需先核实哪一版已真正送达",
                        (key.label(),)))
                elif not is_protected:
                    # A count at or above ``min_cell``, or a small count in a
                    # metric outside ``protected_metrics`` (a public
                    # denominator such as eligible), is not a small cell.  It
                    # is not a disclosure by itself even when interval
                    # reasoning pins it down as a component of a protected
                    # count.
                    status = PUBLISHABLE if (key, metric) in proposed_exact else NOT_PROTECTED
                    cell_reasons.append(Reason(
                        "not_protected",
                        "该格不适用最小保护单元，可按单元发布",
                        (key.label(),), pre_width, post_width))
                elif not affects:
                    status = INTERNAL
                    cell_reasons.append(Reason(
                        "not_affected",
                        "本次拟发布内容不涉及该保护组，保留在内部报告",
                        (key.label(),), pre_width, post_width))
                elif already_exposed:
                    status = ALREADY_EXPOSED
                    cell_reasons.append(Reason(
                        "already_exposed",
                        "该组在本次发布前已可被推断到保护下限以内；撤回不能让读者忘记",
                        (key.label(),), pre_width, post_width))
                elif (key, metric) in proposed_small:
                    status = BLOCKED
                    # The explicit reason is already recorded above.
                elif post_width is not None and post_width < self.policy.max_interval_width:
                    status = BLOCKED
                    blocked.append(key)
                    narrow = Reason(
                        "interval_too_narrow",
                        "本次拟发布内容会把该保护组的人数范围缩小到政策下限以内；"
                        "应合并发布、换口径或撤下相应发布项",
                        (key.label(),), pre_width, post_width)
                    cell_reasons.append(narrow)
                    reasons.append(narrow)
                else:
                    status = INTERNAL
                    cell_reasons.append(Reason(
                        "still_protected",
                        "保留在内部报告即可；公开发布不会把范围缩到保护下限以内",
                        (key.label(),), pre_width, post_width))

                if status == PENDING_VERIFICATION and (key, metric) not in post.bounds:
                    pending.append(key)
                if collect_assessments:
                    options: tuple[MergeOption, ...] = ()
                    if include_merge_options and is_protected and true_value is not None:
                        options = tuple(self.merge_options(key, metric, visible, proposed))
                    assessments.append(CellAssessment(
                        cell=key, metric=metric, status=status,
                        is_protected=is_protected,
                        pre_width=pre_width, post_width=post_width,
                        already_exposed=already_exposed, affected=affects,
                        reasons=tuple(cell_reasons), merge_options=options))

        for note in (*pre.notes, *post.notes):
            reasons.append(Reason("constraint_note", note))
        if not post.complete:
            reasons.append(Reason(
                "incomplete_proof",
                "现有资料不足以证明可以放行；相关单元只能待核核实"))

        held: list[Fact] = []
        releasable: list[Fact] = []
        pending_set = set(pending)
        for fact in proposed:
            covers_pending = bool(set(fact.cells) & pending_set)
            if fact.canonical_metric in COUNT_METRICS:
                if metric_values_missing(self.ledger, fact.cells, fact.canonical_metric):
                    covers_pending = True
            if covers_pending:
                held.append(fact)
            else:
                releasable.append(fact)

        held_cells = {cell for fact in held for cell in fact.cells}
        blocked_unique = tuple(dict.fromkeys(blocked))
        pending_unique = tuple(dict.fromkeys(pending))
        safe = (
            not blocked_unique
            and not pending_unique
            and not held_cells
            and post.complete
            and not conflicts
        )
        if not post.complete:
            pending_unique = tuple(dict.fromkeys((*pending_unique, *blocked_unique)))
        return BatchVerdict(
            safe=safe,
            blocked_cells=blocked_unique,
            reasons=tuple(reasons),
            pending_cells=pending_unique,
            conflicts=conflicts,
            held_facts=tuple(held),
            releasable_facts=tuple(releasable),
            assessments=tuple(assessments),
        )

    # -- merge alternatives ----------------------------------------------

    def merge_groups(self, target: CellKey, metric: str) -> list[tuple[CellKey, ...]]:
        fixed = self.policy.fixed_merge_dimensions
        if fixed == ("*",):
            peers = [key for key in self.ledger.keys() if key != target]
        else:
            fixed_set = set(fixed)
            if {"period", "district"} <= fixed_set:
                peers = [key for key in self.ledger.keys()
                         if key.period == target.period and key.district == target.district
                         and key != target]
            elif "period" in fixed_set:
                peers = [key for key in self.ledger.keys()
                         if key.period == target.period and key != target]
            elif "district" in fixed_set:
                peers = [key for key in self.ledger.keys()
                         if key.district == target.district and key != target]
            else:
                peers = [key for key in self.ledger.keys() if key != target]
        groups: list[tuple[CellKey, ...]] = []
        peers = sorted(peers)
        for size in range(0, min(len(peers), 8) + 1):
            for combination in combinations(peers, size):
                groups.append(tuple(sorted((target, *combination))))
        groups.sort(key=lambda group: (len(group), [cell.label() for cell in group]))
        return groups

    def merge_options(self, target: CellKey, metric: str, visible: Sequence[Fact],
                      proposed: Sequence[Fact] = (), *, limit: int = 6) -> list[MergeOption]:
        metric = canonical_metric(metric)
        base_proposed = list(proposed)
        options: list[MergeOption] = []
        explored = 0
        for group in self.merge_groups(target, metric):
            if explored >= limit * 8:
                break
            explored += 1
            fact, option = self.aggregate_fact(group, metric, source_id="merge-option")
            option.redact_below = self.policy.min_cell
            candidate = base_proposed + ([fact] if fact is not None else [])
            post = self.bounds_including(visible, candidate, (metric,))
            target_width = width(post.cell(target, metric)) if (target, metric) in post.bounds else None
            option.per_cell_post_width = {
                cell.label(): width(post.cell(cell, metric))
                for cell in group if (cell, metric) in post.bounds
            }
            if option.kind == "exact" and option.value is not None \
                    and option.value < self.policy.min_cell:
                option.status = INTERNAL
                option.reasons = (Reason(
                    "merged_total_too_small",
                    "合并后的总数仍低于最小保护单元，不能作为公开数字",
                    tuple(cell.label() for cell in group)),)
                options.append(option)
                continue
            verdict = self.appraise(visible, candidate, metrics=(metric,))
            if verdict.blocked_cells or (target_width is not None
                                         and target_width < self.policy.max_interval_width):
                option.safe = False
                option.status = BLOCKED
                option.reasons = option.reasons + (Reason(
                    "merge_still_discloses",
                    "该合并口径仍会使保护组可被确定，不能放行",
                    (target.label(),), None, target_width),)
                options.append(option)
                continue
            if verdict.pending_cells:
                option.safe = False
                option.status = PENDING_VERIFICATION
                option.reasons = option.reasons + (Reason(
                    "merge_unverified",
                    "合并范围内有资料缺失或记录冲突，需先核实",
                    tuple(cell.label() for cell in verdict.pending_cells)),)
                options.append(option)
                continue
            option.safe = True
            option.status = MERGE_REQUIRES_ROSTER if option.kind == "range_requires_roster" else MERGE_ONLY
            options.append(option)
            if sum(1 for item in options if item.safe) >= limit:
                break
        return options

    # -- rates -----------------------------------------------------------

    def rate(self, cells: Sequence[CellKey], *, numerator: str | None = None,
             denominator: str | None = None,
             denominator_cells: Sequence[CellKey] | None = None,
             source_id: str = "rate", status: str = "draft",
             delivered: str = "unknown", destination: str | None = None,
             rate_exact: bool = True) -> Fact | None:
        numerator = canonical_metric(numerator or self.policy.rate_numerator)
        denominator = canonical_metric(denominator or self.policy.rate_denominator)
        cells = tuple(sorted(cells))
        denominator_cells = tuple(sorted(denominator_cells or cells))
        if metric_values_missing(self.ledger, cells, numerator):
            return None
        if metric_values_missing(self.ledger, denominator_cells, denominator):
            return None
        denominator_total = sum(self.ledger.value(cell, denominator) or 0
                                for cell in denominator_cells)
        if denominator_total == 0:
            return None
        numerator_total = sum(self.ledger.value(cell, numerator) or 0 for cell in cells)
        fact = make_fact(
            source_id=source_id, metric="attendance_rate",
            value=Fraction(numerator_total, denominator_total),
            cells=cells, kind="rate", status=status, delivered=delivered,
            numerator_metric=numerator, denominator_metric=denominator,
            denominator_cells=denominator_cells, destination=destination,
            definition_version=self.ledger.definition_version,
        )
        raw = dict(fact.raw)
        raw["rate_exact"] = "yes" if rate_exact else "no"
        raw["rate_tolerance"] = str(self.policy.rate_tolerance)
        return Fact(**{**fact.__dict__, "raw": raw})

    def rate_explanation(self, cells: Sequence[CellKey], *, numerator: str | None = None,
                         denominator: str | None = None) -> Reason:
        numerator = canonical_metric(numerator or self.policy.rate_numerator)
        denominator = canonical_metric(denominator or self.policy.rate_denominator)
        missing_num = metric_values_missing(self.ledger, cells, numerator)
        missing_den = metric_values_missing(self.ledger, cells, denominator)
        missing = sorted({cell.label() for cell in (*missing_num, *missing_den)})
        if missing:
            return Reason(
                "rate_missing_denominator",
                "到场率的一格或多格分母没有登记值；空分母不等于零，不能计算，也不能据此放行",
                tuple(missing))
        denominator_total = sum(self.ledger.value(cell, denominator) or 0 for cell in cells)
        if denominator_total == 0:
            return Reason("rate_zero_denominator", "分母为零，到场率未定义")
        return Reason("rate_ok", "分母完整，可以计算到场率并纳入联合判定")

    # -- editor explanation / alternative tables -------------------------

    def alternatives(self, metric: str, *, visible: Sequence[Fact],
                     proposed: Sequence[Fact] = (), period: str | None = None,
                     include_values: bool = False) -> list[dict]:
        """Candidate aggregation tables that never echo protected values.

        Values are only included when ``include_values`` is true (the internal
        report path).  The default public-facing explanation contains group
        membership, status and interval widths.
        """

        metric = canonical_metric(metric)
        groups: list[tuple[CellKey, ...]] = []
        fixed = set(self.policy.fixed_merge_dimensions)
        if fixed == {"*"}:
            groups.append(self.ledger.keys())
        elif {"period", "district"} <= fixed:
            for period_key in self.ledger.periods():
                for district in self.ledger.districts():
                    group = self.ledger.where(period=period_key, district=district)
                    if group:
                        groups.append(group)
        elif "period" in fixed:
            for period_key in self.ledger.periods():
                for course in self.ledger.course_types():
                    group = self.ledger.where(period=period_key, course_type=course)
                    if group:
                        groups.append(group)
        elif "district" in fixed:
            for district in self.ledger.districts():
                for course in self.ledger.course_types():
                    group = self.ledger.where(district=district, course_type=course)
                    if group:
                        groups.append(group)
        if period:
            groups = [group for group in groups
                      if all(cell.period == period for cell in group)]
        results: list[dict] = []
        for group in groups:
            fact, option = self.aggregate_fact(group, metric, source_id="alternative")
            entry: dict = {
                "cells": [cell.label() for cell in group],
                "metric": metric,
            }
            if option.kind == "exact" and option.value is not None \
                    and option.value < self.policy.min_cell:
                entry.update(status=INTERNAL,
                             reason="合并后的总数仍低于最小保护单元，不能作为公开数字")
                results.append(entry)
                continue
            candidate = list(proposed) + ([fact] if fact is not None else [])
            verdict = self.appraise(visible, candidate, metrics=(metric,))
            if option.kind == "range_requires_roster" and not include_values:
                # No safe number can be published for this group: the union
                # is unproven and its range lower bound is the largest member.
                # Report the group and the missing roster proof, but no figure.
                entry.update(status=option.status,
                             reason="成员名单未证明互不重叠；该合并范围不能作为公开数字，"
                                    "需补名单证明或改合并范围",
                             reasons=[item.as_dict() for item in option.reasons])
                results.append(entry)
                continue
            post = self.bounds_including(visible, candidate, (metric,))
            entry["member_post_widths"] = {
                cell.label(): width(post.cell(cell, metric))
                for cell in group if (cell, metric) in post.bounds
            }
            if verdict.safe and fact is not None and fact.kind != "range":
                entry.update(status=MERGE_ONLY)
                if include_values:
                    entry["value"] = fact.value
            elif verdict.safe and option.kind == "range_requires_roster":
                entry.update(status=MERGE_REQUIRES_ROSTER,
                             lower=option.lower, upper=option.upper)
            else:
                entry.update(status=BLOCKED,
                             reasons=[reason.as_dict() for reason in verdict.reasons[:4]])
            results.append(entry)
        return results


def plan_for_period(ledger: Ledger, policy: Policy, visible: Sequence[Fact], period: str,
                    metrics: Sequence[str],
                    course_groups: Sequence[Sequence[str]] | None = None,
                    *, destination: str = "external-public",
                    audience: str = "public") -> tuple[list[Fact], BatchVerdict]:
    """Build the safest per-period sheet for the requested course groupings."""

    planner = Planner(ledger, policy)
    course_groups = course_groups or [[course] for course in ledger.course_types()]
    proposed: list[Fact] = []
    for district in ledger.districts():
        for courses in course_groups:
            for metric in metrics:
                cells = [CellKey(period, district, course) for course in courses
                         if CellKey(period, district, course) in ledger.cells]
                if not cells:
                    continue
                fact, _option = planner.aggregate_fact(
                    cells, metric, source_id=f"{period}/{district}",
                    destination=destination, audience=audience)
                if fact is not None:
                    proposed.append(fact)
    return proposed, planner.appraise(visible, proposed, metrics=metrics)
