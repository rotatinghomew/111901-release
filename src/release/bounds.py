"""Interval reasoning about small counts.

A reader never sees the internal ledger.  What a reader can infer is an
integer interval for each cell, from

* the ledger structure (every count is a non-negative integer);
* previously published exact cells and totals;
* the candidate release being evaluated;
* published attendance rates whose numerator and denominator are exact;
* exact linear combinations of published totals.

The solver propagates intervals to a fixed point and then performs exact
Gaussian elimination over the exact totals, so subset and pivot differences are
accounted for.  It is deliberately conservative:

* a total over a unique-person count is **not** treated as additive unless the
  rosters are proven disjoint, because one resident may appear in several
  course types.  It only gives a union lower bound and a ceiling per cell;
* overlapping or unknown rosters never produce a "sum equals total" equation;
* a rate is used only when it was generated as a ratio of the reported sums,
  with a complete denominator.  A rate over unique-person counts with
  overlapping rosters is not used;
* a missing ledger value is not zero.  It simply has the full initial
  interval and cannot be proven safe.

The arithmetic is conservative: when the propagator cannot decide something,
it leaves a wider interval, so a plan that passes cannot be narrower than
reality.  The one deliberate assumption, documented in :func:`_collect_count`,
is that a publisher presenting an exact total as a count is assumed to mean an
additive count unless the ledger proves the rosters overlap.

Terminology: the *width* of an integer interval ``[lo, hi]`` is the number of
possible values, ``hi - lo + 1``.  ``max_interval_width`` is the smallest
acceptable width for a protected cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor
from typing import Iterable, Sequence

from .model import CellKey, Fact, Ledger, MetricDefinition, canonical_metric
from .policy import Policy


def _components(constraints: Sequence[SumConstraint]) -> list[list[SumConstraint]]:
    """Split exact totals into groups that share cells (connected components)."""

    parent: dict[CellKey, CellKey] = {}

    def find(cell: CellKey) -> CellKey:
        parent.setdefault(cell, cell)
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for item in constraints:
        cells = list(item.cells)
        for other in cells[1:]:
            root_a, root_b = find(cells[0]), find(other)
            if root_a != root_b:
                parent[root_b] = root_a
    groups: dict[CellKey, list[SumConstraint]] = {}
    for item in constraints:
        if not item.cells:
            continue
        groups.setdefault(find(item.cells[0]), []).append(item)
    return list(groups.values())


def _solve_singletons(rows: Sequence[Sequence[Fraction]], variables: int) -> dict[int, int]:
    """Return variables pinned to a non-negative integer by ``rows``.

    ``rows`` are ``A x = b`` with an augmented final column.  RREF is computed
    exactly.  A row with exactly one nonzero coefficient solves that variable;
    rows that merely constrain a combination are ignored, because the interval
    propagator already handles them soundly.
    """

    matrix = [list(row) for row in rows]
    pivot_columns: list[int] = []
    row = 0
    for column in range(variables):
        pivot = next((candidate for candidate in range(row, len(matrix))
                      if matrix[candidate][column] != 0), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        divisor = matrix[row][column]
        matrix[row] = [value / divisor for value in matrix[row]]
        for other in range(len(matrix)):
            if other != row and matrix[other][column] != 0:
                factor = matrix[other][column]
                matrix[other] = [value - factor * base
                                 for value, base in zip(matrix[other], matrix[row])]
        pivot_columns.append(column)
        row += 1
        if row == len(matrix):
            break
    solved: dict[int, int] = {}
    for line in matrix:
        nonzero = [index for index in range(variables) if line[index] != 0]
        if len(nonzero) != 1:
            continue
        value = line[-1] / line[nonzero[0]]
        if value.denominator == 1 and value >= 0:
            solved[nonzero[0]] = int(value)
    return solved


Bounds = tuple[int, int]
DEFAULT_MAX_COUNT = 10**6
INFINITY = 10**9


def width(bounds: Bounds) -> int:
    return bounds[1] - bounds[0] + 1


def intersects(left: Bounds, right: Bounds) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


@dataclass(frozen=True)
class SumConstraint:
    """``lo <= sum(cells) <= hi`` for one metric.

    ``kind="sum"`` is an additive equality/range.  ``kind="union"`` is a
    unique-person total whose rosters are not proven disjoint: it implies each
    member ``<= hi`` and ``sum(members) >= lo``, and never that the members add
    up to the total.
    """

    cells: tuple[CellKey, ...]
    metric: str
    lower: int
    upper: int
    kind: str = "sum"
    source: str | None = None


@dataclass(frozen=True)
class RateConstraint:
    cells: tuple[CellKey, ...]
    numerator: str
    denominator: str
    denominator_cells: tuple[CellKey, ...]
    value: Fraction
    source: str | None = None
    exact: bool = True
    tolerance: Fraction = Fraction(0)


@dataclass
class ConstraintSet:
    sums: list[SumConstraint] = field(default_factory=list)
    rates: list[RateConstraint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


@dataclass
class BoundsReport:
    bounds: dict[tuple[CellKey, str], Bounds]
    constraints: ConstraintSet
    complete: bool
    notes: list[str] = field(default_factory=list)

    def cell(self, key: CellKey, metric: str) -> Bounds:
        return self.bounds[(key, canonical_metric(metric))]

    def width(self, key: CellKey, metric: str) -> int:
        return width(self.cell(key, metric))


def _units_ok(fact: Fact, cells: Iterable[CellKey], ledger: Ledger) -> bool:
    for cell in cells:
        statement = ledger.cells.get(cell)
        if statement is None:
            return False
        if fact.definition_version and statement.definition_version and \
                fact.definition_version != statement.definition_version:
            return False
        if fact.unit and statement.unit and fact.unit != statement.unit:
            return False
    return True


class BoundsSolver:
    def __init__(self, ledger: Ledger, policy: Policy, facts: Sequence[Fact], *,
                 include_pending: bool = False, max_iterations: int = 200):
        self.ledger = ledger
        self.policy = policy
        self.visible = tuple(facts if include_pending else
                             (fact for fact in facts if fact.visible))
        self.max_iterations = max_iterations

    @classmethod
    def including_proposed(cls, ledger: Ledger, policy: Policy, visible: Sequence[Fact],
                           proposed: Sequence[Fact], *,
                           max_iterations: int = 200) -> "BoundsSolver":
        """Treat pending facts as visible for a what-if calculation.

        A copy is made; the caller's records are never mutated.
        """

        combined = list(visible)
        for fact in proposed:
            if fact.visible:
                combined.append(fact)
            else:
                combined.append(Fact(
                    source_id=fact.source_id, metric=fact.metric, value=fact.value,
                    cells=fact.cells, kind=fact.kind, status="sent", delivered="no",
                    visible=True, seen_risk=fact.seen_risk, revision=fact.revision,
                    definition_version=fact.definition_version, unit=fact.unit,
                    roster=fact.roster, numerator_metric=fact.numerator_metric,
                    denominator_metric=fact.denominator_metric,
                    denominator_cells=fact.denominator_cells,
                    destination=fact.destination, evidence=fact.evidence, raw=fact.raw,
                ))
        return cls(ledger, policy, combined, max_iterations=max_iterations)

    # -- constraint collection -------------------------------------------

    def collect(self, metrics: Iterable[str]) -> ConstraintSet:
        wanted = {canonical_metric(metric) for metric in metrics}
        result = ConstraintSet()
        for fact in self.visible:
            if fact.is_rate:
                self._collect_rate(fact, wanted, result)
                continue
            self._collect_count(fact, wanted, result)
        return result

    def _collect_rate(self, fact: Fact, wanted: set[str], result: ConstraintSet) -> None:
        numerator = fact.numerator_metric or "attended"
        denominator = fact.denominator_metric or "eligible"
        if numerator not in wanted and denominator not in wanted:
            return
        cells = tuple(cell for cell in fact.cells if cell in self.ledger.cells)
        denominator_cells = tuple(
            cell for cell in (fact.denominator_cells or cells)
            if cell in self.ledger.cells
        )
        if not cells or not denominator_cells:
            result.ignored.append(f"{fact.source_id}: rate without complete scope")
            return
        if not _units_ok(fact, cells, self.ledger):
            result.ignored.append(f"{fact.source_id}: rate definition/unit mismatch")
            return
        if fact.value is None or fact.value < 0:
            result.ignored.append(f"{fact.source_id}: invalid rate value")
            return
        exact = str(fact.raw.get("rate_exact", "")).strip().lower() in {"1", "true", "yes"}
        # A rate is a ratio of reported sums.  If the numerator's rosters are
        # not proven additive, the numerator is not necessarily the sum of the
        # cells and the rate must not be used to narrow them.
        numerator_definition = MetricDefinition.for_metric(numerator)
        relation = (self.ledger.roster_relation(cells, numerator)
                    if len(cells) > 1 else "same")
        if not (numerator_definition.additive or relation in {"disjoint", "same"}) \
                and relation != "unknown":
            exact = False
        result.rates.append(RateConstraint(
            cells=cells, numerator=numerator, denominator=denominator,
            denominator_cells=denominator_cells, value=Fraction(fact.value),
            source=fact.source_id, exact=exact,
            tolerance=self.policy.rate_tolerance,
        ))

    def _collect_count(self, fact: Fact, wanted: set[str], result: ConstraintSet) -> None:
        covers = fact.raw.get("covers_metric")
        metric = canonical_metric(str(covers)) if covers else fact.canonical_metric
        if metric not in wanted:
            return
        cells = tuple(cell for cell in fact.cells if cell in self.ledger.cells)
        if not cells:
            result.ignored.append(f"{fact.source_id}: no ledger cells in scope")
            return
        if not _units_ok(fact, cells, self.ledger):
            result.ignored.append(
                f"{fact.source_id}: definition/unit mismatch for {metric}")
            return
        definition = MetricDefinition.for_metric(metric)
        relation = self.ledger.roster_relation(cells, metric) if len(cells) > 1 else "same"
        # Risk model: an exact total that the publisher presents as a count is
        # assumed additive by a reader unless the ledger *proves* that the
        # rosters overlap.  We cannot prove a residual is wide merely because
        # member lists are absent, so absence must not relax the check.  When
        # rosters are proven overlapping, the total is only a union bound and
        # never an equality.
        additive = (
            definition.additive
            or relation in {"disjoint", "same"}
            or relation == "unknown"
        ) and relation != "overlapping"

        if fact.kind == "range":
            try:
                lower = int(Fraction(str(fact.raw.get("lower"))))
                upper = int(Fraction(str(fact.raw.get("upper"))))
            except (TypeError, ValueError, ZeroDivisionError):
                result.ignored.append(f"{fact.source_id}: invalid range")
                return
            if lower < 0 or upper < lower:
                result.ignored.append(f"{fact.source_id}: invalid range")
                return
            result.sums.append(SumConstraint(
                cells=cells, metric=metric, lower=lower, upper=upper,
                kind="sum" if additive else "union", source=fact.source_id))
            return

        if fact.value is None:
            result.ignored.append(f"{fact.source_id}: count without value")
            return
        if fact.value.denominator != 1:
            result.ignored.append(f"{fact.source_id}: non-integer count")
            return
        number = int(fact.value)
        if number < 0:
            result.ignored.append(f"{fact.source_id}: negative count")
            return
        if fact.kind in {"union", "visit_sum"}:
            result.sums.append(SumConstraint(
                cells=cells, metric=metric, lower=number, upper=INFINITY,
                kind="union", source=fact.source_id))
            return
        if additive:
            result.sums.append(SumConstraint(
                cells=cells, metric=metric, lower=number, upper=number,
                kind="sum", source=fact.source_id))
        elif fact.kind == "cell" and len(cells) == 1:
            # An exact single-cell count is a point value even for a
            # unique-person metric: it is the cell's own roster.
            result.sums.append(SumConstraint(
                cells=cells, metric=metric, lower=number, upper=number,
                kind="sum", source=fact.source_id))
        else:
            # A unique-person total over unproven rosters is a union lower
            # bound, not a sum.
            result.sums.append(SumConstraint(
                cells=cells, metric=metric, lower=number, upper=number,
                kind="union", source=fact.source_id))

    # -- propagation ------------------------------------------------------

    def _max_count(self) -> int:
        raw = self.policy.raw.get("max_count", DEFAULT_MAX_COUNT)
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return DEFAULT_MAX_COUNT

    def propagate(self, metrics: Iterable[str]) -> BoundsReport:
        metrics = tuple(dict.fromkeys(canonical_metric(metric) for metric in metrics))
        constraints = self.collect(metrics)
        cap = self._max_count()
        bounds: dict[tuple[CellKey, str], Bounds] = {
            (key, metric): (0, cap)
            for key in self.ledger.keys() for metric in metrics
        }
        notes = constraints.notes
        complete = True

        for _ in range(self.max_iterations):
            changed = False
            for constraint in constraints.sums:
                if constraint.metric not in metrics:
                    continue
                cells = [cell for cell in constraint.cells
                         if (cell, constraint.metric) in bounds]
                if not cells:
                    continue
                changed |= self._apply_sum(constraint, cells, bounds, notes)
                if self._contradicted:
                    complete = False
                    self._contradicted = False
            changed |= self._apply_rates(constraints, metrics, bounds, notes)
            if self._contradicted:
                complete = False
                self._contradicted = False
            changed |= self._apply_derived(constraints, metrics, bounds, notes)
            if self._contradicted:
                complete = False
                self._contradicted = False
            if not changed:
                break
        return BoundsReport(bounds=bounds, constraints=constraints,
                            complete=complete, notes=notes)

    def _apply_derived(self, constraints: ConstraintSet,
                       metrics: Sequence[str],
                       bounds: dict[tuple[CellKey, str], Bounds],
                       notes: list[str]) -> bool:
        """Eliminate exact totals against each other to a fixed point.

        A reader who sees exact totals can combine them: ``A`` and ``B`` with
        ``B ⊂ A`` give ``sum(A \\ B) = A - B``, and three pairwise totals
        ``a+b``, ``a+c``, ``b+c`` give ``2a = (a+b)+(a+c)-(b+c)``.  Plain
        interval propagation treats the members independently and misses
        both.  This routine performs exact Gaussian elimination over the
        rationals, component by component, and applies any single-cell value
        it can solve.

        Only exact totals are used, so no roster or rounding assumption is
        introduced beyond the additivity already encoded in the constraint's
        ``kind``.  Derived equations are computed afresh on each round.
        """

        changed = False
        for metric in metrics:
            exacts = [
                constraint for constraint in constraints.sums
                if constraint.metric == metric and constraint.kind == "sum"
                and constraint.lower == constraint.upper
            ]
            if len(exacts) < 2:
                continue
            for component in _components(exacts):
                cells = sorted({cell for item in component for cell in item.cells
                                if (cell, metric) in bounds})
                if not cells or len(cells) > 64 or len(component) > 64:
                    continue
                position = {cell: index for index, cell in enumerate(cells)}
                rows: list[list[Fraction]] = []
                for item in component:
                    row = [Fraction(0)] * (len(cells) + 1)
                    for cell in item.cells:
                        index = position.get(cell)
                        if index is not None:
                            row[index] += 1
                    row[-1] = Fraction(item.lower)
                    rows.append(row)
                for index, value in _solve_singletons(rows, len(cells)).items():
                    cell = cells[index]
                    current = bounds[(cell, metric)]
                    tightened = (max(current[0], value), min(current[1], value))
                    if tightened[0] > tightened[1]:
                        notes.append(f"derived value conflicts for {cell.label()} {metric}")
                        self._contradicted = True
                        continue
                    if tightened != current:
                        bounds[(cell, metric)] = tightened
                        changed = True
        return changed

    _contradicted = False

    def _apply_sum(self, constraint: SumConstraint, cells: list[CellKey],
                   bounds: dict[tuple[CellKey, str], Bounds], notes: list[str]) -> bool:
        metric = constraint.metric
        changed = False
        current = [bounds[(cell, metric)] for cell in cells]
        lo_sum = sum(lo for lo, _hi in current)
        hi_sum = sum(hi for _lo, hi in current)
        if constraint.kind == "union":
            total = constraint.lower
            if total > hi_sum:
                # The unique-person total cannot exceed the sum of the cells'
                # maxima, so the visible records are inconsistent.
                notes.append(f"union total {constraint.source} exceeds cell capacity")
                self._contradicted = True
                return changed
            for cell, (lo, hi) in zip(cells, current):
                if hi > total:
                    bounds[(cell, metric)] = (lo, min(hi, total))
                    changed = True
            current = [bounds[(cell, metric)] for cell in cells]
            max_others = sum(hi for _lo, hi in current)
            for cell, (lo, hi) in zip(cells, current):
                lower = max(lo, total - (max_others - hi))
                if lower > hi:
                    notes.append(f"union total {constraint.source} conflicts for {cell.label()}")
                    self._contradicted = True
                    continue
                if lower != lo:
                    bounds[(cell, metric)] = (lower, hi)
                    changed = True
            return changed

        if lo_sum > constraint.upper or hi_sum < constraint.lower:
            notes.append(f"inconsistent total {constraint.source} for {metric}")
            self._contradicted = True
            return changed
        for cell, (lo, hi) in zip(cells, current):
            others_lo = lo_sum - lo
            others_hi = hi_sum - hi
            new_lo = max(lo, constraint.lower - others_hi, 0)
            new_hi = min(hi, constraint.upper - others_lo)
            if new_lo > new_hi:
                notes.append(f"conflicting constraints for {cell.label()} {metric}")
                self._contradicted = True
                continue
            if (new_lo, new_hi) != (lo, hi):
                bounds[(cell, metric)] = (new_lo, new_hi)
                changed = True
        return changed

    def _apply_rates(self, constraints: ConstraintSet, metrics: Sequence[str],
                     bounds: dict[tuple[CellKey, str], Bounds], notes: list[str]) -> bool:
        changed = False
        for rate in constraints.rates:
            if not rate.exact:
                continue
            if rate.numerator not in metrics or rate.denominator not in metrics:
                continue
            denom_bounds = [bounds.get((cell, rate.denominator))
                            for cell in rate.denominator_cells]
            if any(entry is None for entry in denom_bounds):
                continue
            d_lo = sum(entry[0] for entry in denom_bounds)  # type: ignore[index]
            d_hi = sum(entry[1] for entry in denom_bounds)  # type: ignore[index]
            if d_lo == d_hi and d_lo == 0:
                notes.append(f"rate {rate.source}: zero denominator")
                continue
            tolerance = rate.tolerance
            n_lo = max(0, ceil((rate.value - tolerance) * d_lo)) if d_lo else 0
            n_hi = floor((rate.value + tolerance) * d_hi)
            if n_lo > n_hi:
                notes.append(f"rate {rate.source}: no integer numerator fits denominator range")
                self._contradicted = True
                continue
            cells = [cell for cell in rate.cells if (cell, rate.numerator) in bounds]
            if not cells:
                continue
            constraint = SumConstraint(
                cells=tuple(cells), metric=rate.numerator,
                lower=n_lo, upper=n_hi, kind="sum", source=rate.source)
            changed |= self._apply_sum(constraint, cells, bounds, notes)
        return changed
