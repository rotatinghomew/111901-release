import csv
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release import (  # noqa: E402
    ALREADY_EXPOSED,
    BLOCKED,
    INTERNAL,
    MERGE_ONLY,
    PUBLISHABLE,
    CellKey,
    Fact,
    Ledger,
    LineItem,
    Material,
    Policy,
    Workflow,
    build_visible_state,
    load_published,
    make_fact,
)
from release.planner import Planner  # noqa: E402


DATA = ROOT / "data"


class PolicyTest(unittest.TestCase):
    def test_fixture_policy(self):
        policy = Policy.load(DATA / "policy.json")
        self.assertEqual(policy.min_cell, 8)
        self.assertEqual(policy.max_interval_width, 4)
        self.assertEqual(policy.fixed_merge_dimensions, ("period", "district"))

    def test_merge_scope_keeps_district_and_period(self):
        policy = Policy.load(DATA / "policy.json")
        self.assertFalse(policy.allows_group_key(("district",)))
        self.assertFalse(policy.allows_group_key(("period",)))
        self.assertTrue(policy.allows_group_key(("period", "district")))
        self.assertTrue(policy.allows_group_key(("period", "district", "course_type")))

    def test_interval_floor(self):
        policy = Policy.load(DATA / "policy.json")
        self.assertTrue(policy.interval_is_protected(0, 3))
        self.assertFalse(policy.interval_is_protected(0, 2))
        self.assertFalse(policy.interval_is_protected(5, 5))

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            Policy.from_mapping({"min_cell": 0})
        with self.assertRaises(ValueError):
            Policy.from_mapping({"max_interval_width": -1})


class FixtureLedgerTest(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger.load(DATA / "counts.csv")
        self.policy = Policy.load(DATA / "policy.json")
        self.facts = load_published(DATA / "published.csv")

    def test_fixture_shape(self):
        self.assertEqual(len(self.ledger.keys()), 80)
        self.assertEqual(self.ledger.periods(), ("2026-01", "2026-02", "2026-03", "2026-04"))

    def test_empty_field_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "period,district,course_type,eligible,participants,attended\n"
                "2026-05,d0,t0,10,4,\n",
                encoding="utf-8")
            ledger = Ledger.load(path)
            key = CellKey("2026-05", "d0", "t0")
            self.assertEqual(ledger.value(key, "participants"), 4)
            self.assertIsNone(ledger.value(key, "attended"))
            self.assertIsNone(ledger.value(key, "confirmed"))
            planner = Planner(ledger, Policy.load(DATA / "policy.json"))
            self.assertIsNone(planner.rate([key]))
            reason = planner.rate_explanation([key])
            self.assertEqual(reason.code, "rate_missing_denominator")

    def test_published_rows_are_visible_and_conflicts_detected(self):
        state = build_visible_state(self.facts, self.ledger)
        self.assertEqual(len(state.visible_facts), 24)
        from release.planner import detect_conflicts

        conflicts = detect_conflicts(state.visible_facts, self.ledger)
        self.assertTrue(any("2026-01/d2/t0" in item for item in conflicts))
        self.assertTrue(any("2026-02/d4/t0" in item for item in conflicts))


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger.load(DATA / "counts.csv")
        self.cell = CellKey("2026-03", "d0", "t2")
        self.key = (self.cell, "participants")

    def test_retraction_stays_visible(self):
        fact = make_fact(source_id="r", metric="participants", value=2,
                         cells=[self.cell], kind="cell", status="retracted",
                         delivered="yes")
        self.assertTrue(fact.visible)
        self.assertTrue(fact.seen_risk)

    def test_uncertain_delivery_counts_as_visible(self):
        fact = make_fact(source_id="u", metric="participants", value=2,
                         cells=[self.cell], kind="cell", status="sent",
                         delivered="0")
        self.assertTrue(fact.visible)
        self.assertTrue(fact.seen_risk)

    def test_only_proof_makes_invisible(self):
        unproven = make_fact(source_id="f", metric="participants", value=2,
                             cells=[self.cell], kind="cell", status="failed",
                             delivered="0")
        proven = make_fact(source_id="f2", metric="participants", value=2,
                           cells=[self.cell], kind="cell", status="failed",
                           delivered="0", proof_unsent=True)
        self.assertTrue(unproven.visible)
        self.assertFalse(proven.visible)
        self.assertFalse(proven.seen_risk)

    def test_draft_is_pending(self):
        fact = make_fact(source_id="d", metric="participants", value=2,
                         cells=[self.cell], kind="cell", status="draft")
        self.assertFalse(fact.visible)


class IntervalAttackTest(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger.load(DATA / "counts.csv")
        self.policy = Policy.load(DATA / "policy.json")
        self.planner = Planner(self.ledger, self.policy)

    def test_month_total_then_every_other_cell_is_blocked(self):
        period = "2026-01"
        cells = self.ledger.where(period=period)
        target = CellKey(period, "d0", "t0")
        total = sum(self.ledger.value(cell, "participants") for cell in cells)
        proposed = [make_fact(source_id="total", metric="participants",
                              value=total, cells=cells, kind="sum")]
        proposed += [
            make_fact(source_id="rest", metric="participants",
                      value=self.ledger.value(cell, "participants"),
                      cells=[cell], kind="cell")
            for cell in cells if cell != target
        ]
        verdict = self.planner.appraise([], proposed, metrics=["participants"])
        self.assertFalse(verdict.safe)
        self.assertIn(target, verdict.blocked_cells)
        reason_codes = {reason.code for reason in verdict.reasons}
        self.assertIn("interval_too_narrow", reason_codes)

    def test_complementary_totals_are_blocked(self):
        period = "2026-04"
        cells = self.ledger.where(period=period)
        target = CellKey(period, "d0", "t0")
        others = tuple(cell for cell in cells if cell != target)
        total = sum(self.ledger.value(cell, "participants") for cell in cells)
        other_total = sum(self.ledger.value(cell, "participants") for cell in others)
        first = make_fact(source_id="all", metric="participants",
                          value=total, cells=cells, kind="sum")
        second = make_fact(source_id="rest", metric="participants",
                           value=other_total, cells=others, kind="sum")
        verdict = self.planner.appraise([first], [second], metrics=["participants"])
        self.assertFalse(verdict.safe)
        self.assertIn(target, verdict.blocked_cells)

    def test_overlapping_roster_is_not_additive(self):
        cells = (CellKey("2026-03", "d0", "t0"), CellKey("2026-03", "d0", "t1"))
        first = self.ledger.value(cells[0], "participants")
        second = self.ledger.value(cells[1], "participants")
        aggregate = self.ledger.aggregate(cells, "participants")
        self.assertIsNone(aggregate.exact)
        # Without a roster proof the union lies between the largest member and
        # the arithmetic sum; it is never reported as a point count.
        self.assertEqual(aggregate.lower, max(first, second))
        self.assertEqual(aggregate.upper, first + second)
        fact, option = self.planner.aggregate_fact(cells, "participants", source_id="m")
        self.assertEqual(option.kind, "range_requires_roster")
        self.assertEqual(option.lower, max(first, second))
        self.assertEqual(option.upper, first + second)
        # The range is not publishable: its lower bound equals the larger
        # member, which would disclose that member.
        self.assertIsNone(fact)
        self.assertEqual(option.status, "merge_requires_roster")

    def test_disjoint_roster_makes_a_sum_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "period,district,course_type,eligible,participants,attended,member_ids\n"
                "2026-05,d0,t0,10,4,4,a,b,c,d\n"
                "2026-05,d0,t1,10,5,5,e,f,g,h,i\n",
                encoding="utf-8")
            ledger = Ledger.load(path)
            planner = Planner(ledger, self.policy)
            fact, option = planner.aggregate_fact(
                ledger.where(period="2026-05"), "participants", source_id="m")
            self.assertEqual(option.kind, "exact")
            self.assertEqual(option.value, 9)
            self.assertEqual(fact.kind, "sum")

    def test_pairwise_totals_are_eliminated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "period,district,course_type,eligible,participants,attended\n"
                "2026-05,d0,t0,10,3,3\n"
                "2026-05,d0,t1,10,5,5\n"
                "2026-05,d0,t2,10,9,9\n",
                encoding="utf-8")
            ledger = Ledger.load(path)
            planner = Planner(ledger, self.policy)
            cells = ledger.where(period="2026-05")
            totals = [
                make_fact(source_id="ab", metric="participants", value=8,
                          cells=cells[:2], kind="sum"),
                make_fact(source_id="ac", metric="participants", value=12,
                          cells=(cells[0], cells[2]), kind="sum"),
                make_fact(source_id="bc", metric="participants", value=14,
                          cells=cells[1:], kind="sum"),
            ]
            verdict = planner.appraise([], totals, metrics=["participants"])
            self.assertFalse(verdict.safe)
            self.assertIn(cells[0], verdict.blocked_cells)

    def test_subset_residual_pins_a_single_cell(self):
        period = "2026-04"
        cells = self.ledger.where(period=period)
        target = CellKey(period, "d4", "t2")
        rest = tuple(cell for cell in cells if cell != target)
        total = sum(self.ledger.value(cell, "participants") for cell in cells)
        rest_total = sum(self.ledger.value(cell, "participants") for cell in rest)
        whole = make_fact(source_id="all", metric="participants", value=total,
                          cells=cells, kind="sum")
        remainder = make_fact(source_id="rest", metric="participants",
                              value=rest_total, cells=rest, kind="sum")
        verdict = self.planner.appraise([whole], [remainder],
                                        metrics=["participants"])
        self.assertFalse(verdict.safe)
        self.assertIn(target, verdict.blocked_cells)

    def test_wide_subset_residual_leaves_cells_unpinned(self):
        period = "2026-04"
        cells = self.ledger.where(period=period)
        rest = tuple(cell for cell in cells if cell.district != "d4")
        total = sum(self.ledger.value(cell, "participants") for cell in cells)
        rest_total = sum(self.ledger.value(cell, "participants") for cell in rest)
        whole = make_fact(source_id="all", metric="participants", value=total,
                          cells=cells, kind="sum")
        remainder = make_fact(source_id="rest2", metric="participants",
                              value=rest_total, cells=rest, kind="sum")
        verdict = self.planner.appraise([whole], [remainder],
                                        metrics=["participants"])
        # The residual is the sum of four cells, so each individual interval
        # stays wide.  Narrowing them further would need more releases.
        self.assertTrue(verdict.safe)

    def test_protected_value_never_in_explanations(self):
        period = "2026-03"
        cell = CellKey(period, "d0", "t2")
        value = self.ledger.value(cell, "participants")
        self.assertLess(value, self.policy.min_cell)
        verdict = self.planner.appraise(
            [], [make_fact(source_id="x", metric="participants", value=value,
                           cells=[cell], kind="cell")],
            metrics=["participants"], collect_assessments=True)
        payload = json.dumps(
            [reason.as_dict() for reason in verdict.reasons]
            + [assessment.to_public_dict() for assessment in verdict.assessments],
            ensure_ascii=False)
        self.assertNotIn(f'"value": {value}', payload)
        self.assertNotIn(f"={value}", payload)

    def test_missing_value_is_pending_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "period,district,course_type,eligible,participants,attended\n"
                "2026-05,d0,t0,10,,4\n",
                encoding="utf-8")
            ledger = Ledger.load(path)
            planner = Planner(ledger, self.policy)
            # An empty sheet is fileable (it carries no number).  The same
            # sheet with the attended value is not, because its denominator
            # cell has no registered value.
            self.assertTrue(planner.appraise([], [], metrics=["participants"]).safe)
            verdict = planner.appraise(
                [], [make_fact(source_id="x", metric="attended", value=4,
                               cells=[CellKey("2026-05", "d0", "t0")], kind="cell")],
                metrics=["participants", "attended"])
            self.assertFalse(verdict.safe)
            self.assertIn(CellKey("2026-05", "d0", "t0"), verdict.pending_cells)

    def test_rate_narrows_attended_when_denominator_is_complete(self):
        period = "2026-03"
        cells = self.ledger.where(period=period)
        # Exact attendance rate over the whole period plus every participant
        # cell but one leaves the last participation cell pinned.
        numerator = sum(self.ledger.value(cell, "attended") for cell in cells)
        denominator = sum(self.ledger.value(cell, "eligible") for cell in cells)
        fact = make_fact(
            source_id="rate", metric="attendance_rate",
            value=Fraction(numerator, denominator), cells=cells, kind="rate",
            numerator_metric="attended", denominator_metric="eligible",
            raw={"rate_exact": "yes"})
        report = self.planner.bounds_including(
            [], [fact], ["attended", "eligible"])
        self.assertTrue(report.complete)
        # The rate on its own cannot pin a member below the floor.
        widths = [report.width(cell, "attended") for cell in cells]
        self.assertTrue(all(width >= 4 for width in widths))


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.outbox = Path(self.directory.name) / "outbox"
        self.workflow = Workflow.from_files(
            counts=DATA / "counts.csv", policy=DATA / "policy.json",
            published=DATA / "published.csv", outbox=self.outbox)

    def _material(self, material_id, cells, values, *, recipients=("partner-a",),
                  audience="external-public", kind="cell"):
        items = tuple(
            LineItem(source_id=material_id, metric="participants", cells=(cell,),
                     value=Fraction(value) if value is not None else None,
                     kind=kind,
                     suppressed=value is None
                     or (self.workflow.policy.is_protected_metric("participants")
                         and value < self.workflow.policy.min_cell))
            for cell, value in zip(cells, values))
        return self.workflow.register_material(Material(
            material_id=material_id, release_id=material_id, period=cells[0].period,
            audience=audience, recipients=tuple(recipients), items=items,
            definition_version="default", policy_digest=self.workflow.policy.digest(),
            ledger_fingerprint=self.workflow.ledger.fingerprint(),
            visible_fingerprint=self.workflow.visible_fingerprint()))

    def test_period_sheet_suppresses_small_cells(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        self.assertEqual(len(material.suppressed_cells()), 12)
        self.assertTrue(self.workflow.appraise_material(material).releasable)
        for item in material.items:
            if item.value is not None and not item.suppressed:
                self.assertGreaterEqual(item.value, 8)

    def test_internal_report_keeps_values(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], audience="internal")
        self.assertEqual(material.suppressed_cells(), [])
        self.assertEqual(len(material.facts()), 20)

    def test_merged_group_without_roster_proof_is_suppressed(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"],
            course_groups=[["t0", "t1", "t2", "t3"]])
        # Without member lists the union is unproven; every merged group is
        # suppressed rather than published as a sum or a disclosive range.
        self.assertTrue(all(item.suppressed for item in material.items))
        self.assertEqual(material.facts(), [])
        self.assertTrue(self.workflow.appraise_material(material).releasable)

    def test_send_writes_receipt_and_visible_facts(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        schedule = self.workflow.schedule(approval, due_at="2026-04-05T09:00:00+08:00")
        before = len(self.workflow.visible)
        receipt = self.workflow.send(approval, delivered="yes", schedule=schedule)
        self.assertEqual(receipt.status, "sent")
        self.assertEqual(schedule.status, "sent")
        self.assertGreater(len(self.workflow.visible), before)
        files = list(self.outbox.glob("*.json"))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(record["receipt"]["material_id"], material.material_id)
        self.assertEqual(record["material"]["content_hash"], material.content_hash)

    def test_uncertain_delivery_is_visible(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        receipt = self.workflow.send(approval, delivered="unknown")
        self.assertTrue(receipt.visible)
        self.assertTrue(receipt.uncertain)
        self.assertTrue(any(fact.source_id == material.material_id
                            for fact in self.workflow.visible))

    def test_proven_undelivered_is_removed(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        receipt = self.workflow.send(approval, delivered="unknown")
        self.workflow.prove_undelivered(receipt.receipt_id, evidence="returned envelope")
        self.assertFalse(any(fact.source_id == material.material_id
                             for fact in self.workflow.visible))

    def test_retraction_does_not_remove_visibility(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        self.workflow.send(approval, delivered="yes")
        self.workflow.retract(material, reason="correction")
        self.assertTrue(any(fact.source_id == material.material_id
                            for fact in self.workflow.visible))

    def _sum_material(self, material_id, cells, *, recipients=("partner-a",)):
        total = sum(self.workflow.ledger.value(cell, "participants") for cell in cells)
        items = (LineItem(source_id=material_id, metric="participants",
                          cells=tuple(cells), value=Fraction(total), kind="sum"),)
        return self.workflow.register_material(Material(
            material_id=material_id, release_id=material_id, period=cells[0].period,
            audience="external-public", recipients=tuple(recipients), items=items,
            definition_version="default",
            policy_digest=self.workflow.policy.digest(),
            ledger_fingerprint=self.workflow.ledger.fingerprint(),
            visible_fingerprint=self.workflow.visible_fingerprint()))

    def test_second_editor_approval_withdrawn_after_first_send(self):
        period = "2026-04"
        cells = self.workflow.ledger.where(period=period)
        small = CellKey(period, "d0", "t0")
        others = tuple(cell for cell in cells if cell != small)
        whole = self._sum_material("whole", cells)
        rest = self._sum_material("rest", others)
        self.assertTrue(self.workflow.appraise_material(whole).releasable)
        self.assertTrue(self.workflow.appraise_material(rest).releasable)
        approval_whole = self.workflow.approve(whole, editor="alice")
        approval_rest = self.workflow.approve(rest, editor="bob")
        self.workflow.send(approval_whole, delivered="yes")
        self.assertEqual(approval_rest.status, "withdrawn")
        with self.assertRaises(ValueError):
            self.workflow.send(approval_rest, delivered="yes")

    def test_editor_can_change_aggregation_and_reapprove(self):
        period = "2026-04"
        cells = self.workflow.ledger.where(period=period)
        whole = self._sum_material("whole2", cells)
        approval = self.workflow.approve(whole, editor="alice")
        self.workflow.send(approval, delivered="yes")
        revised = self._sum_material("revised", self.workflow.ledger.where(period="2026-03"),
                                     recipients=("partner-b",))
        new_approval = self.workflow.revise(
            self.workflow.approvals[approval.approval_id],
            material=revised, editor="alice")
        self.assertEqual(new_approval.status, "approved")

    def test_delivery_manifest_joins_receipts_and_versions(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        self.workflow.send(approval, recipients=["partner-a"], delivered="yes")
        manifest = self.workflow.delivery_manifest()
        row = next(item for item in manifest["materials"]
                   if item["material_id"] == material.material_id)
        self.assertEqual(row["status"], "sent")
        self.assertEqual(row["receipts"][0]["recipient"], "partner-a")
        self.assertEqual(row["receipts"][0]["content_hash"], material.content_hash)
        self.assertEqual(row["approvals"][0]["status"], "sent")

    def test_scheduling_two_materials_for_same_reader_is_jointly_checked(self):
        period = "2026-04"
        cells = self.workflow.ledger.where(period=period)
        small = CellKey(period, "d0", "t0")
        others = tuple(cell for cell in cells if cell != small)
        whole = self._sum_material("whole3", cells)
        rest = self._sum_material("rest3", others)
        approval_whole = self.workflow.approve(whole, editor="alice")
        approval_rest = self.workflow.approve(rest, editor="bob")
        self.workflow.schedule(approval_whole, due_at="2026-04-01T09:00:00+08:00")
        with self.assertRaises(ValueError):
            self.workflow.schedule(approval_rest, due_at="2026-04-01T10:00:00+08:00")

    def test_resend_is_checked_against_what_readers_already_have(self):
        period = "2026-04"
        cells = self.workflow.ledger.where(period=period)
        whole = self._sum_material("whole4", cells, recipients=("partner-a",))
        approval = self.workflow.approve(whole, editor="alice")
        self.workflow.send(approval, delivered="yes")
        clone, new_approval = self.workflow.resend(whole, editor="alice")
        self.assertEqual(clone.version, 2)
        # The resent version repeats the same content, which is safe because
        # it reveals nothing beyond the already-visible version.
        self.assertEqual(new_approval.status, "approved")

    def test_approval_records_visible_fingerprint_and_version(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        self.assertEqual(approval.visible_fingerprint,
                         self.workflow.visible_fingerprint())
        manifest = material.manifest()
        self.assertEqual(manifest["definition_version"], "default")
        self.assertEqual(manifest["policy_digest"], self.workflow.policy.digest())
        self.assertIn("ledger_fingerprint", manifest)

    def test_withdraw_schedule_marks_material(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        approval = self.workflow.approve(material, editor="alice")
        schedule = self.workflow.schedule(approval, due_at="2026-04-01T09:00:00+08:00")
        self.workflow.withdraw_schedule(schedule, reason="typo in course label")
        self.assertEqual(schedule.status, "withdrawn")
        self.assertEqual(material.status, "withdrawn")

    def test_conflicting_visible_record_blocks_period(self):
        material = self.workflow.prepare_period_material(
            period="2026-01", metrics=["participants"], recipients=["partner-a"])
        with self.assertRaises(ValueError):
            self.workflow.approve(material, editor="alice")
        explanation = self.workflow.explain_material(material)
        self.assertFalse(explanation["releasable"])
        self.assertTrue(any("2026-01/d2/t0" in item
                            for item in explanation["conflicts"]))

    def test_external_alternative_explanation_has_no_values(self):
        material = self.workflow.prepare_period_material(
            period="2026-03", metrics=["participants"], recipients=["partner-a"])
        alternatives = self.workflow.alternatives(material, "participants")
        payload = json.dumps(alternatives, ensure_ascii=False)
        for cell in self.workflow.ledger.where(period="2026-03"):
            value = self.workflow.ledger.value(cell, "participants")
            if value is not None and value < 8:
                self.assertNotIn(f'"value": {value}', payload)
        with self.assertRaises(ValueError):
            self.workflow.alternatives(material, "participants", include_values=True)

    def test_no_network_import(self):
        text = (ROOT / "src" / "release" / "workflow.py").read_text(encoding="utf-8")
        for forbidden in ("smtplib", "socket", "urllib", "requests", "http.client"):
            self.assertNotIn(forbidden, text)
        text = (ROOT / "src" / "release" / "planner.py").read_text(encoding="utf-8")
        for forbidden in ("smtplib", "socket", "urllib", "requests", "http.client"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
