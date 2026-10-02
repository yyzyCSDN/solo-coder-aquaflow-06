import unittest
from datetime import datetime, timezone

from aquaflow.api import AquaFlowService
from aquaflow.customers.triage import (
    Complaint,
    ComplaintTriageQueue,
    LeakIndicator,
    SensorSignal,
    TriageConfig,
    build_evidence,
    categorize,
)

UTC = timezone.utc


def dt(hour: int) -> datetime:
    return datetime(2026, 10, 2, hour, 0, tzinfo=UTC)


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.q = AquaFlowService().triage_queue(TriageConfig(evidence_threshold=2))

    # --- scenario 1: repeat complaints are grouped, never double-counted ---
    def test_repeat_complaints_group_by_dma_and_dedup(self):
        c1 = Complaint("C-1", "DMA-N7", "pressure", dt(8), account="A100",
                       issue="low pressure in shower")
        c2 = Complaint("C-2", "DMA-N7", "pressure", dt(9), account="A100",
                       issue="still low pressure")           # same customer -> repeat
        c3 = Complaint("C-3", "DMA-N7", "quality", dt(9), account="A200",
                       issue="brown discoloured water")

        r1 = self.q.ingest(complaints=[c1])[0]
        self.assertEqual(r1.evidence_count, 1)
        self.assertIsNone(r1.investigation_id)
        self.assertEqual(r1.next_action, "await_corroboration")

        r2 = self.q.ingest(complaints=[c2])[0]
        self.assertTrue(r2.is_duplicate)
        self.assertEqual(r2.evidence_count, 1)               # repeat does not add evidence
        self.assertIsNone(r2.investigation_id)

        r3 = self.q.ingest(complaints=[c3])[0]
        self.assertFalse(r3.is_duplicate)
        self.assertEqual(r3.evidence_count, 2)               # two independent witnesses
        self.assertIsNotNone(r3.investigation_id)
        self.assertEqual(r3.affected_area, "DMA-N7")
        self.assertEqual(r3.next_action, "confirm_or_assign")
        self.assertEqual(r3.complaint_ids, ("C-1", "C-2", "C-3"))

        inv = self.q.investigation("DMA-N7")
        # originals preserved as immutable references
        self.assertEqual(inv.complaint_refs, ("C-1", "C-2", "C-3"))
        self.assertEqual(self.q.complaint("C-1").issue, "low pressure in shower")

    def test_repeat_complaint_with_indicator_still_opens_case(self):
        # repeat complaint + DMA leak indicator => 2 independent buckets
        results = self.q.ingest(
            complaints=[
                Complaint("C-1", "DMA-E2", "pressure", dt(8), account="A1"),
                Complaint("C-2", "DMA-E2", "pressure", dt(10), account="A1"),
            ],
            indicators=[LeakIndicator("DMA-E2", "night_flow", 14.0, 10.0, at=dt(10))],
        )
        r = results[0]
        self.assertTrue(r.is_duplicate)
        self.assertEqual(r.evidence_count, 2)
        self.assertIsNotNone(r.investigation_id)

    # --- scenario 2: one strong sensor alone, no cross-corroboration --------
    def test_single_strong_sensor_without_corroboration(self):
        results = self.q.ingest(signals=[
            SensorSignal("S-99", "DMA-W4", "acoustic", strength=0.95, threshold=0.6, at=dt(7)),
        ])
        r = results[0]
        self.assertEqual(r.evidence_count, 1)
        self.assertIsNone(r.investigation_id)                # no complaint, no cross-support
        self.assertEqual(r.next_action, "await_corroboration")

        # a sensor corroborated by one complaint reaches threshold
        r2 = self.q.ingest(complaints=[
            Complaint("C-9", "DMA-W4", "pressure", dt(11), account="A9"),
        ])[0]
        self.assertEqual(r2.evidence_count, 2)
        self.assertIsNotNone(r2.investigation_id)

    def test_two_sensors_in_dma_without_complaint_still_waits(self):
        # sensors corroborate each other but no member of the public reported it
        results = self.q.ingest(signals=[
            SensorSignal("S-1", "DMA-W5", "acoustic", 0.8, 0.6),
            SensorSignal("S-2", "DMA-W5", "pressure_drop", 0.9, 0.6),
        ])
        r = results[0]
        self.assertEqual(r.evidence_count, 2)
        self.assertIsNone(r.investigation_id)

    # --- scenario 3: late complaint arrives after dispatch -------------------
    def test_late_complaint_after_dispatch(self):
        self.q.ingest(
            complaints=[
                Complaint("C-1", "DMA-S1", "pressure", dt(8), account="A1"),
                Complaint("C-2", "DMA-S1", "quality", dt(8), account="A2"),
            ],
        )
        self.q.apply("DMA-S1", "confirm", dt(9))
        self.q.apply("DMA-S1", "assign", dt(10), assignee="crew-leak-7")
        inv = self.q.investigation("DMA-S1")
        self.assertEqual(inv.status, "assigned")
        self.assertEqual(inv.assignee, "crew-leak-7")

        r = self.q.ingest(complaints=[
            Complaint("C-3", "DMA-S1", "pressure", dt(15), account="A3",
                      issue="pressure dropped again"),
        ])[0]
        self.assertTrue(r.is_late)
        self.assertFalse(r.is_duplicate)
        self.assertEqual(r.investigation_id, inv.investigation_id)   # same case, no duplicate
        self.assertEqual(r.next_action, "investigate_on_site")
        self.assertEqual(r.evidence_count, 3)
        self.assertEqual(self.q.investigation("DMA-S1").evidence_count, 3)
        self.assertIn("C-3", self.q.investigation("DMA-S1").complaint_refs)
        self.assertTrue(any(h[0] == "late_complaint" for h in inv.history))

    def test_late_repeat_complaint_from_same_customer(self):
        self.q.ingest(complaints=[
            Complaint("C-1", "DMA-S2", "pressure", dt(8), account="A1"),
            Complaint("C-2", "DMA-S2", "quality", dt(8), account="A2"),
        ])
        self.q.apply("DMA-S2", "assign", dt(9), assignee="crew-2")
        r = self.q.ingest(complaints=[
            Complaint("C-3", "DMA-S2", "pressure", dt(20), account="A1"),
        ])[0]
        self.assertTrue(r.is_late)
        self.assertTrue(r.is_duplicate)
        self.assertEqual(r.evidence_count, 2)                # same witness, same count

    # --- lifecycle: confirm / assign / defer / close ------------------------
    def test_close_requires_disposition(self):
        self.q.ingest(complaints=[
            Complaint("C-1", "DMA-K1", "pressure", dt(8), account="A1"),
            Complaint("C-2", "DMA-K1", "pressure", dt(8), account="A2"),
        ])
        self.q.apply("DMA-K1", "confirm", dt(9))
        with self.assertRaises(ValueError):
            self.q.apply("DMA-K1", "close", dt(11))         # no conclusion -> rejected

        inv = self.q.apply(
            "DMA-K1", "close", dt(11),
            disposition="burst service main repaired; pressure restored",
        )
        self.assertEqual(inv.status, "closed")
        self.assertEqual(inv.disposition, "burst service main repaired; pressure restored")
        self.assertEqual(self.q.investigation("DMA-K1").evidence_count, 2)

    def test_defer_then_resume_then_close(self):
        self.q.ingest(complaints=[
            Complaint("C-1", "DMA-K2", "pressure", dt(8), account="A1"),
            Complaint("C-2", "DMA-K2", "quality", dt(8), account="A2"),
        ])
        self.q.apply("DMA-K2", "defer", dt(9), note="awaiting night-flow data")
        self.assertEqual(self.q.investigation("DMA-K2").status, "deferred")
        self.assertEqual(self.q.investigation("DMA-K2").evidence_count, 2)
        self.q.apply("DMA-K2", "assign", dt(12), assignee="crew-3")
        with self.assertRaises(ValueError):
            self.q.apply("DMA-K2", "close", dt(13))
        self.q.apply("DMA-K2", "close", dt(14), disposition="no leak found; PRV adjusted")
        self.assertEqual(self.q.investigation("DMA-K2").status, "closed")

    def test_assign_requires_assignee(self):
        self.q.ingest(complaints=[
            Complaint("C-1", "DMA-K3", "pressure", dt(8), account="A1"),
            Complaint("C-2", "DMA-K3", "quality", dt(8), account="A2"),
        ])
        with self.assertRaises(ValueError):
            self.q.apply("DMA-K3", "assign", dt(9))

    # --- invariants -----------------------------------------------------------
    def test_original_complaints_are_immutable_references(self):
        original = Complaint("C-1", "DMA-Z1", "pressure", dt(8), account="A1")
        self.q.ingest(complaints=[
            original,
            Complaint("C-2", "DMA-Z1", "quality", dt(8), account="A2"),
        ])
        inv = self.q.investigation("DMA-Z1")
        stored = self.q.complaint("C-1")
        self.assertIs(stored, original)
        with self.assertRaises(Exception):
            stored.category = "quality"                        # frozen record
        for ref in inv.complaint_refs:
            self.q.complaint(ref)                              # every ref resolves

    def test_duplicate_complaint_id_rejected(self):
        self.q.ingest(complaints=[Complaint("C-1", "DMA-Z2", "pressure", dt(8), account="A1")])
        with self.assertRaises(ValueError):
            self.q.ingest(complaints=[Complaint("C-1", "DMA-Z2", "quality", dt(9), account="A2")])

    def test_different_dmas_are_independent_queues(self):
        results = self.q.ingest(
            complaints=[
                Complaint("C-1", "DMA-A", "pressure", dt(8), account="A1"),
                Complaint("C-2", "DMA-B", "pressure", dt(8), account="A2"),
                Complaint("C-3", "DMA-A", "quality", dt(8), account="A3"),
            ],
        )
        by_dma = {r.dma: r for r in results}
        self.assertEqual(by_dma["DMA-A"].evidence_count, 2)
        self.assertIsNotNone(by_dma["DMA-A"].investigation_id)
        self.assertEqual(by_dma["DMA-B"].evidence_count, 1)
        self.assertIsNone(by_dma["DMA-B"].investigation_id)

    def test_categorize_free_text(self):
        self.assertEqual(
            categorize(Complaint("C", "D", "other", dt(1), issue="water looks brown / discoloured")),
            "quality",
        )
        self.assertEqual(
            categorize(Complaint("C", "D", "", dt(1), issue="very low pressure evenings")),
            "pressure",
        )

    def test_evidence_helpers(self):
        ev = build_evidence(
            [Complaint("C-1", "D", "pressure", dt(8), account="A1")],
            [LeakIndicator("D", "night_flow", 12, 10)],
            [SensorSignal("S1", "D", "acoustic", 0.9, 0.6)],
        )["D"]
        self.assertEqual(len(ev), 3)
        kinds = [e.kind for e in ev]
        self.assertEqual(kinds, ["complaint", "dma_leak", "sensor"])

    def test_configurable_threshold(self):
        q = AquaFlowService().triage_queue(TriageConfig(evidence_threshold=3))
        r = q.ingest(
            complaints=[
                Complaint("C-1", "DMA-T", "pressure", dt(8), account="A1"),
            ],
            signals=[SensorSignal("S1", "DMA-T", "acoustic", 0.9, 0.6)],
        )[0]
        self.assertEqual(r.evidence_count, 2)
        self.assertIsNone(r.investigation_id)
        self.assertEqual(r.threshold, 3)


if __name__ == "__main__":
    unittest.main()
