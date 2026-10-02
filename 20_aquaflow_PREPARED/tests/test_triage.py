import unittest
from datetime import datetime, timedelta

from aquaflow.customers.triage import (
    Complaint, DmaMetric, SensorSignal, TriageQueue,
)
from aquaflow.api import AquaFlowService

T0 = datetime(2026, 10, 2, 9, 0)


def at(m):
    return T0 + timedelta(minutes=m)


class TriageTests(unittest.TestCase):
    def test_duplicate_complaints_count_as_one(self):
        q = TriageQueue()
        first = q.add_complaint(Complaint("C1", "DMA-7", "pressure", "acct-1", at(0)))
        self.assertIsNone(first["investigation_id"])
        self.assertEqual(first["evidence_count"], 1)

        # Same zone/issue/customer repeated: duplicate, no new corroborator.
        dup = q.add_complaint(Complaint("C2", "DMA-7", "pressure", "acct-1", at(5)))
        self.assertTrue(dup["duplicate"])
        self.assertEqual(dup["original_complaint_id"], "C1")
        self.assertEqual(dup["evidence_count"], 1)
        self.assertIsNone(dup["investigation_id"])

        # A different customer plus a DMA breach crosses the default threshold.
        view = q.add_complaint(Complaint("C3", "DMA-7", "pressure", "acct-2", at(10)))
        self.assertEqual(view["evidence_count"], 2)
        inv_id = view["investigation_id"]
        self.assertIsNotNone(inv_id)
        self.assertEqual(view["state"], "open")
        self.assertEqual(view["next_action"], "confirm")
        self.assertEqual(view["zone"], "DMA-7")

        # Only the two distinct complaints are retained as immutable references.
        refs = q.complaints(inv_id)
        self.assertEqual([c.complaint_id for c in refs], ["C1", "C3"])
        with self.assertRaises(Exception):
            refs[0].detail = "tampered"

    def test_single_strong_sensor_without_cross_corroboration(self):
        q = TriageQueue()
        q.add_signal(SensorSignal("S-99", "DMA-3", 0.99, at(0)))
        view = q.evaluate("DMA-3")
        self.assertEqual(view["evidence_count"], 1)
        self.assertEqual(view["corroborators"], ["sensor:S-99"])
        self.assertIsNone(view["investigation_id"])
        self.assertEqual(view["state"], None)
        self.assertEqual(view["next_action"], "wait_for_evidence")

        # Weak signals are ignored entirely; a breaching DMA metric finally
        # provides the independent cross-corroboration needed to open.
        q.add_signal(SensorSignal("S-100", "DMA-3", 0.1, at(1)))
        q.add_dma_metric(DmaMetric("DMA-3", 42.0, 30.0, at(2)))
        view = q.evaluate("DMA-3")
        self.assertEqual(view["evidence_count"], 2)
        self.assertIsNotNone(view["investigation_id"])

    def test_late_complaint_after_dispatch_attaches_to_investigation(self):
        q = TriageQueue()
        q.add_complaint(Complaint("C1", "DMA-9", "quality", "acct-1", at(0)))
        q.add_dma_metric(DmaMetric("DMA-9", 50.0, 20.0, at(1)))
        inv_id = q.evaluate("DMA-9")["investigation_id"]
        q.confirm(inv_id, at(10))
        q.assign(inv_id, "crew-42", at(20))

        # Complaint arrives after dispatch: same zone, different customer.
        late = q.add_complaint(Complaint("C2", "DMA-9", "quality", "acct-2", at(30)))
        self.assertEqual(late["investigation_id"], inv_id)
        self.assertEqual(late["state"], "assigned")
        self.assertEqual(late["next_action"], "field_investigation")
        self.assertEqual(late["evidence_count"], 3)

        inv = q.investigation(inv_id)
        self.assertEqual(inv.complaint_refs, ("C1", "C2"))
        self.assertEqual(inv.assignee, "crew-42")
        self.assertEqual(inv.version, 3)  # confirm + assign; attach is not a state move
        actions = [e[1] for e in inv.events]
        self.assertIn("complaint_attached", actions)
        self.assertEqual(q.complaints(inv_id)[1].complaint_id, "C2")

    def test_close_without_disposition_rejected(self):
        q = TriageQueue()
        q.add_complaint(Complaint("C1", "DMA-1", "pressure", "acct-1", at(0)))
        q.add_signal(SensorSignal("S1", "DMA-1", 0.8, at(1)))
        inv_id = q.evaluate("DMA-1")["investigation_id"]
        with self.assertRaises(ValueError):
            q.close(inv_id, "", at(5))
        with self.assertRaises(ValueError):
            q.close(inv_id, "   ", at(5))
        self.assertEqual(q.investigation(inv_id).state, "open")

        closed = q.close(inv_id, "burst repaired; service restored", at(30))
        self.assertEqual(closed["state"], "closed")
        self.assertEqual(closed["next_action"], "none")

    def test_defer_and_resume(self):
        q = TriageQueue()
        q.add_complaint(Complaint("C1", "DMA-2", "pressure", "acct-1", at(0)))
        q.add_dma_metric(DmaMetric("DMA-2", 31.0, 30.0, at(1)))
        inv_id = q.evaluate("DMA-2")["investigation_id"]
        q.confirm(inv_id, at(10))
        with self.assertRaises(ValueError):
            q.defer(inv_id, at(0), "", at(20))  # no reason / past until
        deferred = q.defer(inv_id, at(60), "await night-flow batch", at(20))
        self.assertEqual(deferred["state"], "deferred")
        self.assertEqual(deferred["next_action"], "resume")
        resumed = q.resume(inv_id, at(40))
        # Never assigned, so resume returns to confirmed-stage pending assignment.
        self.assertEqual(resumed["state"], "confirmed")

    def test_configurable_threshold(self):
        q = TriageQueue(evidence_threshold=3)
        q.add_complaint(Complaint("C1", "DMA-5", "pressure", "acct-1", at(0)))
        q.add_signal(SensorSignal("S1", "DMA-5", 0.9, at(1)))
        self.assertIsNone(q.evaluate("DMA-5")["investigation_id"])
        q.add_dma_metric(DmaMetric("DMA-5", 60.0, 20.0, at(2)))
        view = q.evaluate("DMA-5")
        self.assertEqual(view["evidence_count"], 3)
        self.assertIsNotNone(view["investigation_id"])

    def test_fresh_evidence_required_after_closure(self):
        q = TriageQueue()
        q.add_complaint(Complaint("C1", "DMA-4", "pressure", "acct-1", at(0)))
        q.add_signal(SensorSignal("S1", "DMA-4", 0.7, at(1)))
        inv_id = q.evaluate("DMA-4")["investigation_id"]
        q.close(inv_id, "closed as isolated valve fault", at(90))

        # Old evidence alone cannot immediately reopen a new investigation.
        self.assertIsNone(q.evaluate("DMA-4")["investigation_id"])
        reopened = q.add_signal(SensorSignal("S2", "DMA-4", 0.8, at(120)))
        self.assertIsNotNone(reopened["investigation_id"])
        self.assertNotEqual(reopened["investigation_id"], inv_id)

    def test_service_entrypoint(self):
        q = AquaFlowService().triage_queue()
        self.assertEqual(q.evidence_threshold, 2)
        self.assertEqual(q.signal_floor, 0.5)


if __name__ == "__main__":
    unittest.main()
