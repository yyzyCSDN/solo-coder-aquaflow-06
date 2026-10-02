"""Complaint triage queue linking customer reports to leakage evidence.

Customer pressure/water-quality complaints are grouped by network zone (DMA)
and cross-checked against mutually independent corroborators: distinct
complainants, breaching DMA leakage metrics and strong sensor signals.  Once
the evidence count reaches a configurable threshold an investigation is
opened.  Investigations move through confirm -> assign -> defer/resume ->
close, and closing requires an explicit disposition.  Original complaint
records stay frozen and are retained on the investigation as immutable
references; duplicates are never counted twice.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime

ISSUES = ("pressure", "quality")


@dataclass(frozen=True)
class Complaint:
    complaint_id: str
    zone: str
    issue: str  # "pressure" or "quality"
    customer_id: str
    at: datetime
    detail: str = ""


@dataclass(frozen=True)
class DmaMetric:
    zone: str
    observed: float
    limit: float
    at: datetime
    kind: str = "night_flow"

    @property
    def breach(self) -> bool:
        return self.observed > self.limit


@dataclass(frozen=True)
class SensorSignal:
    sensor_id: str
    zone: str
    strength: float  # 0..1 confidence; only signals at/above the floor corroborate
    at: datetime


@dataclass(frozen=True)
class Investigation:
    investigation_id: str
    zone: str
    state: str
    opened_at: datetime
    complaint_refs: tuple[str, ...]
    version: int = 1
    assignee: str | None = None
    deferred_until: datetime | None = None
    defer_reason: str | None = None
    disposition: str | None = None
    updated_at: datetime | None = None
    events: tuple[tuple[datetime, str, str], ...] = ()


TRANSITIONS = {
    ("open", "confirm"): "confirmed",
    ("confirmed", "assign"): "assigned",
    ("confirmed", "defer"): "deferred",
    ("assigned", "defer"): "deferred",
    ("deferred", "resume"): "assigned",
    ("open", "close"): "closed",
    ("confirmed", "close"): "closed",
    ("assigned", "close"): "closed",
    ("deferred", "close"): "closed",
}

_NEXT_ACTION = {
    "open": "confirm",
    "confirmed": "assign",
    "assigned": "field_investigation",
    "deferred": "resume",
    "closed": "none",
}


def transition(inv: Investigation, action: str, at: datetime, *,
               assignee: str | None = None, deferred_until: datetime | None = None,
               defer_reason: str | None = None, disposition: str | None = None) -> Investigation:
    """Apply one lifecycle action to a frozen investigation record."""
    target = TRANSITIONS.get((inv.state, action))
    # Resume returns to whichever pre-defer stage the case had reached.
    if action == "resume":
        target = "assigned" if inv.assignee else "confirmed"
    if target is None:
        raise ValueError("invalid transition")
    if inv.updated_at is not None and at < inv.updated_at:
        raise ValueError("time reversal")
    if action == "assign" and not assignee:
        raise ValueError("assignee required")
    if action == "defer":
        if not defer_reason:
            raise ValueError("defer reason required")
        if not isinstance(deferred_until, datetime) or deferred_until <= at:
            raise ValueError("defer until must be in the future")
    if action == "close" and not (disposition or "").strip():
        raise ValueError("disposition required to close")

    detail = ""
    next_assignee, next_until, next_reason, next_disposition = (
        inv.assignee, inv.deferred_until, inv.defer_reason, inv.disposition)
    if action == "assign":
        next_assignee = assignee
    elif action == "defer":
        next_until, next_reason = deferred_until, defer_reason
        detail = defer_reason or ""
    elif action == "resume":
        next_until = next_reason = None
    elif action == "close":
        next_disposition = disposition
        detail = disposition or ""
    return replace(
        inv, state=target, version=inv.version + 1, updated_at=at,
        assignee=next_assignee, deferred_until=next_until,
        defer_reason=next_reason, disposition=next_disposition,
        events=inv.events + ((at, action, detail),))


def _render(key: tuple) -> str:
    kind = key[0]
    if kind == "complaint":
        return f"complaint:{key[1]}:{key[2]}"
    if kind == "dma":
        return f"dma:{key[1]}"
    return f"sensor:{key[1]}"


class TriageQueue:
    """Group complaints by zone and tally independent corroborators."""

    def __init__(self, evidence_threshold: int = 2, signal_floor: float = 0.5):
        if evidence_threshold < 1:
            raise ValueError("threshold")
        if not 0.0 <= signal_floor <= 1.0:
            raise ValueError("signal_floor")
        self.evidence_threshold = evidence_threshold
        self.signal_floor = signal_floor
        self._complaints: dict[tuple[str, str, str], Complaint] = {}
        self._complaint_ids: dict[str, Complaint] = {}
        self._zone_complaints: dict[str, list[Complaint]] = defaultdict(list)
        self._metrics: dict[str, list[DmaMetric]] = defaultdict(list)
        self._signals: dict[str, list[SensorSignal]] = defaultdict(list)
        self._investigations: dict[str, Investigation] = {}
        self._active: dict[str, str] = {}
        self._last: dict[str, str] = {}
        self._closed_at: dict[str, datetime] = {}
        self._seq = 0

    # ---- evidence intake -------------------------------------------------

    def add_complaint(self, c: Complaint) -> dict:
        if c.issue not in ISSUES:
            raise ValueError("unknown complaint issue")
        if not c.complaint_id or not c.zone or not c.customer_id:
            raise ValueError("complaint identity required")
        if c.complaint_id in self._complaint_ids and \
                self._complaint_ids[c.complaint_id] != c:
            raise ValueError("complaint id already used")
        key = (c.zone, c.issue, c.customer_id)
        if key in self._complaints:
            view = self._view(c.zone)
            view["duplicate"] = True
            view["original_complaint_id"] = self._complaints[key].complaint_id
            return view

        self._complaints[key] = c
        self._complaint_ids[c.complaint_id] = c
        self._zone_complaints[c.zone].append(c)

        active_id = self._active.get(c.zone)
        if active_id is not None:
            # Late complaint for an open investigation: attach the immutable
            # reference but leave the lifecycle state untouched.
            current = self._investigations[active_id]
            if c.complaint_id not in current.complaint_refs:
                current = replace(
                    current,
                    complaint_refs=current.complaint_refs + (c.complaint_id,),
                    events=current.events + ((c.at, "complaint_attached", c.complaint_id),))
                self._investigations[active_id] = current
        return self._view(c.zone)

    def add_dma_metric(self, m: DmaMetric) -> dict:
        if m.limit < 0 or m.observed < 0:
            raise ValueError("limit")
        if not m.zone:
            raise ValueError("zone required")
        self._metrics[m.zone].append(m)
        return self._view(m.zone)

    def add_signal(self, s: SensorSignal) -> dict:
        if not 0.0 <= s.strength <= 1.0:
            raise ValueError("strength")
        if not s.sensor_id or not s.zone:
            raise ValueError("signal identity required")
        self._signals[s.zone].append(s)
        return self._view(s.zone)

    # ---- corroboration ---------------------------------------------------

    def _corroborators(self, zone: str) -> set[tuple]:
        # Independent witnesses: one tally per distinct customer/issue pair.
        keys = {("complaint", c.customer_id, c.issue)
                for c in self._zone_complaints[zone]}
        # Zone-level DMA breaches: one tally per metric kind.
        keys.update(("dma", m.kind) for m in self._metrics[zone] if m.breach)
        # Sensor corroboration: one tally per distinct sensor, however strong.
        keys.update(("sensor", s.sensor_id) for s in self._signals[zone]
                    if s.strength >= self.signal_floor)
        return keys

    def _has_fresh_evidence(self, zone: str) -> bool:
        closed_at = self._closed_at.get(zone)
        if closed_at is None:
            return True
        evidence = (self._zone_complaints[zone]
                    + self._metrics[zone] + self._signals[zone])
        return any(e.at > closed_at for e in evidence)

    def _open(self, zone: str) -> Investigation:
        self._seq += 1
        investigation_id = f"INV-{zone}-{self._seq}"
        evidence_at = max(
            [e.at for e in self._zone_complaints[zone]]
            + [e.at for e in self._metrics[zone]]
            + [e.at for e in self._signals[zone]])
        refs = tuple(c.complaint_id for c in
                     sorted(self._zone_complaints[zone], key=lambda c: c.at))
        inv = Investigation(
            investigation_id=investigation_id, zone=zone, state="open",
            opened_at=evidence_at, complaint_refs=refs, updated_at=evidence_at,
            events=((evidence_at, "opened", ""),))
        self._investigations[investigation_id] = inv
        self._active[zone] = investigation_id
        self._last[zone] = investigation_id
        return inv

    def _view(self, zone: str, current: Investigation | None = None,
              explicit: bool = False) -> dict:
        keys = self._corroborators(zone)
        if current is None:
            active_id = self._active.get(zone)
            current = self._investigations[active_id] if active_id else None
            if current is None and zone in self._last:
                historical = self._investigations[self._last[zone]]
                if historical.state != "closed":
                    current = historical
            if (current is None and len(keys) >= self.evidence_threshold
                    and self._has_fresh_evidence(zone)):
                current = self._open(zone)
            elif current is not None and current.state == "closed" \
                    and not explicit and not self._has_fresh_evidence(zone):
                # Closed case is retained as history, not as the live queue view.
                current = None
        return {
            "zone": zone,
            "evidence_count": len(keys),
            "corroborators": sorted(_render(k) for k in keys),
            "investigation_id": current.investigation_id if current else None,
            "state": current.state if current else None,
            "next_action": _NEXT_ACTION.get(current.state, "wait_for_evidence")
            if current else "wait_for_evidence",
            "deferred_until": current.deferred_until if current else None,
        }

    def evaluate(self, zone: str) -> dict:
        return self._view(zone)

    # ---- investigation lifecycle ----------------------------------------

    def investigation(self, investigation_id: str) -> Investigation:
        return self._investigations[investigation_id]

    def complaints(self, investigation_id: str) -> tuple[Complaint, ...]:
        """Original (frozen) complaints referenced by an investigation."""
        inv = self.investigation(investigation_id)
        return tuple(self._complaint_ids[r] for r in inv.complaint_refs)

    def snapshot(self, investigation_id: str) -> dict:
        inv = self.investigation(investigation_id)
        return self._view(inv.zone, current=inv, explicit=True)

    def _move(self, investigation_id: str, action: str, at: datetime, **kw) -> dict:
        inv = self._investigations[investigation_id]
        updated = transition(inv, action, at, **kw)
        self._investigations[investigation_id] = updated
        if updated.state == "closed":
            self._active.pop(inv.zone, None)
            self._closed_at[inv.zone] = at
        return self.snapshot(investigation_id)

    def confirm(self, investigation_id: str, at: datetime) -> dict:
        return self._move(investigation_id, "confirm", at)

    def assign(self, investigation_id: str, assignee: str, at: datetime) -> dict:
        return self._move(investigation_id, "assign", at, assignee=assignee)

    def defer(self, investigation_id: str, until: datetime,
              reason: str, at: datetime) -> dict:
        return self._move(investigation_id, "defer", at,
                          deferred_until=until, defer_reason=reason)

    def resume(self, investigation_id: str, at: datetime) -> dict:
        return self._move(investigation_id, "resume", at)

    def close(self, investigation_id: str, disposition: str, at: datetime) -> dict:
        return self._move(investigation_id, "close", at, disposition=disposition)
