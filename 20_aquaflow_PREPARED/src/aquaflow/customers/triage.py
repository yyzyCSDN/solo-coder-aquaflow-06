"""Complaint triage queue correlating customer reports with DMA leakage evidence.

Customer complaints about pressure or water quality are grouped by network area
(DMA) and corroborated against DMA leakage indicators and independent sensor
signals (acoustic / pressure-drop).  When the number of *mutually independent*
pieces of evidence reaches a configurable threshold an investigation is opened.

Evidence independence (each source counts once, they never stack):
  * one bucket per distinct complaining customer and category — a repeat report
    from the same customer and category is flagged ``is_duplicate`` and does not
    raise the count;
  * a breached DMA leakage indicator contributes at most one bucket;
  * each distinct active sensor contributes one bucket.

An investigation is only opened for a DMA that has at least one complaint; a
strong sensor signal on its own, without cross-corroboration, never suffices.

Original complaints are retained in an immutable registry; investigations hold
references by id and never mutate or delete the originals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

# --- inputs -----------------------------------------------------------------

PRESSURE_KEYWORDS = ("pressure", "low pressure", "surge", "no water")
QUALITY_KEYWORDS = ("quality", "discolour", "discolor", "turbid", "taste", "odour", "odor")


@dataclass(frozen=True)
class Complaint:
    complaint_id: str
    dma: str
    category: str            # "pressure" | "quality" | free text normalized via issue
    received_at: datetime
    account: str = ""
    issue: str = ""          # free-text description used when category is vague
    location: str = ""


@dataclass(frozen=True)
class LeakIndicator:
    dma: str
    metric: str              # e.g. "night_flow", "m3_h"
    value: float
    threshold: float
    at: datetime | None = None

    @property
    def breached(self) -> bool:
        return self.value > self.threshold


@dataclass(frozen=True)
class SensorSignal:
    sensor_id: str
    dma: str
    kind: str                # "acoustic" | "pressure_drop" | ...
    strength: float
    threshold: float
    at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.strength >= self.threshold


@dataclass(frozen=True)
class TriageConfig:
    evidence_threshold: int = 2
    late_window_hours: float = 24.0


# --- derived evidence -------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """One independent corroboration bucket."""
    bucket: str               # unique identity within a DMA
    kind: str                 # "complaint" | "dma_leak" | "sensor"
    detail: str = ""
    complaint_ids: tuple[str, ...] = ()


def categorize(complaint: Complaint) -> str:
    raw = (complaint.category or "").strip().lower()
    if raw in {"pressure", "quality"}:
        return raw
    text = " ".join([raw, (complaint.issue or "").lower()])
    if any(k in text for k in QUALITY_KEYWORDS):
        return "quality"
    if any(k in text for k in PRESSURE_KEYWORDS):
        return "pressure"
    return raw or "unknown"


def build_evidence(
    complaints: list[Complaint],
    indicators: list[LeakIndicator] | None = None,
    signals: list[SensorSignal] | None = None,
) -> dict[str, list[Evidence]]:
    """Group evidence by DMA into independent, deterministically ordered buckets."""
    by_dma: dict[str, list[Complaint]] = {}
    for c in complaints:
        by_dma.setdefault(c.dma, []).append(c)
    indicators_by_dma: dict[str, list[LeakIndicator]] = {}
    for ind in indicators or []:
        indicators_by_dma.setdefault(ind.dma, []).append(ind)
    signals_by_dma: dict[str, list[SensorSignal]] = {}
    for sig in signals or []:
        signals_by_dma.setdefault(sig.dma, []).append(sig)

    out: dict[str, list[Evidence]] = {}
    for dma in sorted(set(by_dma) | set(indicators_by_dma) | set(signals_by_dma)):
        buckets: list[Evidence] = []
        # one independent witness per (category, account-or-complaint-id)
        witnesses: dict[tuple[str, str], list[Complaint]] = {}
        for c in by_dma.get(dma, []):
            key = (categorize(c), c.account or c.complaint_id)
            witnesses.setdefault(key, []).append(c)
        for (category, _acct), rows in sorted(witnesses.items()):
            rows = sorted(rows, key=lambda c: c.received_at)
            buckets.append(Evidence(
                bucket=f"complaint:{category}:{rows[0].account or rows[0].complaint_id}",
                kind="complaint",
                detail=f"{category} report from customer {rows[0].account or rows[0].complaint_id} ({len(rows)} call(s))",
                complaint_ids=tuple(c.complaint_id for c in rows),
            ))
        breached = [i for i in indicators_by_dma.get(dma, []) if i.breached]
        if breached:
            top = max(breached, key=lambda i: i.value)
            buckets.append(Evidence(
                bucket="dma_leak",
                kind="dma_leak",
                detail=f"{top.metric} {top.value:g} exceeds threshold {top.threshold:g}",
            ))
        by_sensor: dict[str, SensorSignal] = {}
        for sig in signals_by_dma.get(dma, []):
            if sig.active and sig.sensor_id not in by_sensor:
                by_sensor[sig.sensor_id] = sig
        for sensor_id in sorted(by_sensor):
            sig = by_sensor[sensor_id]
            buckets.append(Evidence(
                bucket=f"sensor:{sensor_id}",
                kind="sensor",
                detail=f"{sig.kind} sensor {sensor_id} strength {sig.strength:g} >= {sig.threshold:g}",
            ))
        out[dma] = buckets
    return out


def evidence_count(evidence: list[Evidence]) -> int:
    return len(evidence)


# --- investigation lifecycle ------------------------------------------------

TRANSITIONS = {
    ("open", "confirm"): "confirmed",
    ("open", "assign"): "assigned",
    ("open", "defer"): "deferred",
    ("confirmed", "assign"): "assigned",
    ("confirmed", "defer"): "deferred",
    ("assigned", "confirm"): "confirmed",
    ("assigned", "defer"): "deferred",
    ("deferred", "confirm"): "confirmed",
    ("deferred", "assign"): "assigned",
}

ACTIVE_STATES = {"assigned", "confirmed", "deferred"}


@dataclass
class Investigation:
    investigation_id: str
    dma: str
    opened_at: datetime
    evidence: list[Evidence]
    complaint_refs: tuple[str, ...]          # immutable references to originals
    status: str = "open"
    assignee: str | None = None
    disposition: str | None = None           # resolution conclusion required to close
    history: list[tuple[str, datetime, str]] = field(default_factory=list)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    def affected_area(self) -> str:
        return self.dma


def apply(inv: Investigation, action: str, at: datetime, *, assignee: str | None = None,
          disposition: str | None = None, note: str = "") -> Investigation:
    """Apply confirm / assign / defer / close.

    Closing without a disposition (处置结论) is rejected from any state.
    """
    if action == "close":
        if inv.status == "closed":
            raise ValueError("invalid transition")
        if not (disposition or inv.disposition):
            raise ValueError("closing requires a disposition")
        inv.status = "closed"
        inv.disposition = disposition or inv.disposition
        if assignee is not None:
            inv.assignee = assignee
        inv.history.append(("close", at, note or inv.disposition))
        return inv
    target = TRANSITIONS.get((inv.status, action))
    if target is None:
        raise ValueError("invalid transition")
    inv.status = target
    if action == "assign":
        if not assignee:
            raise ValueError("assign requires an assignee")
        inv.assignee = assignee
    if disposition:
        inv.disposition = disposition
    inv.history.append((action, at, note or assignee or disposition or ""))
    return inv


def next_action(inv: Investigation) -> str:
    match inv.status:
        case "open":
            return "confirm_or_assign"
        case "confirmed":
            return "assign"
        case "assigned":
            return "investigate_on_site"
        case "deferred":
            return "resume"
        case "closed":
            return "none"
    return "none"


# --- triage result -----------------------------------------------------------

@dataclass(frozen=True)
class TriageResult:
    dma: str
    evidence_count: int
    affected_area: str
    investigation_id: str | None
    next_action: str
    complaint_ids: tuple[str, ...]
    is_duplicate: bool = False
    is_late: bool = False
    threshold: int = 2


# --- triage queue ------------------------------------------------------------

class ComplaintTriageQueue:
    """Stateful queue: ingest complaints/evidence, open and manage investigations."""

    def __init__(self, config: TriageConfig | None = None):
        self.config = config or TriageConfig()
        self._complaints: dict[str, Complaint] = {}
        self._indicators: list[LeakIndicator] = []
        self._signals: list[SensorSignal] = []
        self._investigations: dict[str, Investigation] = {}     # dma -> current
        self._seq = 0

    # immutable complaint registry -- originals can be read, never replaced
    def complaint(self, complaint_id: str) -> Complaint:
        return self._complaints[complaint_id]

    def complaints(self, dma: str | None = None) -> list[Complaint]:
        rows = list(self._complaints.values())
        if dma is not None:
            rows = [c for c in rows if c.dma == dma]
        return sorted(rows, key=lambda c: (c.received_at, c.complaint_id))

    def investigation(self, dma: str) -> Investigation | None:
        return self._investigations.get(dma)

    def _new_id(self, dma: str) -> str:
        self._seq += 1
        return f"INV-{dma}-{self._seq:04d}"

    def ingest(
        self,
        complaints: list[Complaint] | None = None,
        indicators: list[LeakIndicator] | None = None,
        signals: list[SensorSignal] | None = None,
        at: datetime | None = None,
    ) -> list[TriageResult]:
        touched: set[str] = set()
        duplicates: set[str] = set()
        new_by_dma: dict[str, set[str]] = {}
        for c in complaints or []:
            if c.complaint_id in self._complaints:
                raise ValueError("duplicate complaint id")
            category = categorize(c)
            is_dup = bool(c.account) and any(
                x.dma == c.dma and x.account == c.account and categorize(x) == category
                for x in self._complaints.values()
            )
            if is_dup:
                duplicates.add(c.complaint_id)
            self._complaints[c.complaint_id] = c
            new_by_dma.setdefault(c.dma, set()).add(c.complaint_id)
            touched.add(c.dma)
        for ind in indicators or []:
            self._indicators.append(ind)
            touched.add(ind.dma)
        for sig in signals or []:
            self._signals.append(sig)
            touched.add(sig.dma)
        return [
            self._evaluate_dma(dma, at, duplicates, new_by_dma.get(dma, set()))
            for dma in sorted(touched)
        ]

    def _dma_evidence(self, dma: str) -> list[Evidence]:
        return build_evidence(
            self.complaints(dma),
            [i for i in self._indicators if i.dma == dma],
            [s for s in self._signals if s.dma == dma],
        ).get(dma, [])

    def _evaluate_dma(self, dma: str, now: datetime | None, duplicates: set[str],
                      new_ids: set[str]) -> TriageResult:
        evidence = self._dma_evidence(dma)
        dma_complaints = self.complaints(dma)
        if now is None:
            stamps = [c.received_at for c in dma_complaints]
            stamps += [i.at for i in self._indicators if i.dma == dma and i.at]
            stamps += [s.at for s in self._signals if s.dma == dma and s.at]
            now = max(stamps) if stamps else datetime.now().astimezone()

        inv = self._investigations.get(dma)
        is_late = False
        if inv is not None:
            # a complaint arriving after the case was assigned/confirmed/deferred is late
            is_late = inv.status in ACTIVE_STATES and bool(new_ids)
            # late or follow-up evidence is merged into the live investigation,
            # but original complaint records stay untouched
            inv.evidence = evidence
            inv.complaint_refs = tuple(sorted(
                set(inv.complaint_refs) | {c.complaint_id for c in dma_complaints}
            ))
            if is_late:
                inv.history.append((
                    "late_complaint", now,
                    f"late complaint(s) {sorted(new_ids)} attached; evidence={len(evidence)}",
                ))
        elif dma_complaints and len(evidence) >= self.config.evidence_threshold:
            inv = Investigation(
                investigation_id=self._new_id(dma),
                dma=dma,
                opened_at=now,
                evidence=list(evidence),
                complaint_refs=tuple(c.complaint_id for c in dma_complaints),
                history=[("open", now, f"{len(evidence)} independent evidence buckets")],
            )
            self._investigations[dma] = inv

        return TriageResult(
            dma=dma,
            evidence_count=len(evidence),
            affected_area=dma,
            investigation_id=inv.investigation_id if inv else None,
            next_action=next_action(inv) if inv else "await_corroboration",
            complaint_ids=tuple(c.complaint_id for c in dma_complaints),
            is_duplicate=bool(duplicates & new_ids),
            is_late=is_late,
            threshold=self.config.evidence_threshold,
        )

    # lifecycle passthrough ---------------------------------------------------
    def apply(self, dma: str, action: str, at: datetime, **kw) -> Investigation:
        inv = self._investigations.get(dma)
        if inv is None:
            raise ValueError("no investigation for DMA")
        return apply(inv, action, at, **kw)

    def late_window_end(self, inv: Investigation) -> datetime:
        return inv.opened_at + timedelta(hours=self.config.late_window_hours)
