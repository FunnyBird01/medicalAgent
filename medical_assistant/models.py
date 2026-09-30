"""Small, serialisable domain models shared by storage, tools and GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class Source:
    title: str
    text: str
    score: float
    path: str = ""
    chunk_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "text": self.text, "score": round(self.score, 4), "path": self.path, "chunk_id": self.chunk_id}


@dataclass(slots=True)
class SearchResult:
    title: str
    text: str
    score: float
    path: str = ""
    chunk_id: str = ""

    def as_source(self) -> Source:
        return Source(self.title, self.text, self.score, self.path, self.chunk_id)


@dataclass(slots=True)
class InventoryItem:
    id: int | None
    department: str
    drug_name: str
    specification: str
    quantity: int
    reorder_level: int
    unit: str = "盒"
    updated_at: str = field(default_factory=now_iso)

    @property
    def status(self) -> str:
        if self.quantity <= 0:
            return "缺货"
        if self.quantity <= self.reorder_level:
            return "紧缺"
        return "正常"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "department": self.department, "drug_name": self.drug_name, "specification": self.specification,
                "quantity": self.quantity, "reorder_level": self.reorder_level, "unit": self.unit, "updated_at": self.updated_at, "status": self.status}


@dataclass(slots=True)
class AgentEvent:
    node: str
    event_type: str
    detail: str
    timestamp: str = field(default_factory=now_iso)

    def as_dict(self) -> dict[str, str]:
        return {"node": self.node, "event_type": self.event_type, "detail": self.detail, "timestamp": self.timestamp}


@dataclass(slots=True)
class AgentResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)


@dataclass(slots=True)
class Prescription:
    id: int | None
    case_id: str
    drug_name: str
    specification: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    quantity: str = ""
    route: str = "口服"
    prescribing_doctor: str = ""
    notes: str = ""
    prescribed_at: str = field(default_factory=now_iso)
    created_at: str = field(default_factory=now_iso)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "case_id": self.case_id, "drug_name": self.drug_name,
                "specification": self.specification, "dosage": self.dosage,
                "frequency": self.frequency, "duration": self.duration,
                "quantity": self.quantity, "route": self.route,
                "prescribing_doctor": self.prescribing_doctor, "notes": self.notes,
                "prescribed_at": self.prescribed_at, "created_at": self.created_at}

