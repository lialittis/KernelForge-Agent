from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    layout: str = "contiguous"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpSpec:
    id: str
    name: str
    tier: str
    category: str
    source_path: str
    reference_class: str = "Model"
    candidate_class: str = "ModelNew"
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    semantics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    sketch: dict[str, Any] = field(default_factory=dict)
    submission: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
            "category": self.category,
            "source_path": self.source_path,
            "reference_class": self.reference_class,
            "candidate_class": self.candidate_class,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "semantics": self.semantics,
            "validation": self.validation,
            "performance": self.performance,
            "sketch": self.sketch,
            "submission": self.submission,
        }


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

