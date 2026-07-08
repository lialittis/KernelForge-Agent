from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kernel_forge.submission import display_path


ROOT = Path(__file__).resolve().parents[2]


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderRequest:
    case_id: str
    candidate_index: int
    pass_n: int
    backend: str
    prompt_version: str
    prompt: str
    opspec: dict[str, Any]
    sketch: dict[str, Any]
    retrieved_skills: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_text(self, request: ProviderRequest) -> ProviderResponse:
        ...


class ReplayProvider:
    provider_name = "replay"
    model_name = "replay-v1"

    def __init__(self, repo_root: str | Path | None = None):
        self.repo_root = Path(repo_root) if repo_root is not None else ROOT
        self._templates = {
            "t1/sigmoid_scale_sum": {
                1: "kernel_forge/candidates/sigmoid_scale_sum_v1.py",
                2: "kernel_forge/candidates/sigmoid_scale_sum_v2.py",
                3: "kernel_forge/candidates/sigmoid_scale_sum_v3.py",
                4: "kernel_forge/candidates/sigmoid_scale_sum_v4.py",
            }
        }

    def generate_text(self, request: ProviderRequest) -> ProviderResponse:
        source = self._template_path(request.case_id, request.candidate_index)
        source_path = self.repo_root / source
        text = source_path.read_text(encoding="utf-8")
        return ProviderResponse(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            metadata={
                "replay_source_path": display_path(source_path, self.repo_root),
                "case_id": request.case_id,
                "candidate_index": request.candidate_index,
            },
        )

    def _template_path(self, case_id: str, candidate_index: int) -> str:
        case_templates = self._templates.get(case_id)
        if case_templates is None or candidate_index not in case_templates:
            raise ProviderError(
                f"Replay provider has no template for case_id={case_id!r}, "
                f"candidate_index={candidate_index}"
            )
        return case_templates[candidate_index]


def load_provider(name: str, *, repo_root: str | Path | None = None) -> TextProvider:
    if name == "replay":
        return ReplayProvider(repo_root=repo_root)
    raise ProviderError(f"Unknown provider: {name}")
