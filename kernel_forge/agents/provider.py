from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

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


class OpenAIResponsesProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        responses_url: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        http_post: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model or os.environ.get("KERNEL_FORGE_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL")
        self.responses_url = responses_url or os.environ.get(
            "KERNEL_FORGE_OPENAI_RESPONSES_URL",
            "https://api.openai.com/v1/responses",
        )
        self.timeout = timeout if timeout is not None else float(os.environ.get("KERNEL_FORGE_OPENAI_TIMEOUT", "120"))
        self.max_output_tokens = max_output_tokens or int(os.environ.get("KERNEL_FORGE_OPENAI_MAX_OUTPUT_TOKENS", "8192"))
        self.temperature = temperature
        if self.temperature is None and "KERNEL_FORGE_OPENAI_TEMPERATURE" in os.environ:
            self.temperature = float(os.environ["KERNEL_FORGE_OPENAI_TEMPERATURE"])
        self._http_post = http_post or _post_json

        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for provider=openai")
        if not self.model_name:
            raise ProviderError("KERNEL_FORGE_OPENAI_MODEL or OPENAI_MODEL is required for provider=openai")

    def generate_text(self, request: ProviderRequest) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": (
                "You are the Code Agent for SketchSkill-AKG. Return only the Python "
                "candidate source file. Do not include Markdown fences or explanation."
            ),
            "input": request.prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._http_post(self.responses_url, headers, payload, self.timeout)
        text = _strip_code_fence(_extract_response_text(response))
        return ProviderResponse(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            metadata={
                "response_id": response.get("id"),
                "responses_url": self.responses_url,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "usage": response.get("usage"),
            },
        )


def load_provider(name: str, *, repo_root: str | Path | None = None) -> TextProvider:
    if name == "replay":
        return ReplayProvider(repo_root=repo_root)
    if name == "openai":
        return OpenAIResponsesProvider()
    raise ProviderError(f"Unknown provider: {name}")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"OpenAI Responses API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"OpenAI Responses API request failed: {exc.reason}") from exc

    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ProviderError("OpenAI Responses API returned a non-object JSON response")
    return parsed


def _extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    if chunks:
        return "\n".join(chunks)
    raise ProviderError("OpenAI Responses API response did not contain output text")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip() + "\n"
    return text
