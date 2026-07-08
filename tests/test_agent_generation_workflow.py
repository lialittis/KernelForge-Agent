from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_forge.agents.prompts import render_code_prompt
from kernel_forge.agents.provider import OpenAIResponsesProvider, ProviderError, ProviderRequest, ReplayProvider
from kernel_forge.agents.skills import load_retrieved_skills, retrieve_skill_paths
from kernel_forge.agents.workflow import generate_passn_candidates
from kernel_forge.benchmark import read_yaml
from kernel_forge.submission import CaseMapping, create_submission_layout


OPSPEC_PATH = ROOT / "benchmarks/parsed/t1_sigmoid_scale_sum.yaml"


def test_replay_provider_returns_known_sigmoid_candidate():
    opspec = read_yaml(OPSPEC_PATH)
    request = ProviderRequest(
        case_id="t1/sigmoid_scale_sum",
        candidate_index=2,
        pass_n=4,
        backend="triton_ascend",
        prompt_version="code_agent.v1",
        prompt="prompt",
        opspec=opspec,
        sketch=opspec["sketch"],
        retrieved_skills=["skills/reduction/SKILL.md"],
    )

    response = ReplayProvider(repo_root=ROOT).generate_text(request)

    assert response.provider == "replay"
    assert response.model == "replay-v1"
    assert response.metadata["replay_source_path"] == "kernel_forge/candidates/sigmoid_scale_sum_v2.py"
    assert "class ModelNew" in response.text
    assert "triton_row_reduce_bs8192" in response.text


def test_openai_provider_requires_api_key_and_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KERNEL_FORGE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIResponsesProvider()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ProviderError, match="KERNEL_FORGE_OPENAI_MODEL"):
        OpenAIResponsesProvider()


def test_openai_provider_posts_responses_request_and_extracts_code():
    opspec = read_yaml(OPSPEC_PATH)
    request = ProviderRequest(
        case_id="t1/sigmoid_scale_sum",
        candidate_index=1,
        pass_n=4,
        backend="triton_ascend",
        prompt_version="code_agent.v1",
        prompt="Generate a candidate.",
        opspec=opspec,
        sketch=opspec["sketch"],
        retrieved_skills=["skills/reduction/SKILL.md"],
    )
    calls = []

    def fake_post(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {
            "id": "resp_test",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "```python\nclass ModelNew:\n    pass\n```",
                        }
                    ]
                }
            ],
            "usage": {"total_tokens": 123},
        }

    provider = OpenAIResponsesProvider(
        api_key="test-key",
        model="test-model",
        responses_url="https://example.test/v1/responses",
        timeout=3.0,
        max_output_tokens=2048,
        temperature=0.2,
        http_post=fake_post,
    )

    response = provider.generate_text(request)

    assert response.provider == "openai"
    assert response.model == "test-model"
    assert response.text == "class ModelNew:\n    pass\n"
    assert response.metadata["response_id"] == "resp_test"
    assert response.metadata["usage"] == {"total_tokens": 123}

    assert len(calls) == 1
    url, headers, payload, timeout = calls[0]
    assert url == "https://example.test/v1/responses"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert payload["model"] == "test-model"
    assert payload["input"] == "Generate a candidate."
    assert payload["max_output_tokens"] == 2048
    assert payload["temperature"] == 0.2
    assert "Return only the Python candidate source file" in payload["instructions"]
    assert timeout == 3.0


def test_openai_provider_accepts_top_level_output_text():
    request = ProviderRequest(
        case_id="case",
        candidate_index=1,
        pass_n=1,
        backend="triton_ascend",
        prompt_version="code_agent.v1",
        prompt="prompt",
        opspec={},
        sketch={},
        retrieved_skills=[],
    )

    provider = OpenAIResponsesProvider(
        api_key="test-key",
        model="test-model",
        http_post=lambda *_args: {"id": "resp_test", "output_text": "class ModelNew:\n    pass\n"},
    )

    response = provider.generate_text(request)

    assert response.text == "class ModelNew:\n    pass\n"


def test_skill_retrieval_and_prompt_rendering_include_required_context():
    opspec = read_yaml(OPSPEC_PATH)
    skill_paths = retrieve_skill_paths(opspec)
    skills = load_retrieved_skills(skill_paths, repo_root=ROOT)

    prompt, prompt_version = render_code_prompt(
        opspec=opspec,
        backend="triton_ascend",
        candidate_index=2,
        pass_n=4,
        skills=skills,
        repo_root=ROOT,
    )

    assert prompt_version == "code_agent.v1"
    assert "Case: t1/sigmoid_scale_sum" in prompt
    assert "Backend: triton_ascend" in prompt
    assert "Candidate: 2 / 4" in prompt
    assert "sigmoid_scale_sum_row_reduction" in prompt
    assert "skills/reduction/SKILL.md" in prompt
    assert "skills/broadcast/SKILL.md" in prompt


def test_submission_helper_preserves_flat_and_nested_layout(tmp_path):
    source = "kernel_forge/candidates/sigmoid_scale_sum_v1.py"
    nested_root = create_submission_layout(
        team="nested_team",
        candidate="nested_candidate",
        cases=[CaseMapping("t1/sigmoid_scale_sum", source)],
        output_root=tmp_path / "nested",
        layout="nested",
        repo_root=ROOT,
    )
    flat_root = create_submission_layout(
        team="flat_team",
        candidate="flat_candidate",
        cases=[CaseMapping("t1/sigmoid_scale_sum", source)],
        output_root=tmp_path / "flat",
        layout="flat",
        repo_root=ROOT,
    )

    assert nested_root == tmp_path / "nested/nested_team/nested_team"
    assert flat_root == tmp_path / "flat/flat_team"
    assert (nested_root / "t1/sigmoid_scale_sum.py").read_text() == (ROOT / source).read_text()
    assert json.loads((flat_root / "meta.json").read_text())["candidate"] == "flat_candidate"


def test_generate_passn_candidates_replay_workflow(tmp_path):
    generated = generate_passn_candidates(
        opspec_path=OPSPEC_PATH,
        provider_name="replay",
        backend="triton_ascend",
        pass_n=4,
        run_id="test-replay-sigmoid",
        output_root=tmp_path,
        repo_root=ROOT,
    )

    experiment = yaml.safe_load((ROOT / generated.experiment_path).read_text())
    assert generated.run_id == "test-replay-sigmoid"
    assert len(generated.candidates) == 4
    assert experiment["generation"]["provider"] == "replay"
    assert experiment["generation"]["model"] == "replay-v1"
    assert experiment["generation"]["prompt_version"] == "code_agent.v1"
    assert experiment["generation"]["pass_n"] == 4
    assert experiment["generation"]["candidates"][1]["team_name"] == "sigmoid_scale_sum_replay_v2"
    assert "skills/reduction/SKILL.md" in experiment["generation"]["retrieved_skills"]

    for candidate in generated.candidates:
        candidate_path = ROOT / candidate.candidate_path
        submission_case = ROOT / candidate.submission_root / "t1/sigmoid_scale_sum.py"
        assert candidate_path.exists()
        assert submission_case.exists()
        assert submission_case.read_text() == candidate_path.read_text()


def test_generate_candidate_cli_writes_summary_and_experiment(tmp_path):
    output_root = tmp_path / "generated"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_candidate.py"),
            "--opspec",
            "benchmarks/parsed/t1_sigmoid_scale_sum.yaml",
            "--provider",
            "replay",
            "--backend",
            "triton_ascend",
            "--pass-n",
            "4",
            "--run-id",
            "cli-replay-sigmoid",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["run_id"] == "cli-replay-sigmoid"
    assert len(payload["candidates"]) == 4
    experiment = yaml.safe_load((ROOT / payload["experiment_path"]).read_text())
    assert experiment["benchmark"]["task_id"] == "t1/sigmoid_scale_sum"
    assert experiment["generation"]["provider"] == "replay"
    assert experiment["artifacts"]["submissions"].endswith("cli-replay-sigmoid/submissions")
