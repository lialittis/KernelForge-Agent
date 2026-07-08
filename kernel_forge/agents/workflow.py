from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_forge.benchmark import read_yaml, write_yaml
from kernel_forge.submission import CaseMapping, create_submission_layout, display_path

from .prompts import render_code_prompt
from .provider import ProviderRequest, load_provider
from .skills import load_retrieved_skills, retrieve_skill_paths


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GeneratedCandidate:
    index: int
    team_name: str
    candidate_path: str
    submission_root: str
    provider: str
    model: str
    prompt_version: str
    retrieved_skills: list[str]
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedRun:
    run_id: str
    output_root: str
    experiment_path: str
    candidates: list[GeneratedCandidate]


def generate_passn_candidates(
    *,
    opspec_path: str | Path,
    provider_name: str,
    backend: str,
    pass_n: int,
    run_id: str,
    output_root: str | Path,
    repo_root: str | Path | None = None,
) -> GeneratedRun:
    root = Path(repo_root) if repo_root is not None else ROOT
    opspec = read_yaml(root / opspec_path if not Path(opspec_path).is_absolute() else opspec_path)
    case_id = str(opspec["id"])
    case_name = str(opspec["name"])
    run_root = _resolve_output_root(output_root, root) / run_id
    candidate_dir = run_root / "candidates"
    prompt_dir = run_root / "prompts"
    submission_root = run_root / "submissions"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    provider = load_provider(provider_name, repo_root=root)
    skill_paths = retrieve_skill_paths(opspec)
    skills = load_retrieved_skills(skill_paths, repo_root=root)

    generated: list[GeneratedCandidate] = []
    for index in range(1, pass_n + 1):
        team_name = f"{case_name}_replay_v{index}" if provider_name == "replay" else f"{case_name}_{provider_name}_v{index}"
        prompt, prompt_version = render_code_prompt(
            opspec=opspec,
            backend=backend,
            candidate_index=index,
            pass_n=pass_n,
            skills=skills,
            repo_root=root,
        )
        request = ProviderRequest(
            case_id=case_id,
            candidate_index=index,
            pass_n=pass_n,
            backend=backend,
            prompt_version=prompt_version,
            prompt=prompt,
            opspec=opspec,
            sketch=opspec.get("sketch", {}),
            retrieved_skills=skill_paths,
            metadata={"run_id": run_id, "team_name": team_name},
        )
        response = provider.generate_text(request)

        candidate_path = candidate_dir / f"{team_name}.py"
        prompt_path = prompt_dir / f"{team_name}.{prompt_version}.md"
        metadata_path = candidate_dir / f"{team_name}.metadata.json"
        candidate_path.write_text(response.text, encoding="utf-8")
        prompt_path.write_text(prompt, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "request": request.to_dict(),
                    "response": response.to_dict(),
                    "prompt_path": display_path(prompt_path, root),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        created_submission = create_submission_layout(
            team=team_name,
            candidate=team_name,
            cases=[CaseMapping(case_id=case_id, source=display_path(candidate_path, root))],
            output_root=submission_root,
            layout="flat",
            repo_root=root,
        )
        generated.append(
            GeneratedCandidate(
                index=index,
                team_name=team_name,
                candidate_path=display_path(candidate_path, root),
                submission_root=display_path(created_submission, root),
                provider=response.provider,
                model=response.model,
                prompt_version=prompt_version,
                retrieved_skills=skill_paths,
                provider_metadata=response.metadata,
            )
        )

    experiment_path = run_root / "experiment.yaml"
    experiment = _build_experiment(
        run_id=run_id,
        opspec_path=display_path(Path(opspec_path), root),
        opspec=opspec,
        backend=backend,
        provider_name=provider.provider_name,
        model_name=provider.model_name,
        pass_n=pass_n,
        candidates=generated,
        submission_root=display_path(submission_root, root),
    )
    write_yaml(experiment, experiment_path)
    return GeneratedRun(
        run_id=run_id,
        output_root=display_path(run_root, root),
        experiment_path=display_path(experiment_path, root),
        candidates=generated,
    )


def _build_experiment(
    *,
    run_id: str,
    opspec_path: str,
    opspec: dict[str, Any],
    backend: str,
    provider_name: str,
    model_name: str,
    pass_n: int,
    candidates: list[GeneratedCandidate],
    submission_root: str,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "date": None,
        "agent": "sketchskill-akg",
        "machine": None,
        "branch": None,
        "commit": None,
        "status": "generated",
        "benchmark": {
            "source": "akg_kernels_bench_lite",
            "source_commit": None,
            "task_id": opspec.get("id"),
            "operator_name": opspec.get("name"),
            "operator_category": opspec.get("category"),
            "reference_path": opspec.get("source_path"),
        },
        "generation": {
            "provider": provider_name,
            "model": model_name,
            "agent_role": "code_agent_replay" if provider_name == "replay" else "code_agent",
            "backend": backend,
            "prompt_version": "code_agent.v1",
            "retrieved_skills": candidates[0].retrieved_skills if candidates else [],
            "opspec_path": opspec_path,
            "sketch_path": f"{opspec_path}#sketch",
            "pass_n": pass_n,
            "candidates": [
                {
                    "index": candidate.index,
                    "team_name": candidate.team_name,
                    "candidate_path": candidate.candidate_path,
                    "submission_root": candidate.submission_root,
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "prompt_version": candidate.prompt_version,
                    "provider_metadata": candidate.provider_metadata,
                }
                for candidate in candidates
            ],
        },
        "results": {
            "correctness": {"status": "not_run"},
            "pass_n": {"pass_at_1": None, "pass_at_4": None},
            "performance": {"status": "not_run"},
        },
        "artifacts": {
            "submissions": submission_root,
            "results": None,
            "reports": None,
        },
        "notes": "Generated by the pluggable provider workflow; benchmark not run yet.",
    }


def _resolve_output_root(output_root: str | Path, repo_root: Path) -> Path:
    path = Path(output_root)
    if path.is_absolute():
        return path
    return repo_root / path
