from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .opspec import read_yaml


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "level": self.level, "message": self.message}


REQUIRED_OPSPEC_KEYS = {
    "id",
    "name",
    "tier",
    "category",
    "source_path",
    "reference_class",
    "candidate_class",
    "inputs",
    "outputs",
    "semantics",
    "validation",
    "performance",
    "sketch",
    "submission",
}

REQUIRED_TENSOR_KEYS = {"name", "shape", "dtype", "layout"}
REQUIRED_VALIDATION_KEYS = {"rtol", "atol", "shape_cases", "dtype_cases"}
REQUIRED_PERFORMANCE_KEYS = {"metric", "warmup", "repeats", "num_trials"}
REQUIRED_SKETCH_KEYS = {
    "operator_category",
    "compute_pattern",
    "parallel_axes",
    "tile_plan",
    "memory_plan",
    "pipeline_plan",
    "boundary_mask",
    "accumulation_dtype",
    "backend_target",
    "performance_knobs",
    "known_risks",
}


def validate_opspec_file(path: str | Path) -> list[ValidationIssue]:
    spec_path = Path(path)
    try:
        spec = read_yaml(spec_path)
    except Exception as exc:
        return [ValidationIssue(spec_path.as_posix(), "error", f"cannot read YAML: {exc}")]
    return validate_opspec(spec, path=spec_path.as_posix())


def validate_opspec(spec: dict[str, Any], *, path: str = "<memory>") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _require_keys(issues, path, spec, REQUIRED_OPSPEC_KEYS, "OpSpec")

    case_id = str(spec.get("id", ""))
    name = str(spec.get("name", ""))
    tier = str(spec.get("tier", ""))
    if "/" not in case_id:
        _error(issues, path, "id must use tier/name format")
    elif case_id != f"{tier}/{name}":
        _error(issues, path, f"id {case_id!r} must equal tier/name {tier}/{name!r}")
    if tier not in {"t1", "t2", "t3", "t4", "t5"}:
        _error(issues, path, f"tier must be a benchmark tier, got {tier!r}")
    if spec.get("reference_class") != "Model":
        _error(issues, path, "reference_class must be Model")
    if spec.get("candidate_class") != "ModelNew":
        _error(issues, path, "candidate_class must be ModelNew")

    inputs = spec.get("inputs")
    outputs = spec.get("outputs")
    if not isinstance(inputs, list) or not inputs:
        _error(issues, path, "inputs must be a non-empty list")
    else:
        for index, tensor in enumerate(inputs):
            _validate_tensor(issues, path, tensor, f"inputs[{index}]")
    if not isinstance(outputs, list) or not outputs:
        _error(issues, path, "outputs must be a non-empty list")
    else:
        for index, tensor in enumerate(outputs):
            _validate_tensor(issues, path, tensor, f"outputs[{index}]")

    validation = spec.get("validation")
    if not isinstance(validation, dict):
        _error(issues, path, "validation must be a mapping")
    else:
        _require_keys(issues, path, validation, REQUIRED_VALIDATION_KEYS, "validation")
        if not validation.get("shape_cases"):
            _error(issues, path, "validation.shape_cases must be non-empty")
        if not validation.get("dtype_cases"):
            _error(issues, path, "validation.dtype_cases must be non-empty")

    performance = spec.get("performance")
    if not isinstance(performance, dict):
        _error(issues, path, "performance must be a mapping")
    else:
        _require_keys(issues, path, performance, REQUIRED_PERFORMANCE_KEYS, "performance")

    submission = spec.get("submission")
    if not isinstance(submission, dict):
        _error(issues, path, "submission must be a mapping")
    else:
        if submission.get("entrypoint") != "ModelNew":
            _error(issues, path, "submission.entrypoint must be ModelNew")
        expected = [f"{tier}/{name}.py"] if tier and name else None
        if expected and submission.get("required_files") != expected:
            _error(issues, path, f"submission.required_files must be {expected}")

    sketch = spec.get("sketch")
    if not isinstance(sketch, dict):
        _error(issues, path, "sketch must be a mapping")
    else:
        issues.extend(validate_sketch(sketch, spec=spec, path=path))

    if isinstance(outputs, list) and len(outputs) > 1:
        _validate_tuple_outputs(issues, path, spec)

    return issues


def validate_sketch(
    sketch: dict[str, Any],
    *,
    spec: dict[str, Any] | None = None,
    path: str = "<memory>",
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _require_keys(issues, path, sketch, REQUIRED_SKETCH_KEYS, "sketch")

    if sketch.get("operator_category") in {None, "", "unknown"}:
        _error(issues, path, "sketch.operator_category must be non-generic")
    if sketch.get("compute_pattern") in {None, "", "unsupported_pending_classification"}:
        _error(issues, path, "sketch.compute_pattern must be non-generic")

    tile_plan = sketch.get("tile_plan")
    if not isinstance(tile_plan, dict):
        _error(issues, path, "sketch.tile_plan must be a mapping")
    else:
        if tile_plan.get("strategy") in {None, "", "manual_required"}:
            _error(issues, path, "sketch.tile_plan.strategy must be concrete")
        if not tile_plan.get("shape"):
            _error(issues, path, "sketch.tile_plan.shape must be non-empty")

    pipeline = sketch.get("pipeline_plan")
    if not isinstance(pipeline, dict) or not pipeline.get("stages"):
        _error(issues, path, "sketch.pipeline_plan.stages must be non-empty")
    if not sketch.get("parallel_axes"):
        _error(issues, path, "sketch.parallel_axes must be non-empty")
    if not sketch.get("known_risks"):
        _error(issues, path, "sketch.known_risks must be non-empty")
    if not isinstance(sketch.get("performance_knobs"), dict):
        _error(issues, path, "sketch.performance_knobs must be a mapping")

    category = sketch.get("operator_category")
    if category == "matmul_like":
        _validate_matmul_like_sketch(issues, path, sketch)
    elif category == "normalization":
        _validate_rowwise_sketch(issues, path, sketch, "normalization")
    elif category == "reduction":
        _validate_rowwise_sketch(issues, path, sketch, "reduction")

    if spec and len(spec.get("outputs", [])) > 1:
        contracts = sketch.get("output_contract")
        output_names = [item.get("name") for item in spec.get("outputs", []) if isinstance(item, dict)]
        if not isinstance(contracts, list):
            _error(issues, path, "tuple-output sketch must define output_contract")
        elif [item.get("name") for item in contracts if isinstance(item, dict)] != output_names:
            _error(issues, path, "sketch.output_contract must match OpSpec outputs")

    return issues


def validate_opspec_dir(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = sorted(root.glob("*.yaml"))
    rows = []
    all_issues: list[ValidationIssue] = []
    for file_path in files:
        issues = validate_opspec_file(file_path)
        all_issues.extend(issues)
        rows.append(
            {
                "path": file_path.as_posix(),
                "status": "pass" if not issues else "fail",
                "issue_count": len(issues),
                "issues": [issue.to_dict() for issue in issues],
            }
        )
    return {
        "path": root.as_posix(),
        "total_files": len(files),
        "passed_files": sum(1 for row in rows if row["status"] == "pass"),
        "failed_files": sum(1 for row in rows if row["status"] == "fail"),
        "status": "pass" if not all_issues else "fail",
        "files": rows,
        "issues": [issue.to_dict() for issue in all_issues],
    }


def _validate_tensor(
    issues: list[ValidationIssue],
    path: str,
    tensor: Any,
    label: str,
) -> None:
    if not isinstance(tensor, dict):
        _error(issues, path, f"{label} must be a mapping")
        return
    _require_keys(issues, path, tensor, REQUIRED_TENSOR_KEYS, label)
    if not tensor.get("name"):
        _error(issues, path, f"{label}.name must be non-empty")
    shape = tensor.get("shape")
    if not isinstance(shape, list) or not shape or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        _error(issues, path, f"{label}.shape must be a non-empty list of positive integers")
    if not tensor.get("dtype"):
        _error(issues, path, f"{label}.dtype must be non-empty")
    if not tensor.get("layout"):
        _error(issues, path, f"{label}.layout must be non-empty")


def _validate_tuple_outputs(
    issues: list[ValidationIssue],
    path: str,
    spec: dict[str, Any],
) -> None:
    outputs = spec.get("outputs", [])
    output_names = [item.get("name") for item in outputs if isinstance(item, dict)]
    semantics = spec.get("semantics") if isinstance(spec.get("semantics"), dict) else {}
    semantic_outputs = semantics.get("outputs")
    if not isinstance(semantic_outputs, list):
        _error(issues, path, "tuple-output OpSpec must define semantics.outputs")
    elif [item.get("name") for item in semantic_outputs if isinstance(item, dict)] != output_names:
        _error(issues, path, "semantics.outputs must match OpSpec outputs")


def _validate_matmul_like_sketch(
    issues: list[ValidationIssue],
    path: str,
    sketch: dict[str, Any],
) -> None:
    pattern = sketch.get("compute_pattern")
    if pattern in {"matmul_basic", "matmul_biasadd"}:
        axes = sketch.get("tile_plan", {}).get("axes") if isinstance(sketch.get("tile_plan"), dict) else None
        if axes != ["M", "N", "K"]:
            _error(issues, path, "matmul sketch tile_plan.axes must be ['M', 'N', 'K']")
        axis_map = sketch.get("axis_map")
        if not isinstance(axis_map, dict) or not {"M", "N", "K"} <= set(axis_map):
            _error(issues, path, "matmul sketch axis_map must include M, N, and K")
        dtype_plan = sketch.get("dtype_plan")
        if not isinstance(dtype_plan, dict) or dtype_plan.get("accumulator") != "float32":
            _error(issues, path, "matmul sketch dtype_plan.accumulator must be float32")
        if pattern == "matmul_biasadd":
            bias = sketch.get("memory_plan", {}).get("bias") if isinstance(sketch.get("memory_plan"), dict) else None
            if not isinstance(bias, dict) or "broadcast" not in bias:
                _error(issues, path, "matmul_biasadd sketch must describe bias broadcast")


def _validate_rowwise_sketch(
    issues: list[ValidationIssue],
    path: str,
    sketch: dict[str, Any],
    category: str,
) -> None:
    if sketch.get("accumulation_dtype") != "float32":
        _error(issues, path, f"{category} sketch accumulation_dtype must be float32")
    tile_plan = sketch.get("tile_plan")
    if isinstance(tile_plan, dict) and len(tile_plan.get("shape") or []) < 2:
        _error(issues, path, f"{category} sketch tile_plan.shape must include row and feature/reduction dimensions")


def _require_keys(
    issues: list[ValidationIssue],
    path: str,
    value: dict[str, Any],
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    for key in missing:
        _error(issues, path, f"{label} missing required key {key!r}")


def _error(issues: list[ValidationIssue], path: str, message: str) -> None:
    issues.append(ValidationIssue(path=path, level="error", message=message))
