from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .opspec import OpSpec, TensorSpec, read_yaml
from .sketch import build_gelu_sketch


class ExtractionError(ValueError):
    """Raised when a benchmark case does not match supported extraction patterns."""


def extract_opspec(
    case_path: str | Path,
    *,
    experiment_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    backend_target: str = "triton_ascend",
) -> dict[str, Any]:
    case = Path(case_path)
    root = Path(repo_root) if repo_root else Path.cwd()
    source_path = _display_path(case, root)
    module = ast.parse(case.read_text(encoding="utf-8"), filename=str(case))

    tier = case.parent.name
    name = case.stem
    if tier != "t1" or name != "gelu":
        raise ExtractionError("v1 extractor only supports t1/gelu.py")

    input_specs = _extract_get_inputs(module)
    if len(input_specs) != 1:
        raise ExtractionError("GELU extractor expects exactly one input tensor")
    init_inputs = _extract_get_init_inputs(module)
    if init_inputs != []:
        raise ExtractionError("GELU extractor expects get_init_inputs() to return []")

    forward_args, expression = _extract_forward_expression(module)
    if forward_args != [input_specs[0].name]:
        raise ExtractionError(
            f"Forward args {forward_args} do not match inputs {[item.name for item in input_specs]}"
        )
    if expression != f"torch.nn.functional.gelu({input_specs[0].name})":
        raise ExtractionError(f"Unsupported GELU expression: {expression}")

    output = TensorSpec(
        name="output",
        shape=input_specs[0].shape,
        dtype=input_specs[0].dtype,
        layout=input_specs[0].layout,
    )
    performance = _build_performance(experiment_path)
    spec = OpSpec(
        id=f"{tier}/{name}",
        name=name,
        tier=tier,
        category="elementwise",
        source_path=source_path,
        inputs=input_specs,
        outputs=[output],
        semantics={
            "expression": expression,
            "formula": "0.5 * x * (1 + erf(x / sqrt(2)))",
            "broadcast": "none",
            "reduction_axes": [],
            "normalization_axes": [],
            "layout_transform": "none",
            "boundary_conditions": {"tail_mask_required": True},
        },
        validation={
            "rtol": 1.0e-2,
            "atol": 1.0e-2,
            "max_error_required": None,
            "shape_cases": [input_specs[0].shape],
            "dtype_cases": [input_specs[0].dtype],
        },
        performance=performance,
        sketch=build_gelu_sketch(input_specs[0], backend_target),
        submission={
            "required_files": [f"{tier}/{name}.py"],
            "entrypoint": "ModelNew",
        },
    )
    return spec.to_dict()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_get_inputs(module: ast.Module) -> list[TensorSpec]:
    fn = _find_function(module, "get_inputs")
    tensors: dict[str, TensorSpec] = {}
    for node in fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                tensor = _tensor_from_randn(target.id, node.value)
                if tensor is not None:
                    tensors[target.id] = tensor
    returned = _extract_returned_names(fn)
    missing = [name for name in returned if name not in tensors]
    if missing:
        raise ExtractionError(f"Returned inputs are not supported tensor assignments: {missing}")
    return [tensors[name] for name in returned]


def _extract_get_init_inputs(module: ast.Module) -> list[Any]:
    fn = _find_function(module, "get_init_inputs")
    for node in fn.body:
        if isinstance(node, ast.Return):
            return ast.literal_eval(node.value)
    raise ExtractionError("get_init_inputs() has no return statement")


def _extract_forward_expression(module: ast.Module) -> tuple[list[str], str]:
    cls = _find_class(module, "Model")
    forward = _find_method(cls, "forward")
    args = [arg.arg for arg in forward.args.args if arg.arg != "self"]
    assignments: dict[str, str] = {}
    for node in forward.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = ast.unparse(node.value)
        if isinstance(node, ast.Return):
            expression = ast.unparse(node.value)
            expression = assignments.get(expression, expression)
            return args, expression
    raise ExtractionError("Model.forward() has no return statement")


def _build_performance(experiment_path: str | Path | None) -> dict[str, Any]:
    performance = {
        "metric": "median_latency_ms",
        "warmup": 10,
        "repeats": 100,
        "num_trials": 3,
        "baseline_latency_ms": None,
        "baseline_source": "official_reference_model",
    }
    if experiment_path is None:
        return performance

    experiment = read_yaml(experiment_path)
    result_perf = experiment.get("results", {}).get("performance", {})
    performance["warmup"] = result_perf.get("warmup", performance["warmup"])
    performance["repeats"] = result_perf.get("repeats", performance["repeats"])
    performance["num_trials"] = result_perf.get("num_trials", performance["num_trials"])
    performance["baseline_latency_ms"] = result_perf.get("baseline_median_latency_ms")
    return performance


def _tensor_from_randn(name: str, value: ast.AST) -> TensorSpec | None:
    if not isinstance(value, ast.Call) or ast.unparse(value.func) != "torch.randn":
        return None
    shape = [_literal_int(arg) for arg in value.args]
    dtype = "float32"
    for keyword in value.keywords:
        if keyword.arg == "dtype":
            dtype = _dtype_name(keyword.value)
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _literal_int(node: ast.AST) -> int:
    value = ast.literal_eval(node)
    if not isinstance(value, int):
        raise ExtractionError(f"Expected integer shape literal, got {value!r}")
    return value


def _dtype_name(node: ast.AST) -> str:
    text = ast.unparse(node)
    if text.startswith("torch."):
        return text.split(".", 1)[1]
    return text


def _extract_returned_names(fn: ast.FunctionDef) -> list[str]:
    for node in fn.body:
        if isinstance(node, ast.Return):
            if not isinstance(node.value, ast.List):
                raise ExtractionError("get_inputs() must return a list")
            names = []
            for element in node.value.elts:
                if not isinstance(element, ast.Name):
                    raise ExtractionError("get_inputs() return list must contain names")
                names.append(element.id)
            return names
    raise ExtractionError("get_inputs() has no return statement")


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ExtractionError(f"Missing function {name}")


def _find_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ExtractionError(f"Missing class {name}")


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ExtractionError(f"Missing method {cls.name}.{name}")

