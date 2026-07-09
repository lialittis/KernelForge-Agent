from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

from .opspec import OpSpec, TensorSpec, read_yaml
from .sketch import build_operator_sketch


class ExtractionError(ValueError):
    """Raised when a benchmark case does not match supported extraction patterns."""


SUPPORTED_OPSPEC_CASES = {
    "t1/gelu",
    "t1/fused_silu_and_mul",
    "t1/matmul_basic",
    "t1/matmul_biasadd",
    "t1/sigmoid_scale_sum",
    "t1/softmax",
    "t2/add_rmsnorm_cast",
    "t2/add_rmsnorm_quant",
    "t2/moe_topk_softmax",
    "t2/rope",
    "t3/causal_conv1d",
    "t3/decode_mla",
    "t3/layernorm_gated",
}

CASE_CATEGORIES = {
    "t1/gelu": "elementwise",
    "t1/fused_silu_and_mul": "elementwise",
    "t1/sigmoid_scale_sum": "reduction",
    "t1/softmax": "reduction",
    "t1/matmul_basic": "matmul_like",
    "t1/matmul_biasadd": "matmul_like",
    "t2/rope": "transpose_layout",
    "t2/add_rmsnorm_cast": "normalization",
    "t2/add_rmsnorm_quant": "normalization",
    "t2/moe_topk_softmax": "reduction",
    "t3/causal_conv1d": "convolution",
    "t3/decode_mla": "matmul_like",
    "t3/layernorm_gated": "normalization",
}


def extract_opspec(
    case_path: str | Path,
    *,
    experiment_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    backend_target: str = "triton_ascend",
    allow_unsupported: bool = False,
) -> dict[str, Any]:
    parsed = inspect_case(case_path, repo_root=repo_root)
    support = case_support(parsed["id"])
    if support["status"] != "opspec_supported":
        if allow_unsupported:
            return _unsupported_metadata(parsed, support)
        raise ExtractionError(f"Unsupported OpSpec case {parsed['id']}: {support['reason']}")

    input_specs = parsed["inputs"]
    init_inputs = parsed["init_inputs"]
    forward_args = parsed["forward_args"]
    expression = parsed["forward_expression"]

    expected_args = [item.name for item in input_specs]
    if forward_args != expected_args:
        raise ExtractionError(
            f"Forward args {forward_args} do not match inputs {expected_args}"
        )

    outputs = _build_output_specs(parsed["id"], input_specs, init_inputs)
    performance = _build_performance(experiment_path)
    spec = OpSpec(
        id=parsed["id"],
        name=parsed["name"],
        tier=parsed["tier"],
        category=classify_case(parsed["id"]),
        source_path=parsed["source_path"],
        inputs=input_specs,
        outputs=outputs,
        semantics=_build_semantics(parsed["id"], expression, init_inputs),
        validation=_build_validation(input_specs),
        performance=performance,
        sketch=build_operator_sketch(parsed["id"], input_specs, outputs, backend_target),
        submission={
            "required_files": [f"{parsed['tier']}/{parsed['name']}.py"],
            "entrypoint": "ModelNew",
        },
    )
    return spec.to_dict()


def inspect_case(
    case_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    case = Path(case_path)
    root = Path(repo_root) if repo_root else Path.cwd()
    module = ast.parse(case.read_text(encoding="utf-8"), filename=str(case))
    tier = case.parent.name
    name = case.stem
    input_specs = _extract_get_inputs(module)
    init_inputs = _extract_get_init_inputs(module)
    forward_args, expression = _extract_forward_expression(module)
    return {
        "id": f"{tier}/{name}",
        "name": name,
        "tier": tier,
        "category": classify_case(f"{tier}/{name}"),
        "source_path": _display_path(case, root),
        "reference_class": "Model",
        "candidate_class": "ModelNew",
        "inputs": input_specs,
        "init_inputs": init_inputs,
        "forward_args": forward_args,
        "forward_expression": expression,
    }


def classify_case(case_id: str) -> str:
    return CASE_CATEGORIES.get(case_id, "unknown")


def case_support(case_id: str) -> dict[str, str]:
    if case_id in SUPPORTED_OPSPEC_CASES:
        return {
            "status": "opspec_supported",
            "reason": "covered by deterministic OpSpec extraction templates",
        }
    category = classify_case(case_id)
    if category == "matmul_like":
        reason = "matmul-like case is not covered by the current extraction templates"
    elif case_id.startswith("t2/") or case_id.startswith("t3/"):
        reason = "higher-tier cases are deferred until T1 extraction and reporting are stable"
    else:
        reason = "case is not covered by the current extraction templates"
    return {"status": "unsupported", "reason": reason}


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_get_inputs(module: ast.Module) -> list[TensorSpec]:
    fn = _find_function(module, "get_inputs")
    tensors: dict[str, TensorSpec] = {}
    env: dict[str, Any] = {}
    for node in fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            _record_assignment(target, node.value, env)
            if isinstance(target, ast.Name):
                tensor = _tensor_from_randn(target.id, node.value, env)
                if tensor is None:
                    tensor = _tensor_from_cat(target.id, node.value, tensors)
                if tensor is None:
                    tensor = _tensor_from_full(target.id, node.value, env)
                if tensor is None:
                    tensor = _tensor_from_arange(target.id, node.value, env)
                if tensor is not None:
                    tensors[target.id] = tensor
    returned = _extract_returned_names(fn)
    missing = [name for name in returned if name not in tensors]
    if missing:
        raise ExtractionError(f"Returned inputs are not supported tensor assignments: {missing}")
    return [tensors[name] for name in returned]


def _extract_get_init_inputs(module: ast.Module) -> list[Any]:
    fn = _find_function(module, "get_init_inputs")
    assignments: dict[str, Any] = {}
    for node in fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            _record_assignment(target, node.value, assignments)
            if isinstance(target, ast.Name):
                assignments[target.id] = _literal_value(node.value, assignments)
        if isinstance(node, ast.Return):
            if not isinstance(node.value, ast.List):
                raise ExtractionError("get_init_inputs() must return a list")
            values = []
            for element in node.value.elts:
                if isinstance(element, ast.Name):
                    if element.id not in assignments:
                        raise ExtractionError(
                            f"Unsupported get_init_inputs() name {element.id}"
                        )
                    values.append(assignments[element.id])
                else:
                    values.append(_literal_value(element, assignments))
            return values
    raise ExtractionError("get_init_inputs() has no return statement")


def _build_output_specs(
    case_id: str,
    input_specs: list[TensorSpec],
    init_inputs: list[Any] | None = None,
) -> list[TensorSpec]:
    if case_id in {"t1/gelu", "t1/softmax"}:
        first = input_specs[0]
        return [
            TensorSpec(
                name="output",
                shape=first.shape,
                dtype=first.dtype,
                layout=first.layout,
            )
        ]
    if case_id == "t1/fused_silu_and_mul":
        combined = input_specs[0]
        if not combined.shape or combined.shape[-1] % 2 != 0:
            raise ExtractionError("SwiGLU input last dimension must be even")
        output_shape = [*combined.shape[:-1], combined.shape[-1] // 2]
        return [
            TensorSpec(
                name="output",
                shape=output_shape,
                dtype=combined.dtype,
                layout=combined.layout,
            )
        ]
    if case_id == "t1/sigmoid_scale_sum":
        x = input_specs[0]
        if not x.shape:
            raise ExtractionError("sigmoid_scale_sum expects at least one input axis")
        return [
            TensorSpec(
                name="output",
                shape=[*x.shape[:-1], 1],
                dtype=x.dtype,
                layout=x.layout,
            )
        ]
    if case_id in {"t1/matmul_basic", "t1/matmul_biasadd"}:
        a = input_specs[0]
        b = input_specs[1]
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ExtractionError(f"{case_id} expects rank-2 matmul inputs")
        if a.shape[1] != b.shape[0]:
            raise ExtractionError(
                f"{case_id} matmul K dimensions do not match: {a.shape[1]} vs {b.shape[0]}"
            )
        return [
            TensorSpec(
                name="output",
                shape=[a.shape[0], b.shape[1]],
                dtype=a.dtype,
                layout=a.layout,
            )
        ]
    if case_id == "t2/moe_topk_softmax":
        logits = input_specs[0]
        top_k = init_inputs[0] if init_inputs else 2
        return [
            TensorSpec(
                name="top_k_probs",
                shape=[*logits.shape[:-1], top_k],
                dtype=logits.dtype,
                layout=logits.layout,
            ),
            TensorSpec(
                name="top_k_indices",
                shape=[*logits.shape[:-1], top_k],
                dtype="int64",
                layout=logits.layout,
            ),
        ]
    if case_id == "t2/add_rmsnorm_cast":
        x = input_specs[0]
        return [
            TensorSpec(
                name="output",
                shape=x.shape,
                dtype="float16",
                layout=x.layout,
            )
        ]
    if case_id == "t2/add_rmsnorm_quant":
        x = input_specs[0]
        return [
            TensorSpec(
                name="output",
                shape=x.shape,
                dtype="int8",
                layout=x.layout,
            )
        ]
    if case_id in {"t2/rope", "t3/layernorm_gated"}:
        x = input_specs[0]
        return [
            TensorSpec(
                name="output",
                shape=x.shape,
                dtype=x.dtype,
                layout=x.layout,
            )
        ]
    if case_id == "t3/causal_conv1d":
        x = input_specs[0]
        return [
            TensorSpec(
                name="output",
                shape=x.shape,
                dtype=x.dtype,
                layout=x.layout,
            )
        ]
    if case_id == "t3/decode_mla":
        q = input_specs[0]
        v_buffer = input_specs[3]
        return [
            TensorSpec(
                name="output",
                shape=[q.shape[0], q.shape[1], v_buffer.shape[-1]],
                dtype=v_buffer.dtype,
                layout=q.layout,
            )
        ]
    raise ExtractionError(f"No output inference rule for {case_id}")


def _build_semantics(
    case_id: str,
    expression: str,
    init_inputs: list[Any],
) -> dict[str, Any]:
    if case_id == "t1/gelu":
        return {
            "expression": expression,
            "formula": "0.5 * x * (1 + erf(x / sqrt(2)))",
            "broadcast": "none",
            "reduction_axes": [],
            "normalization_axes": [],
            "layout_transform": "none",
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t1/fused_silu_and_mul":
        return {
            "expression": expression,
            "formula": "silu(combined[..., :H]) * combined[..., H:]",
            "broadcast": "none",
            "reduction_axes": [],
            "normalization_axes": [],
            "layout_transform": "split_last_dim",
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t1/sigmoid_scale_sum":
        return {
            "expression": expression,
            "formula": "sum(sigmoid(x * 2.0 + bias), dim=-1, keepdim=True)",
            "broadcast": "bias broadcasts over the leading x axis",
            "reduction_axes": [-1],
            "normalization_axes": [],
            "layout_transform": "none",
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t1/softmax":
        dim = init_inputs[0] if init_inputs else -1
        return {
            "expression": expression,
            "formula": "exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)",
            "broadcast": "none",
            "reduction_axes": [dim],
            "normalization_axes": [dim],
            "layout_transform": "none",
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t1/matmul_basic":
        return {
            "expression": expression,
            "formula": "C[m, n] = sum_k A[m, k] * B[k, n]",
            "broadcast": "none",
            "reduction_axes": ["K"],
            "normalization_axes": [],
            "layout_transform": "none",
            "accumulation_dtype": "float32",
            "boundary_conditions": {
                "tail_mask_required": True,
                "rank": 2,
                "k_dimensions_must_match": True,
            },
        }
    if case_id == "t1/matmul_biasadd":
        return {
            "expression": expression,
            "formula": "C[m, n] = sum_k A[m, k] * B[k, n] + bias[0, n]",
            "broadcast": "bias broadcasts over the M axis",
            "reduction_axes": ["K"],
            "normalization_axes": [],
            "layout_transform": "none",
            "accumulation_dtype": "float32",
            "boundary_conditions": {
                "tail_mask_required": True,
                "rank": 2,
                "bias_shape": [1, "N"],
                "k_dimensions_must_match": True,
            },
        }
    if case_id == "t2/add_rmsnorm_cast":
        target_dtype = init_inputs[1] if len(init_inputs) > 1 else "torch.float16"
        return {
            "expression": expression,
            "formula": "cast(((x + residual) * rsqrt(mean((x + residual)^2, dim=-1) + eps)) * gamma, target_dtype)",
            "broadcast": "gamma broadcasts over batch and sequence axes",
            "reduction_axes": [-1],
            "normalization_axes": [-1],
            "layout_transform": "none",
            "epsilon": init_inputs[0] if init_inputs else 1.0e-6,
            "target_dtype": _normalize_dtype_name(str(target_dtype)),
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t2/add_rmsnorm_quant":
        return {
            "expression": expression,
            "formula": "round((((x + residual) * rsqrt(mean((x + residual)^2, dim=-1) + eps)) * gamma) / scale + zero_point).clamp(-128, 127).to(int8)",
            "broadcast": "gamma broadcasts over batch and sequence axes; scale and zero_point are scalar tensors",
            "reduction_axes": [-1],
            "normalization_axes": [-1],
            "layout_transform": "none",
            "epsilon": init_inputs[0] if init_inputs else 1.0e-6,
            "quantization": {"dtype": "int8", "round": True, "clamp": [-128, 127]},
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t2/moe_topk_softmax":
        top_k = init_inputs[0] if init_inputs else 2
        return {
            "expression": expression,
            "formula": "topk_probs, topk_indices = topk(softmax(gating_logits, dim=-1), k); topk_probs = topk_probs / sum(topk_probs, dim=-1, keepdim=True)",
            "broadcast": "none",
            "reduction_axes": [-1],
            "normalization_axes": [-1],
            "layout_transform": "topk_tuple_output",
            "top_k": top_k,
            "outputs": [
                {"name": "top_k_probs", "semantics": "renormalized top-k probabilities"},
                {"name": "top_k_indices", "semantics": "top-k expert indices"},
            ],
            "boundary_conditions": {
                "tail_mask_required": True,
                "stable_topk_tie_behavior_required": True,
                "indices_dtype": "int64",
            },
        }
    if case_id == "t2/rope":
        return {
            "expression": expression,
            "formula": "cos * x + sin * concat(-x[..., D/2:], x[..., :D/2])",
            "broadcast": "cos and sin broadcast over batch and head axes",
            "reduction_axes": [],
            "normalization_axes": [],
            "layout_transform": "rotate_half_last_dim",
            "boundary_conditions": {
                "tail_mask_required": True,
                "last_dim_even": True,
                "last_dim_multiple_of_64": True,
            },
        }
    if case_id == "t3/layernorm_gated":
        eps = init_inputs[0] if init_inputs else 1.0e-6
        norm_before_gate = init_inputs[1] if len(init_inputs) > 1 else True
        is_rms_norm = init_inputs[2] if len(init_inputs) > 2 else True
        return {
            "expression": expression,
            "formula": "((x * rsqrt(mean(x^2, dim=-1) + eps)) * weight) * sigmoid(z)",
            "broadcast": "weight broadcasts over batch and sequence axes",
            "reduction_axes": [-1],
            "normalization_axes": [-1],
            "layout_transform": "none",
            "epsilon": eps,
            "norm_before_gate": norm_before_gate,
            "is_rms_norm": is_rms_norm,
            "boundary_conditions": {"tail_mask_required": True},
        }
    if case_id == "t3/causal_conv1d":
        activation = init_inputs[0] if init_inputs else "silu"
        return {
            "expression": expression,
            "formula": "silu(sum_{w=0..W-1} padded_x[:, channel, w] * weight[channel, w] + bias[channel])",
            "broadcast": "bias broadcasts over batch axis",
            "reduction_axes": ["conv_width"],
            "normalization_axes": [],
            "layout_transform": "append_current_token_to_conv_state",
            "activation": activation,
            "state_update": "conv_state.copy_(x_padded[:, :, -(width - 1):])",
            "boundary_conditions": {
                "causal": True,
                "tail_mask_required": True,
                "state_width": "weight.shape[1] - 1",
            },
        }
    if case_id == "t3/decode_mla":
        sm_scale = init_inputs[0] if init_inputs else None
        page_size = init_inputs[1] if len(init_inputs) > 1 else None
        return {
            "expression": expression,
            "formula": "softmax(((q_nope @ k_nope.T) + (q_rope @ k_rope.T)) * sm_scale) @ v",
            "broadcast": "KV heads repeat across query heads when num_q_heads != num_kv_heads",
            "reduction_axes": ["qk_nope_dim", "qk_rope_dim", "kv_sequence"],
            "normalization_axes": ["kv_sequence"],
            "layout_transform": "paged_kv_cache_gather",
            "sm_scale": sm_scale,
            "page_size": page_size,
            "boundary_conditions": {
                "variable_sequence_lengths": True,
                "block_table_pages": True,
                "tail_mask_required": True,
            },
        }
    return {"expression": expression}


def _build_validation(input_specs: list[TensorSpec]) -> dict[str, Any]:
    return {
        "rtol": 1.0e-2,
        "atol": 1.0e-2,
        "max_error_required": None,
        "shape_cases": [item.shape for item in input_specs],
        "dtype_cases": sorted({item.dtype for item in input_specs}),
    }


def _unsupported_metadata(
    parsed: dict[str, Any],
    support: dict[str, str],
) -> dict[str, Any]:
    return {
        "id": parsed["id"],
        "name": parsed["name"],
        "tier": parsed["tier"],
        "category": parsed["category"],
        "source_path": parsed["source_path"],
        "reference_class": parsed["reference_class"],
        "candidate_class": parsed["candidate_class"],
        "support": support,
        "inputs": [item.to_dict() for item in parsed["inputs"]],
        "init_inputs": parsed["init_inputs"],
        "forward_args": parsed["forward_args"],
        "forward_expression": parsed["forward_expression"],
        "submission": {
            "required_files": [f"{parsed['tier']}/{parsed['name']}.py"],
            "entrypoint": "ModelNew",
        },
    }


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
    performance["baseline_latency_ms"] = result_perf.get(
        "baseline_median_latency_ms",
        result_perf.get("baseline_latency_ms"),
    )
    return performance


def _tensor_from_randn(
    name: str,
    value: ast.AST,
    env: dict[str, Any] | None = None,
) -> TensorSpec | None:
    call = _find_torch_call(value, "randn")
    if call is None:
        return None
    shape = [_literal_int(arg, env) for arg in call.args]
    dtype = "float32"
    for keyword in call.keywords:
        if keyword.arg == "dtype":
            dtype = _dtype_name(keyword.value, env)
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _tensor_from_cat(
    name: str,
    value: ast.AST,
    tensors: dict[str, TensorSpec],
) -> TensorSpec | None:
    if not isinstance(value, ast.Call) or ast.unparse(value.func) != "torch.cat":
        return None
    if not value.args or not isinstance(value.args[0], ast.List):
        raise ExtractionError("torch.cat input must be a list for benchmark parsing")
    input_names = []
    for element in value.args[0].elts:
        if not isinstance(element, ast.Name):
            raise ExtractionError("torch.cat inputs must be named tensors")
        input_names.append(element.id)
    if not input_names:
        raise ExtractionError("torch.cat requires at least one tensor")
    missing = [item for item in input_names if item not in tensors]
    if missing:
        raise ExtractionError(f"torch.cat inputs are not defined: {missing}")

    input_specs = [tensors[item] for item in input_names]
    rank = len(input_specs[0].shape)
    dim = 0
    for keyword in value.keywords:
        if keyword.arg == "dim":
            dim = _literal_int(keyword.value)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ExtractionError(f"torch.cat dim {dim} is out of range")

    shape = list(input_specs[0].shape)
    dtype = input_specs[0].dtype
    for spec in input_specs[1:]:
        if len(spec.shape) != rank:
            raise ExtractionError("torch.cat inputs must have the same rank")
        if spec.dtype != dtype:
            raise ExtractionError("torch.cat inputs must have the same dtype")
        for axis, size in enumerate(spec.shape):
            if axis == dim:
                continue
            if shape[axis] != size:
                raise ExtractionError("torch.cat non-concat dimensions must match")
        shape[dim] += spec.shape[dim]
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _tensor_from_full(
    name: str,
    value: ast.AST,
    env: dict[str, Any] | None = None,
) -> TensorSpec | None:
    call = _find_torch_call(value, "full")
    if call is None:
        return None
    if len(call.args) < 2:
        raise ExtractionError("torch.full requires shape and fill value")
    shape = _shape_from_arg(call.args[0], env)
    dtype = "float32"
    for keyword in call.keywords:
        if keyword.arg == "dtype":
            dtype = _dtype_name(keyword.value, env)
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _tensor_from_arange(
    name: str,
    value: ast.AST,
    env: dict[str, Any] | None = None,
) -> TensorSpec | None:
    call = _find_torch_call(value, "arange")
    if call is None:
        return None

    reshape_shape = _shape_from_reshape(value, env)
    if reshape_shape is not None:
        shape = reshape_shape
    else:
        shape = [_arange_length(call, env)]

    dtype = "int64"
    for keyword in call.keywords:
        if keyword.arg == "dtype":
            dtype = _dtype_name(keyword.value, env)
    return TensorSpec(name=name, shape=shape, dtype=dtype)


def _record_assignment(target: ast.AST, value: ast.AST, env: dict[str, Any]) -> None:
    if isinstance(target, ast.Name):
        try:
            env[target.id] = _literal_value(value, env)
        except ExtractionError:
            return
        return
    if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
        if len(target.elts) != len(value.elts):
            return
        for target_item, value_item in zip(target.elts, value.elts):
            if isinstance(target_item, ast.Name):
                try:
                    env[target_item.id] = _literal_value(value_item, env)
                except ExtractionError:
                    continue


def _literal_int(node: ast.AST, env: dict[str, Any] | None = None) -> int:
    value = _literal_value(node, env or {})
    if not isinstance(value, int):
        raise ExtractionError(f"Expected integer shape literal, got {value!r}")
    return value


def _literal_value(node: ast.AST, env: dict[str, Any] | None = None) -> Any:
    env = env or {}
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ExtractionError(f"Unsupported name literal {node.id}")
    if isinstance(node, ast.UnaryOp):
        operand = _literal_value(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ExtractionError(f"Unsupported unary literal {ast.unparse(node)}")
    if isinstance(node, ast.BinOp):
        left = _literal_value(node.left, env)
        right = _literal_value(node.right, env)
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        for op_type, op_func in operations.items():
            if isinstance(node.op, op_type):
                return op_func(left, right)
        raise ExtractionError(f"Unsupported arithmetic literal {ast.unparse(node)}")
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(item, env) for item in node.elts)
    if isinstance(node, ast.List):
        return [_literal_value(item, env) for item in node.elts]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ExtractionError(f"Expected literal value, got {ast.unparse(node)}") from exc


def _find_torch_call(node: ast.AST, function_name: str) -> ast.Call | None:
    if isinstance(node, ast.Call):
        if ast.unparse(node.func) == f"torch.{function_name}":
            return node
        if isinstance(node.func, ast.Attribute):
            nested = _find_torch_call(node.func.value, function_name)
            if nested is not None:
                return nested
        for arg in node.args:
            nested = _find_torch_call(arg, function_name)
            if nested is not None:
                return nested
    if isinstance(node, ast.BinOp):
        return _find_torch_call(node.left, function_name) or _find_torch_call(node.right, function_name)
    if isinstance(node, ast.UnaryOp):
        return _find_torch_call(node.operand, function_name)
    return None


def _shape_from_arg(node: ast.AST, env: dict[str, Any] | None = None) -> list[int]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal_int(item, env) for item in node.elts]
    value = _literal_value(node, env or {})
    if isinstance(value, int):
        return [value]
    if isinstance(value, (tuple, list)):
        if not all(isinstance(item, int) for item in value):
            raise ExtractionError(f"Shape contains non-integer values: {value!r}")
        return list(value)
    raise ExtractionError(f"Expected shape literal, got {ast.unparse(node)}")


def _shape_from_reshape(
    value: ast.AST,
    env: dict[str, Any] | None = None,
) -> list[int] | None:
    if not isinstance(value, ast.Call):
        return None
    if not isinstance(value.func, ast.Attribute):
        return None
    if value.func.attr not in {"reshape", "view"}:
        return None
    if len(value.args) == 1 and isinstance(value.args[0], (ast.Tuple, ast.List)):
        return _shape_from_arg(value.args[0], env)
    return [_literal_int(arg, env) for arg in value.args]


def _arange_length(call: ast.Call, env: dict[str, Any] | None = None) -> int:
    if not call.args:
        raise ExtractionError("torch.arange requires at least one bound")
    if len(call.args) == 1:
        start = 0
        stop = _literal_int(call.args[0], env)
        step = 1
    elif len(call.args) in {2, 3}:
        start = _literal_int(call.args[0], env)
        stop = _literal_int(call.args[1], env)
        step = _literal_int(call.args[2], env) if len(call.args) == 3 else 1
    else:
        raise ExtractionError("torch.arange with more than three positional args is unsupported")
    if step == 0:
        raise ExtractionError("torch.arange step must be non-zero")
    if (stop - start) * step <= 0:
        return 0
    return (abs(stop - start) + abs(step) - 1) // abs(step)


def _normalize_dtype_name(dtype: str) -> str:
    return dtype.split(".", 1)[1] if dtype.startswith("torch.") else dtype


def _dtype_name(node: ast.AST, env: dict[str, Any] | None = None) -> str:
    if isinstance(node, ast.Name) and env and node.id in env:
        value = str(env[node.id])
        return _normalize_dtype_name(value)
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
