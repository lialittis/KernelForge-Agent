from __future__ import annotations

from functools import reduce
from operator import mul

from .opspec import TensorSpec


def build_operator_sketch(
    case_id: str,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    backend_target: str,
) -> dict:
    if case_id == "t1/gelu":
        return build_gelu_sketch(input_specs[0], backend_target)
    if case_id == "t1/fused_silu_and_mul":
        return build_fused_elementwise_sketch(
            input_specs,
            output_specs[0],
            backend_target,
            compute_pattern="fused_silu_and_mul",
            stages=["load_combined", "split_last_dim", "compute_silu_mul", "store"],
            known_risks=[
                "last_dim_split_correctness",
                "torch_npu_intrinsic_reference_semantics",
                "tail_mask_correctness",
            ],
        )
    if case_id == "t1/sigmoid_scale_sum":
        return build_rowwise_reduction_sketch(
            input_specs,
            output_specs[0],
            backend_target,
            compute_pattern="sigmoid_scale_sum_row_reduction",
            stages=["load_x_and_bias", "compute_sigmoid", "row_sum", "store"],
            known_risks=[
                "broadcast_bias_correctness",
                "reduction_accuracy",
                "row_boundary_mapping",
            ],
        )
    if case_id == "t1/softmax":
        return build_rowwise_reduction_sketch(
            input_specs,
            output_specs[0],
            backend_target,
            compute_pattern="rowwise_softmax",
            stages=["load_row", "max_reduce", "exp", "sum_reduce", "normalize", "store"],
            known_risks=[
                "max_subtraction_numerical_stability",
                "reduction_accuracy",
                "row_boundary_mapping",
            ],
        )
    if case_id == "t2/add_rmsnorm_cast":
        return build_rowwise_normalization_sketch(
            input_specs,
            output_specs[0],
            backend_target,
            compute_pattern="add_rmsnorm_cast",
            stages=[
                "load_x_residual_gamma",
                "add_residual",
                "sum_square_reduce",
                "rsqrt",
                "scale_gamma",
                "cast_output",
                "store",
            ],
            known_risks=[
                "rmsnorm_reduction_accuracy",
                "gamma_broadcast_correctness",
                "float32_accumulation_before_cast",
                "tail_mask_correctness",
            ],
        )
    if case_id == "t2/add_rmsnorm_quant":
        return build_rowwise_normalization_sketch(
            input_specs,
            output_specs[0],
            backend_target,
            compute_pattern="add_rmsnorm_quant",
            stages=[
                "load_x_residual_gamma_scale_zero_point",
                "add_residual",
                "sum_square_reduce",
                "rsqrt",
                "scale_gamma",
                "quantize_round_clamp",
                "store_int8",
            ],
            known_risks=[
                "rmsnorm_reduction_accuracy",
                "scalar_quant_parameter_broadcast",
                "rounding_semantics",
                "int8_clamp_saturation",
            ],
        )
    if case_id == "t2/rope":
        return build_rope_sketch(input_specs, output_specs[0], backend_target)
    if case_id == "t3/layernorm_gated":
        return build_rowwise_normalization_sketch(
            input_specs,
            output_specs[0],
            backend_target,
            compute_pattern="gated_rmsnorm",
            stages=[
                "load_x_weight_gate",
                "sum_square_reduce",
                "rsqrt",
                "scale_weight",
                "sigmoid_gate",
                "multiply_gate",
                "store",
            ],
            known_risks=[
                "rmsnorm_reduction_accuracy",
                "weight_broadcast_correctness",
                "sigmoid_fp16_accuracy",
                "norm_before_gate_semantics",
            ],
        )
    return build_generic_sketch(input_specs, output_specs, backend_target)


def build_gelu_sketch(input_spec: TensorSpec, backend_target: str) -> dict:
    element_count = reduce(mul, input_spec.shape, 1)
    return {
        "operator_category": "elementwise",
        "compute_pattern": "unary_pointwise_gelu",
        "parallel_axes": ["flattened_output_elements"],
        "tile_plan": {
            "strategy": "contiguous_1d_tiling",
            "shape": [element_count],
            "tunable": True,
        },
        "memory_plan": {
            "input": "contiguous_global_read",
            "output": "contiguous_global_write",
            "intermediate": "local_register_or_vector",
        },
        "pipeline_plan": {"stages": ["load", "compute_gelu", "store"]},
        "boundary_mask": {
            "required": True,
            "reason": "flattened element count may not be divisible by backend tile size",
        },
        "accumulation_dtype": input_spec.dtype,
        "backend_target": backend_target,
        "performance_knobs": {
            "tile_size": None,
            "vector_width": None,
            "num_warps_or_cores": None,
        },
        "known_risks": [
            "backend_gelu_intrinsic_or_erf_availability",
            "tail_mask_correctness",
        ],
    }


def build_fused_elementwise_sketch(
    input_specs: list[TensorSpec],
    output_spec: TensorSpec,
    backend_target: str,
    *,
    compute_pattern: str,
    stages: list[str],
    known_risks: list[str],
) -> dict:
    element_count = reduce(mul, output_spec.shape, 1)
    return {
        "operator_category": "elementwise",
        "compute_pattern": compute_pattern,
        "parallel_axes": ["flattened_output_elements"],
        "tile_plan": {
            "strategy": "contiguous_1d_output_tiling",
            "shape": [element_count],
            "tunable": True,
        },
        "memory_plan": {
            "inputs": [item.name for item in input_specs],
            "input": "contiguous_global_read",
            "output": "contiguous_global_write",
            "intermediate": "local_register_or_vector",
        },
        "pipeline_plan": {"stages": stages},
        "boundary_mask": {
            "required": True,
            "reason": "flattened output element count may not be divisible by tile size",
        },
        "accumulation_dtype": output_spec.dtype,
        "backend_target": backend_target,
        "performance_knobs": {
            "tile_size": None,
            "vector_width": None,
            "num_warps_or_cores": None,
        },
        "known_risks": known_risks,
    }


def build_rowwise_reduction_sketch(
    input_specs: list[TensorSpec],
    output_spec: TensorSpec,
    backend_target: str,
    *,
    compute_pattern: str,
    stages: list[str],
    known_risks: list[str],
) -> dict:
    rows, reduction_width = _rowwise_shape(input_specs[0].shape)
    return {
        "operator_category": "reduction",
        "compute_pattern": compute_pattern,
        "parallel_axes": ["outer_rows"],
        "tile_plan": {
            "strategy": "rowwise_last_dim_tiling",
            "shape": [rows, reduction_width],
            "tunable": True,
        },
        "memory_plan": {
            "inputs": [item.name for item in input_specs],
            "input": "row_contiguous_global_read",
            "output": "contiguous_global_write",
            "intermediate": "local_vector_reduction",
        },
        "pipeline_plan": {"stages": stages},
        "boundary_mask": {
            "required": True,
            "reason": "row count or reduction width may not align with backend tile size",
        },
        "accumulation_dtype": "float32",
        "backend_target": backend_target,
        "performance_knobs": {
            "row_tile": None,
            "reduction_tile": None,
            "num_warps_or_cores": None,
        },
        "known_risks": known_risks,
    }


def build_rowwise_normalization_sketch(
    input_specs: list[TensorSpec],
    output_spec: TensorSpec,
    backend_target: str,
    *,
    compute_pattern: str,
    stages: list[str],
    known_risks: list[str],
) -> dict:
    rows, hidden_size = _rowwise_shape(input_specs[0].shape)
    return {
        "operator_category": "normalization",
        "compute_pattern": compute_pattern,
        "parallel_axes": ["outer_rows", "hidden_dim"],
        "tile_plan": {
            "strategy": "rowwise_last_dim_normalization",
            "shape": [rows, hidden_size],
            "tunable": True,
        },
        "memory_plan": {
            "inputs": [item.name for item in input_specs],
            "input": "row_contiguous_global_read",
            "output": "contiguous_global_write",
            "intermediate": "local_vector_reduction",
        },
        "pipeline_plan": {"stages": stages},
        "boundary_mask": {
            "required": True,
            "reason": "outer rows and hidden dimension may not align with tile size",
        },
        "accumulation_dtype": "float32",
        "backend_target": backend_target,
        "performance_knobs": {
            "row_tile": None,
            "hidden_tile": None,
            "num_warps_or_cores": None,
            "vector_width": None,
        },
        "known_risks": known_risks,
    }


def build_rope_sketch(
    input_specs: list[TensorSpec],
    output_spec: TensorSpec,
    backend_target: str,
) -> dict:
    rows, head_dim = _rowwise_shape(input_specs[0].shape)
    return {
        "operator_category": "transpose_layout",
        "compute_pattern": "rotary_position_embedding",
        "parallel_axes": ["batch_head_sequence_rows", "head_dim"],
        "tile_plan": {
            "strategy": "rowwise_rotate_half_last_dim",
            "shape": [rows, head_dim],
            "tunable": True,
        },
        "memory_plan": {
            "inputs": [item.name for item in input_specs],
            "input": "row_contiguous_global_read_with_cos_sin_broadcast",
            "output": "contiguous_global_write",
            "intermediate": "local_register_or_vector",
        },
        "pipeline_plan": {
            "stages": ["load_x_cos_sin", "rotate_half", "fused_mul_add", "store"]
        },
        "boundary_mask": {
            "required": True,
            "reason": "row count or head dimension may not align with backend tile size",
        },
        "accumulation_dtype": output_spec.dtype,
        "backend_target": backend_target,
        "performance_knobs": {
            "row_tile": None,
            "head_dim_tile": None,
            "num_warps_or_cores": None,
        },
        "known_risks": [
            "rotate_half_pairing_semantics",
            "cos_sin_broadcast_layout",
            "float16_accuracy",
            "last_dim_multiple_constraints",
        ],
    }


def build_generic_sketch(
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    backend_target: str,
) -> dict:
    output_shape = output_specs[0].shape if output_specs else []
    return {
        "operator_category": "unknown",
        "compute_pattern": "unsupported_pending_classification",
        "parallel_axes": [],
        "tile_plan": {
            "strategy": "manual_required",
            "shape": output_shape,
            "tunable": False,
        },
        "memory_plan": {"inputs": [item.name for item in input_specs]},
        "pipeline_plan": {"stages": []},
        "boundary_mask": {"required": None, "reason": "unsupported case"},
        "accumulation_dtype": None,
        "backend_target": backend_target,
        "performance_knobs": {},
        "known_risks": ["unsupported_opspec_case"],
    }


def _rowwise_shape(shape: list[int]) -> tuple[int, int]:
    if not shape:
        return 1, 1
    reduction_width = shape[-1]
    rows = reduce(mul, shape[:-1], 1)
    return rows, reduction_width
