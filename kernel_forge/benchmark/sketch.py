from __future__ import annotations

from functools import reduce
from operator import mul

from .opspec import TensorSpec


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

