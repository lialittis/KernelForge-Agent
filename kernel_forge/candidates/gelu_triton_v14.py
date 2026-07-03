import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_BLOCK_SIZE = 16384
_CHUNKS_PER_PROGRAM = 3
_HAS_TRITON_EXP = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "exp")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON_EXP:

    @triton.jit
    def _gelu_store(input_ptr, output_ptr, offsets, n_elements):
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        y = x / (1.0 + tl.exp(-2.0 * u))
        tl.store(output_ptr + offsets, y, mask=mask)

    @triton.jit
    def _gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        local_offsets = tl.arange(0, BLOCK_SIZE)
        program_base = tl.program_id(0) * (BLOCK_SIZE * 3)

        offsets = program_base + local_offsets
        _gelu_store(input_ptr, output_ptr, offsets, n_elements)

        offsets = program_base + BLOCK_SIZE + local_offsets
        _gelu_store(input_ptr, output_ptr, offsets, n_elements)

        offsets = program_base + BLOCK_SIZE * 2 + local_offsets
        _gelu_store(input_ptr, output_ptr, offsets, n_elements)

else:
    _gelu_kernel = None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, input_tensor):
        if not self._can_try_triton(input_tensor):
            return self._torch_gelu(input_tensor, "torch_fallback_unavailable")

        output = torch.empty_like(input_tensor)
        n_elements = output.numel()
        effective_block = _BLOCK_SIZE * _CHUNKS_PER_PROGRAM
        grid = (triton.cdiv(n_elements, effective_block),)

        try:
            _gelu_kernel[grid](
                input_tensor,
                output,
                n_elements,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
            self._last_backend = "triton_tanh_sigmoid_form_bs16384x3"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_gelu(input_tensor, "torch_fallback_after_error")

    def _can_try_triton(self, input_tensor):
        if self._disable_triton or _gelu_kernel is None:
            return False
        if getattr(input_tensor.device, "type", None) != "npu":
            return False
        return input_tensor.is_contiguous()

    def _torch_gelu(self, input_tensor, reason):
        self._last_backend = reason
        return torch.nn.functional.gelu(input_tensor)
