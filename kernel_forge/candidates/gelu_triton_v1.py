import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_BLOCK_SIZE = 1024
_HAS_TRITON_ERF = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "erf")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON_ERF:

    @triton.jit
    def _gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
        tl.store(output_ptr + offsets, y, mask=mask)

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
        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)

        try:
            _gelu_kernel[grid](
                input_tensor,
                output,
                n_elements,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
            )
            self._last_backend = "triton"
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

