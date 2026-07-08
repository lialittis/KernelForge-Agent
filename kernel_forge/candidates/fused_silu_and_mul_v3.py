import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_BLOCK_SIZE = 4096
_HAS_TRITON = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "exp")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON:

    @triton.jit
    def _swiglu_kernel(combined_ptr, output_ptr, n_outputs, hidden_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_outputs
        row = offsets // hidden_size
        col = offsets - row * hidden_size
        row_base = row * hidden_size * 2
        x = tl.load(combined_ptr + row_base + col, mask=mask, other=0.0)
        y = tl.load(combined_ptr + row_base + hidden_size + col, mask=mask, other=0.0)
        out = x * (1.0 / (1.0 + tl.exp(-x))) * y
        tl.store(output_ptr + offsets, out, mask=mask)

else:
    _swiglu_kernel = None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, combined):
        if not self._can_try_triton(combined):
            return self._torch_reference(combined, "torch_fallback_unavailable")

        hidden_size = combined.shape[-1] // 2
        output = torch.empty((combined.shape[0], hidden_size), device=combined.device, dtype=combined.dtype)
        n_outputs = output.numel()
        try:
            grid = (triton.cdiv(n_outputs, _BLOCK_SIZE),)
            _swiglu_kernel[grid](
                combined,
                output,
                n_outputs,
                hidden_size,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
            self._last_backend = "triton_flat_swiglu_bs4096"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(combined, "torch_fallback_after_error")

    def _can_try_triton(self, combined):
        if self._disable_triton or _swiglu_kernel is None:
            return False
        if getattr(combined.device, "type", None) != "npu":
            return False
        return (
            combined.is_contiguous()
            and combined.ndim == 2
            and combined.shape[-1] % 2 == 0
            and combined.dtype == torch.float32
        )

    def _torch_reference(self, combined, reason):
        x, y = torch.chunk(combined, 2, dim=-1)
        self._last_backend = reason
        return torch.nn.functional.silu(x) * y
