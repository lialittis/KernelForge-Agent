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
    and hasattr(tl, "max")
    and hasattr(tl, "sum")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON:

    @triton.jit
    def _softmax_kernel(x_ptr, out_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
        shifted = x - tl.max(x, axis=0)
        numerator = tl.exp(shifted)
        denominator = tl.sum(numerator, axis=0)
        y = numerator / denominator
        tl.store(out_ptr + row * n_cols + offsets, y, mask=mask)

else:
    _softmax_kernel = None


class ModelNew(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, input_tensor):
        if not self._can_try_triton(input_tensor):
            return self._torch_reference(input_tensor, "torch_fallback_unavailable")

        output = torch.empty_like(input_tensor)
        n_cols = input_tensor.shape[-1]
        n_rows = input_tensor.numel() // n_cols
        try:
            grid = (n_rows,)
            _softmax_kernel[grid](
                input_tensor,
                output,
                n_rows,
                n_cols,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
            self._last_backend = "triton_row_softmax_bs4096_rpp1"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(input_tensor, "torch_fallback_after_error")

    def _can_try_triton(self, input_tensor):
        if self._disable_triton or _softmax_kernel is None:
            return False
        if self.dim not in (-1, input_tensor.ndim - 1):
            return False
        if getattr(input_tensor.device, "type", None) != "npu":
            return False
        return input_tensor.is_contiguous() and input_tensor.ndim >= 1 and input_tensor.dtype == torch.float32

    def _torch_reference(self, input_tensor, reason):
        self._last_backend = reason
        return torch.softmax(input_tensor, dim=self.dim)
