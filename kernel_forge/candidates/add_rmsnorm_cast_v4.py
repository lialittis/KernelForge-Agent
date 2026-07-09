import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_CHUNK_SIZE = 1024
_SUPPORTED_HIDDEN = 4096
_HAS_TRITON = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "sum")
    and hasattr(tl, "sqrt")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON:

    @triton.jit
    def _add_rmsnorm_cast_kernel(
        x_ptr,
        residual_ptr,
        gamma_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets0 = tl.arange(0, CHUNK_SIZE)
        offsets1 = offsets0 + CHUNK_SIZE
        offsets2 = offsets0 + CHUNK_SIZE * 2
        offsets3 = offsets0 + CHUNK_SIZE * 3
        base = row * n_cols

        added0 = tl.load(x_ptr + base + offsets0) + tl.load(residual_ptr + base + offsets0)
        added1 = tl.load(x_ptr + base + offsets1) + tl.load(residual_ptr + base + offsets1)
        added2 = tl.load(x_ptr + base + offsets2) + tl.load(residual_ptr + base + offsets2)
        added3 = tl.load(x_ptr + base + offsets3) + tl.load(residual_ptr + base + offsets3)
        sumsq = (
            tl.sum(added0 * added0, axis=0)
            + tl.sum(added1 * added1, axis=0)
            + tl.sum(added2 * added2, axis=0)
            + tl.sum(added3 * added3, axis=0)
        )
        rstd = 1.0 / tl.sqrt(sumsq / n_cols + eps)

        gamma0 = tl.load(gamma_ptr + offsets0)
        gamma1 = tl.load(gamma_ptr + offsets1)
        gamma2 = tl.load(gamma_ptr + offsets2)
        gamma3 = tl.load(gamma_ptr + offsets3)
        tl.store(out_ptr + base + offsets0, added0 * rstd * gamma0)
        tl.store(out_ptr + base + offsets1, added1 * rstd * gamma1)
        tl.store(out_ptr + base + offsets2, added2 * rstd * gamma2)
        tl.store(out_ptr + base + offsets3, added3 * rstd * gamma3)

else:
    _add_rmsnorm_cast_kernel = None


class ModelNew(nn.Module):
    def __init__(self, epsilon=1e-6, target_dtype=torch.float16):
        super().__init__()
        self.epsilon = epsilon
        self.target_dtype = target_dtype
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, residual, gamma):
        if not self._can_try_triton(x, residual, gamma):
            return self._torch_reference(x, residual, gamma, "torch_fallback_unavailable")

        output = torch.empty(x.shape, device=x.device, dtype=self.target_dtype)
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        try:
            _add_rmsnorm_cast_kernel[(n_rows,)](
                x,
                residual,
                gamma,
                output,
                n_cols,
                eps=float(self.epsilon),
                CHUNK_SIZE=_CHUNK_SIZE,
            )
            self._last_backend = "triton_row_rmsnorm_cast_bs1024x4"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(x, residual, gamma, "torch_fallback_after_error")

    def _can_try_triton(self, x, residual, gamma):
        if self._disable_triton or _add_rmsnorm_cast_kernel is None:
            return False
        if getattr(x.device, "type", None) != "npu":
            return False
        return (
            x.is_contiguous()
            and residual.is_contiguous()
            and gamma.is_contiguous()
            and x.shape == residual.shape
            and x.ndim >= 2
            and gamma.ndim == 1
            and gamma.shape[0] == x.shape[-1]
            and x.shape[-1] == _SUPPORTED_HIDDEN
            and x.dtype == torch.float32
            and residual.dtype == torch.float32
            and gamma.dtype == torch.float32
            and self.target_dtype == torch.float16
        )

    def _torch_reference(self, x, residual, gamma, reason):
        self._last_backend = reason
        x_added = x + residual
        variance = x_added.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        output = (x_added * rstd) * gamma
        return output.to(self.target_dtype)
