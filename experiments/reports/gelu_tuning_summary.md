# GELU Tuning Summary

Date: 2026-07-03

## Current Decision

Stop GELU-only candidate iteration for now. The best tracked real
Triton-Ascend candidate is `gelu_triton_v13`:

| Candidate | Result | Speedup | Score | Note |
| --- | --- | ---: | ---: | --- |
| `gelu_triton_v7` | pass | `0.0728x` | `4.37` | first pure Triton correctness pass |
| `gelu_triton_v8` | pass | `0.2856x` | `17.13` | block size `4096` |
| `gelu_triton_v9` | pass | `0.4869x` | `29.22` | block size `8192` |
| `gelu_triton_v10` | pass | `0.5635x` | `33.81` | block size `16384` |
| `gelu_triton_v11` | compile fail | null | null | UB overflow at block size `32768` |
| `gelu_triton_v12` | pass | `0.5764x` | `34.59` | block size `24576` |
| `gelu_triton_v13` | pass | `0.6059x` | `36.35` | best tracked candidate, `16384 x 2` chunks |
| `gelu_triton_v14` | pass | `0.5875x` | `35.25` | three chunks regressed |
| `gelu_triton_v15` | pass | `0.5858x` | `35.15` | `24576 x 2` regressed |
| `gelu_triton_v16` | pass | `0.5373x` | `32.24` | `tl.sigmoid` lowering regressed |
| `gelu_triton_v17` | compile fail | null | null | non-constexpr global in Triton JIT |
| `gelu_triton_v18` | pass | `0.5764x` | `34.58` | inline `exp2` lowering regressed |

## Lessons

- Backend probes are required before trusting benchmark speedup; fallback
  results are not custom-kernel results.
- For this benchmark, GELU numerics are sensitive to relative error near zero.
- Increasing tile size helped until the UB limit; block size `32768` overflowed
  UB, while `24576` compiled.
- Sequential chunks per program improved the best single-block candidate, but
  too many chunks reduced parallelism.
- `gelu_triton_v13` is useful as a tuning case study, but it remains slower
  than the framework baseline. The project should now move to the broader T1
  non-matmul pipeline.
