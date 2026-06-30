# CUDA To Ascend Migration Skill

## Applicability

Use when borrowing algorithmic or performance structure from CUDA/Triton
kernels for Ascend NPU generation.

## Migration Principle

Do not translate CUDA/Triton code line by line. Extract portable structure into
NPU-aware Sketch first, then lower to the target Ascend backend.

## Portable Patterns

- tiling strategy
- reduction tree structure
- fusion opportunity
- memory coalescing idea
- data reuse plan
- boundary handling
- accumulation dtype choice

## Non-Portable Risks

- GPU warp assumptions
- CUDA shared memory assumptions
- backend-specific synchronization
- incompatible memory hierarchy assumptions
- unsupported intrinsic/API usage

## Sketch Mapping

```text
CUDA/Triton implementation
-> algorithm and tiling summary
-> NPU-aware Sketch
-> Triton-Ascend / TileLang-Ascend / Ascend C candidate
-> compile, verify, profile
```

## Bad-To-Good Cases

No recorded cases yet.

