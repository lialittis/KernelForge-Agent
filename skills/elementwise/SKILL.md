# Elementwise Skill

## Applicability

Use for pointwise operators where each output element is computed from one or
more input elements with the same logical index.

## Non-Applicability

Do not use as the primary skill when the operator performs reduction, layout
reordering, matrix multiplication, or normalization over axes.

## Shape And Dtype Constraints

- Confirm whether all inputs share the same shape.
- If shapes differ, route to `broadcast` unless broadcasting is trivial and
  already represented in OpSpec.
- Track dtype conversion and accumulation behavior explicitly.

## Recommended Sketch Focus

- contiguous memory traversal
- vectorized element mapping
- tail mask handling
- simple GM/UB movement when relevant
- backend-safe expression lowering

## Common Failures

- missing tail mask
- dtype mismatch between input and output
- accidental scalar broadcasting
- unsupported math API in backend

## Profiling And Tuning Notes

- Start with simple contiguous access.
- Tune vector width and tile size only after correctness passes.
- Avoid overcomplicating the pipeline for very small shapes.

## Bad-To-Good Cases

### GELU Tanh-Approximate Stable Form

For `t1/gelu`, the NPU reference behaved consistently with tanh-approximate
GELU numerics. The cancellation-prone form:

```text
0.5 * x * (1 + tanh(u))
```

failed relative error in the far negative tail. The stable equivalent:

```text
x / (1 + exp(-2u))
```

passed correctness when implemented in Triton-Ascend. Use this lesson when an
elementwise activation has tiny reference outputs and the benchmark checks
relative error separately from absolute error.
