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

No recorded cases yet.

