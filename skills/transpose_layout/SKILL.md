# Transpose And Layout Skill

## Applicability

Use for transpose, reshape-with-layout-effect, layout conversion, permutation,
and operators dominated by non-contiguous reads or writes.

## Non-Applicability

Do not use for semantic reductions or arithmetic-only elementwise operators.

## Shape And Dtype Constraints

- Record input layout and output layout.
- Record permutation order.
- Distinguish metadata-only reshape from real memory reorder.
- Track alignment and tail behavior.

## Recommended Sketch Focus

- input/output index mapping
- read-contiguous versus write-contiguous choice
- tile reorder plan
- boundary mask
- bank/conflict risk notes when known

## Common Failures

- wrong permutation order
- treating layout conversion as metadata-only reshape
- non-contiguous indexing bugs
- poor performance from scattered reads and writes

## Profiling And Tuning Notes

- Start with correctness-oriented explicit index mapping.
- Tune tile shapes to improve locality.
- Compare read-contiguous and write-contiguous plans.

## Bad-To-Good Cases

No recorded cases yet.

