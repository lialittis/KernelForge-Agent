# Matmul-Like Skill

## Applicability

Use for matrix multiplication, batched matmul, attention-like inner products,
and operators dominated by M/N/K tiled compute.

## Non-Applicability

Do not use for simple elementwise, broadcast, or layout-only operations.

## Shape And Dtype Constraints

- Record M, N, K dimensions and batch dimensions.
- Record transposition flags.
- Record accumulation dtype.
- Record output dtype and tolerance.

## Recommended Sketch Focus

- M/N/K tile plan
- data reuse strategy
- memory movement plan
- double buffering
- compute/memory overlap
- boundary mask for non-multiple tile sizes

## Common Failures

- wrong M/N/K interpretation
- missing transpose handling
- accumulation dtype mismatch
- incorrect boundary handling
- severe performance loss from poor data reuse

## Profiling And Tuning Notes

- Treat this as a later-stage category unless official benchmark requires it
  early.
- Compare against baseline carefully.
- Search tile sizes and double buffering only after correctness.

## Bad-To-Good Cases

No recorded cases yet.

