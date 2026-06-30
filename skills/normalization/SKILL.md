# Normalization Skill

## Applicability

Use for layer norm, batch norm-like, RMS norm-like, mean/variance
normalization, and operators that combine reductions with elementwise
post-processing.

## Non-Applicability

Do not use for simple reductions without normalization or for pure elementwise
operators.

## Shape And Dtype Constraints

- Record normalized axes.
- Record epsilon behavior.
- Record accumulation dtype.
- Record broadcast shape for scale and bias parameters.

## Recommended Sketch Focus

- reduction plan for statistics
- accumulation dtype
- reuse of mean/variance or RMS terms
- broadcast plan for scale/bias
- numerical stability
- multi-stage pipeline if needed

## Common Failures

- wrong normalized axes
- missing epsilon
- dtype instability
- incorrect parameter broadcasting
- multiple-pass implementation with inconsistent indexing

## Profiling And Tuning Notes

- Prioritize correctness and numerical stability first.
- Profile whether statistics computation or final elementwise pass dominates.
- Consider fusion only after each component is validated.

## Bad-To-Good Cases

No recorded cases yet.

