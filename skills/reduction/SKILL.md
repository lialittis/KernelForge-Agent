# Reduction Skill

## Applicability

Use for operators that reduce one or more axes using sum, max, min, mean,
variance-like, or similar associative patterns.

## Non-Applicability

Do not use as the primary skill for pure elementwise, broadcast-only, or layout
transpose operators.

## Shape And Dtype Constraints

- Record reduction axes and keepdims behavior.
- Record accumulation dtype separately from input/output dtype.
- Record whether reduction axis is contiguous.
- Capture tolerance requirements because reductions amplify numerical error.

## Recommended Sketch Focus

- reduction axis mapping
- tile plan for partial reductions
- accumulation dtype
- parallel reduction strategy
- boundary mask for tail elements
- final writeback shape

## Common Failures

- wrong reduction axis
- missing keepdims handling
- accumulation dtype too narrow
- incorrect tail handling
- race or duplicate accumulation in parallel reduction

## Profiling And Tuning Notes

- Tune tile size and parallel axis after correctness.
- Compare contiguous versus non-contiguous reduction behavior.
- Watch UB pressure and partial-reduction writeback overhead.

## Bad-To-Good Cases

No recorded cases yet.

