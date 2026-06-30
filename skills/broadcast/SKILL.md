# Broadcast Skill

## Applicability

Use for operators where inputs have different shapes and output indexing must
map dimensions according to broadcasting rules.

## Non-Applicability

Do not use for pure elementwise same-shape operations or for reductions.

## Shape And Dtype Constraints

- Record full input and output shapes.
- Identify singleton dimensions.
- Confirm whether leading dimensions are implicit.
- Track dtype conversion and output dtype.

## Recommended Sketch Focus

- index mapping from output coordinates to each input
- broadcasted stride behavior
- boundary mask strategy
- contiguous output writes
- avoidance of repeated expensive index calculations when possible

## Common Failures

- incorrect leading-dimension broadcast
- wrong singleton-dimension indexing
- out-of-bounds input reads
- shape-specific code that fails on another test case

## Profiling And Tuning Notes

- Prefer output-contiguous parallelization.
- Cache or simplify index expressions for repeated broadcast dimensions.
- Tune only after multiple shape cases pass.

## Bad-To-Good Cases

No recorded cases yet.

