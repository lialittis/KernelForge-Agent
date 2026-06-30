# Benchmark Evaluation Skill

## Applicability

Use for benchmark setup, OpSpec extraction, candidate evaluation, correctness
comparison, performance measurement, and experiment recording.

## Benchmark Checklist

- Record official benchmark source URL and commit.
- Record environment versions.
- Preserve raw sample tasks when allowed.
- Generate OpSpec records from raw tasks.
- Link every experiment to its OpSpec.
- Separate environment failures from candidate failures.

## Correctness Checklist

- Run official or equivalent reference implementation.
- Test all required shape/dtype cases.
- Record rtol/atol.
- Record max error and mean error.
- Record failed shapes and failure category.
- Report Pass@1 and Pass@4 for candidate batches.

## Performance Checklist

- Run only after correctness passes.
- Record warmup and repeat counts.
- Record mean and median latency.
- Record throughput when applicable.
- Compare against baseline.
- Save profiling artifact paths.

## Bad-To-Good Cases

No recorded cases yet.

