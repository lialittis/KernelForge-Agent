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

### Separate Backend Proof From Benchmark Pass

`gelu_triton_v1` initially passed the official benchmark with a small speedup,
but the backend probe showed:

```text
last_backend: torch_fallback_after_error
```

Treat this as an environment/backend-support finding, not a custom kernel
result. A benchmark pass is only a custom-kernel pass when the candidate has
also shown that the intended backend launched.

Recommended loop:

- Run the backend probe for candidate-specific status.
- Run the official benchmark for score and latency.
- Import the benchmark JSON into an experiment record.
- Record fallback, compile, runtime, correctness, and performance outcomes as
  separate fields.

### Reference Pre-Eval Can Reveal Environment Failures

Observed in the remaining-reference pre-evaluation run for AKG Bench Lite:

```text
t3/causal_conv1d: official reference failed in F.conv1d before candidate
comparison because CANN/TBE initialization could not import Python module
`tbe`.
```

Evaluation rule:

- When the official reference itself fails, categorize the case as an
  environment or benchmark-runtime failure, not as a candidate correctness
  failure.
- Keep the JSON result and traceback path in the experiment record so the
  environment can be repaired or documented later.
- Continue running independent single-case teams so one environment failure
  does not prevent measurements for other operators.

### Rebaseline After Runner Updates

When the benchmark runner changes, rerun baseline reports before using them as
evidence for new generation work.

Observed in the AKG Bench Lite update from
`bea77cb38db5713056a7e06e5e8a0cbe9d26954b` to
`47aa428fcdc8c68f78d331dc578bc6c74fb9d91d`: operator files were unchanged,
but standalone `run_bench.py` changed correctness protocol to independent
seeded reference/solution inputs, three correctness trials, and NaN/Inf
rejection.

Evaluation rule:

- Treat pre-update benchmark reports as historical, not active baselines.
- Write an append-only replacement report that names the prior report it
  supersedes.
- Compare pass/fail status and score deltas, because timing noise or stricter
  correctness can change next-step priorities even when the operator files did
  not change.
