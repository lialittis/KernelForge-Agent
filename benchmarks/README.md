# Benchmarks

This directory stores benchmark inputs and parsed benchmark metadata.

## Layout

```text
benchmarks/
  raw/
    # official benchmark tasks or small copied samples
  parsed/
    # OpSpec JSON/YAML records derived from raw tasks
```

Do not commit large external benchmark repositories here. If the official
benchmark is large, record its clone URL, commit, and local path in
`docs/benchmark_spec.md`, and commit only small representative samples when
allowed.

## Rules

- Preserve raw official samples exactly when committing them.
- Store generated OpSpec files under `benchmarks/parsed/`.
- Every parsed OpSpec should link back to its raw source.
- Every experiment record should reference the OpSpec it used.

