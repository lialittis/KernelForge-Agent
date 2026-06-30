# Roadmap

This roadmap should be updated after major experiment cycles.

## Milestone 1: Benchmark Understanding

Deliverables:

- `docs/benchmark_spec.md`
- collected official sample tasks
- initial notes on supported runtimes, build commands, correctness rules, and
  performance metrics

Exit criteria:

- A human or agent can explain exactly what input a benchmark task provides and
  what output a candidate implementation must produce.

## Milestone 2: Manual Baseline

Deliverables:

- one manually written operator implementation
- compile command
- run command
- correctness checker
- performance measurement script or command

Exit criteria:

- The baseline can be built, executed, validated, and profiled from a clean
  checkout.

## Milestone 3: Automated Harness

Deliverables:

- benchmark task parser
- candidate workspace layout
- compile/run wrapper
- validator
- profiler
- structured result file writer

Exit criteria:

- A candidate can be evaluated by a single command and produce a structured
  result.

## Milestone 4: Single-Shot Generation

Deliverables:

- generation prompt template
- model/agent invocation wrapper
- saved candidate code
- saved prompt and response metadata

Exit criteria:

- The system can generate one candidate from one benchmark task and evaluate it
  through the harness.

## Milestone 5: Feedback Repair

Deliverables:

- repair prompt template
- compiler error summarization
- runtime error summarization
- numerical mismatch summarization
- repair iteration tracking

Exit criteria:

- Failed candidates can be repaired automatically for a bounded number of
  iterations, with each attempt recorded.

## Milestone 6: Multi-Candidate Pass@N

Deliverables:

- candidate batch generation
- isolated candidate workspaces
- Pass@1, Pass@3, and Pass@5 reporting

Exit criteria:

- The project can report Pass@N for a fixed benchmark subset.

## Milestone 7: Skill Library Retrieval

Deliverables:

- initial `skills/` content
- retriever/indexer
- retrieval logging in experiment records

Exit criteria:

- Generation and repair prompts include traceable retrieved context.

## Milestone 8: Performance Optimization

Deliverables:

- optimization prompt template
- performance regression guard
- comparison against baseline or previous best

Exit criteria:

- Correct candidates can be optimized and compared reproducibly.

## Milestone 9: Final Evaluation

Deliverables:

- full benchmark result summary
- ablation experiments
- final report
- reusable Skill Library snapshot

Exit criteria:

- Results are reproducible from documented commands and experiment records.

