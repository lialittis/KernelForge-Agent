# Roadmap

This roadmap follows the SketchSkill-AKG proposal. Update it after major
experiment cycles.

## Milestone 1: Environment And Benchmark Reproduction

Deliverables:

- completed `docs/benchmark_spec.md`
- Ascend 910 environment notes
- AKG Agents setup notes
- official benchmark source location and commit
- baseline run for at least one simple operator
- initial benchmark classification table

Exit criteria:

- A clean environment can run one official benchmark baseline and produce
  correctness plus performance output.

## Milestone 2: OpSpec And Benchmark Parser

Deliverables:

- parser for `akg_kernels_bench_lite` or confirmed official task format
- structured OpSpec schema
- parsed examples under `benchmarks/parsed/`
- tests for representative task parsing

Exit criteria:

- At least three official tasks can be converted into stable OpSpec records.

## Milestone 3: NPU-Aware Sketch

Deliverables:

- Sketch schema
- category-specific Sketch templates
- validator for required Sketch fields
- examples for elementwise, broadcast, and reduction tasks

Exit criteria:

- An OpSpec can be converted into a valid Sketch for at least three operator
  categories.

## Milestone 4: Triton-Ascend Generation Loop

Deliverables:

- AKG Agents integration path
- Triton-Ascend candidate generation
- prompt templates
- saved candidate metadata
- compile/run wrapper

Exit criteria:

- The system can generate and evaluate at least one Triton-Ascend candidate
  from OpSpec plus Sketch.

## Milestone 5: Correctness Repair And Pass@4

Deliverables:

- correctness validator
- Pass@1 and Pass@4 reporting
- error classifier
- Repair Agent routing rules
- bad-to-good repair records

Exit criteria:

- For a fixed benchmark subset, the project reports Pass@1, Pass@4, failure
  categories, and repair iteration counts.

## Milestone 6: Skill Library Retrieval And Write-Back

Deliverables:

- populated category `SKILL.md` files
- Skill Retriever
- Skill Writer
- experiment-to-skill promotion workflow

Exit criteria:

- Generation and repair prompts include traceable retrieved skills, and
  successful repairs update the relevant skill files.

## Milestone 7: Hardware Profiling And Search

Deliverables:

- Ascend 910 profiling runner
- latency and throughput result parser
- search over tile/core/vector/unroll/double-buffer knobs
- performance comparison reports

Exit criteria:

- Correct candidates can be profiled and optimized reproducibly on Ascend 910.

## Milestone 8: Multi-Backend Enhancement

Deliverables:

- selected TileLang-Ascend experiments
- selected Ascend C experiments
- CUDA/Triton-to-Sketch migration notes
- comparison against the Triton-Ascend main path

Exit criteria:

- At least one representative operator has a documented multi-backend
  comparison or migration case study.

## Milestone 9: Submission And Report

Deliverables:

- prototype code
- benchmark run scripts
- correctness and performance results
- Pass@1/Pass@4 summary
- NPU-aware Sketch templates
- Skill Library snapshot
- reproduction documentation
- final proposal/report and PR materials

Exit criteria:

- Results are reproducible from documented commands and experiment records.

