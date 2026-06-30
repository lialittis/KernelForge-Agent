# Active Tasks

Keep this file short. It tracks the next concrete work items across machines
and agents.

## Current Priorities

1. Complete `docs/benchmark_spec.md` from official competition and benchmark
   materials.
2. Obtain the official `akg_kernels_bench_lite` source and record its URL,
   commit, layout, and commands.
3. Reproduce one official baseline task on Ascend 910.
4. Save raw official sample tasks under `benchmarks/raw/`.
5. Create at least one parsed OpSpec example under `benchmarks/parsed/`.
6. Draft elementwise and broadcast Sketch examples.
7. Add the first experiment record under `experiments/runs/`.
8. Decide whether to commit `SketchSkill_AKG_项目书基础版.pdf`.

## Research Questions

- What exact task format does `akg_kernels_bench_lite` provide?
- Which AKG Agents entrypoints are expected for this competition?
- What Triton-Ascend version and APIs are available?
- What CANN and Ascend 910 environment are provided?
- What are the official rtol/atol or task-specific correctness tolerances?
- What exact metric determines ranking: latency, throughput, relative speedup,
  or a combined score?
- How is Pass@4 evaluated by the community benchmark?
- What is the required submission package layout?
- Which operators should form the first 3-5 task experiment subset?

## Coordination Rules

- Use branch-per-task for concurrent work.
- Update `docs/status.md` before handing off work.
- Add a decision record for non-trivial architecture or workflow changes.
- Do not overwrite another agent's unmerged work.
- Prefer append-only experiment records over editing old records, unless
  correcting an obvious metadata mistake.
- Promote reusable experiment lessons into `skills/`.

