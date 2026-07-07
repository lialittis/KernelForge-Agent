# Competition Alignment

This document maps the competition task requirements to the SketchSkill-AKG
design and current implementation status. Use it as a checklist before adding
new features or preparing the final report.

## Required Tasks

The competition asks for:

1. Operator generation: use AKG or another open-source solution to build an
   automatic flow from problem description to executable operators, then
   compare on community-provided operator benchmarks.
2. Correctness validation: run generated operators on NPU hardware, compare
   against standard implementations, and compute Pass@N correctness metrics.
3. Performance validation: measure execution time, throughput, and related
   performance metrics, and compare with baseline implementations.
4. AI solution design and implementation: clearly explain the AI techniques
   used, such as fine-tuning, retrieval-augmented generation, or Agent
   workflows, and submit a complete technical design document.
5. AI-assisted work and reusable skills: use AI to assist the work and build a
   reusable skill library, including assets such as `SKILL.md` and prompts.

## Design Mapping

| Requirement | SketchSkill-AKG design | Current status | Next gap |
| --- | --- | --- | --- |
| Operator generation | OpSpec -> NPU-aware Sketch -> Skill Retriever -> Code Agent -> Triton-Ascend `ModelNew` | Designed; GELU manually explored; T1 non-matmul OpSpecs exist | Implement generalized Code Agent and Pass@4 generation |
| AKG/open-source base | AKG Agents, AKG Bench Lite, Triton-Ascend, optional TileLang-Ascend/Ascend C | AKG Bench Lite submodule and runner are integrated | Integrate AKG Agents generation path beyond standalone runner |
| Benchmark comparison | Official `akg_kernels_bench_lite` registry, parsed OpSpecs, official runner, result importer | 13 cases registered; 4 T1 non-matmul OpSpecs parsed | Add multi-case reports and Pass@N summaries |
| Correctness on NPU | Official benchmark runner, backend probe, correctness result import, Pass@1/Pass@4 tracking | GELU verified on Ascend worker; result importer implemented | Generalize candidate batch runner and Pass@N calculator |
| Performance validation | Official latency/speedup/score, future profiling/search agent | GELU performance case study recorded; v13 best tracked GELU | Add throughput/profile fields and multi-operator performance report |
| AI technical design | Architecture, workflow, roadmap, decisions, final product/LLM boundary | Core docs exist; decision 0004 defines LLM boundary | Add final `docs/technical_design.md` for submission |
| Agent workflow | Spec Agent, Sketch Agent, Skill Retriever, Code Agent, Verify/Repair Agent, Profiler/Search Agent, Skill Writer | Architecture defined; deterministic parser/sketch pieces implemented | Implement first pluggable LLM adapter and prompt flow |
| RAG/Skill Library | Local `skills/` plus experiment-driven write-back | Skill files exist; GELU lessons promoted | Add prompt templates and retrieval metadata in experiments |
| AI-assisted assets | `AGENTS.md`, `skills/`, experiment records, planned prompts | `AGENTS.md` and skills are active project memory | Add `prompts/` for Spec/Sketch/Code/Repair/Skill Writer |

## Evidence To Produce For Final Submission

- Technical design document covering the full AI/Agent method.
- Benchmark registry and OpSpec/Sketch examples.
- Generated candidate records with provider, model, prompt version, retrieved
  skills, candidate index, and repair iteration.
- Pass@1 and Pass@4 reports over at least one benchmark subset.
- Correctness result tables with max absolute and relative error.
- Performance result tables with latency, speedup, score, and throughput or
  profiling fields when available.
- Bad-to-good trajectories showing how failures became reusable skills.
- Skill Library snapshot and prompt assets.

## Current Stage Assessment

The project design matches the competition requirements. Implementation is in
the transition from deterministic benchmark infrastructure to generalized
agent-driven generation:

- Stage 1 benchmark/environment reproduction: mostly complete.
- Stage 2 OpSpec parser: complete for first T1 non-matmul subset.
- Stage 3 Sketch templates: started for elementwise and rowwise reduction.
- Stage 4 Triton-Ascend generation loop: manual GELU case study only.
- Stage 5 Pass@N correctness loop: designed, not yet generalized.
- Stage 6 Skill/RAG assets: started, prompt templates still missing.
- Stage 7 performance search: demonstrated manually on GELU, not generalized.

The next required step is the first non-GELU Pass@4 cycle, preferably
`t1/sigmoid_scale_sum`, using explicit model/provider metadata and reusable
skills.
