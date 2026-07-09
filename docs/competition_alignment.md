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
| Operator generation | OpSpec -> NPU-aware Sketch -> Skill Retriever -> Code Agent -> Triton-Ascend `ModelNew` | Designed; full Lite benchmark OpSpec/Sketch coverage is available for all 13 cases; deterministic/manual candidates exist for a priority subset | Implement generalized live Code Agent and repair loop |
| AKG/open-source base | AKG Agents, AKG Bench Lite, Triton-Ascend, optional TileLang-Ascend/Ascend C | AKG Bench Lite submodule and runner are integrated | Integrate AKG Agents generation path beyond standalone runner |
| Benchmark comparison | Official `akg_kernels_bench_lite` registry, parsed OpSpecs, official runner, result importer | 13 cases registered; 13 OpSpecs parsed; priority Pass@4 and reference reports recorded | Keep rerunning final reports under the updated AKG pin before final claims |
| Correctness on NPU | Official benchmark runner, backend probe, correctness result import, Pass@1/Pass@4 tracking | GELU verified on Ascend worker; result importer implemented | Generalize candidate batch runner and Pass@N calculator |
| Performance validation | Official latency/speedup/score, future profiling/search agent | GELU performance case study recorded; v13 best tracked GELU | Add throughput/profile fields and multi-operator performance report |
| AI technical design | Architecture, workflow, roadmap, decisions, final product/LLM boundary | Core docs and `docs/technical_design.md` exist; decision 0004 defines LLM boundary | Polish final wording after PR link exists |
| Agent workflow | Spec Agent, Sketch Agent, Skill Retriever, Code Agent, Verify/Repair Agent, Profiler/Search Agent, Skill Writer | Architecture defined; deterministic parser/sketch pieces, replay provider, OpenAI provider adapter, and prompt flow implemented | Configure model provider and run live generation |
| RAG/Skill Library | Local `skills/` plus experiment-driven write-back | Skill files, prompt templates, retrieval metadata, and positive/negative benchmark lessons exist | Add more skill write-back after new live/repair runs |
| AI-assisted assets | `AGENTS.md`, `skills/`, experiment records, prompts | Active project memory, prompts, and experiment records are committed | Keep package materials synchronized with final PR/email |

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
- Stage 2 OpSpec parser: complete for all 13 Lite benchmark cases.
- Stage 3 Sketch templates: complete for all currently parsed Lite benchmark
  cases, including matmul-like and MoE top-k softmax cases.
- Stage 4 Triton-Ascend generation loop: deterministic replay path exists;
  manual candidates and reports cover a priority subset.
- Stage 5 Pass@N correctness loop: implemented for generated/manual candidate
  batches through official runner reports.
- Stage 6 Skill/RAG assets: skills and prompt templates exist, with
  experiment-driven lessons promoted.
- Stage 7 performance search: demonstrated manually on GELU, not generalized.

The next required technical step before live AI credentials is to strengthen
validators, package hygiene, and deterministic replay/reference evidence. Once
credentials exist, run AKG Agents full comparison and live provider Pass@4
generation using the same OpSpec/Sketch/Skill pipeline.
