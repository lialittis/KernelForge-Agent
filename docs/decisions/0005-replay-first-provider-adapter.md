# 0005: Build Replay Provider Before Live LLM Provider

Date: 2026-07-08

## Status

Accepted

## Context

The deterministic T1 non-matmul loop is now stable enough to start the
agent-generation path. The project needs a pluggable provider boundary, but a
live LLM provider would make tests depend on credentials and network behavior.

## Decision

Implement a deterministic `replay` provider first. It uses the same provider
request, prompt rendering, skill retrieval, generated candidate metadata, and
submission materialization path that a live LLM provider will use, but returns
known candidate templates from the repository.

The first replay target is `t1/sigmoid_scale_sum` Pass@4 because it already has
a positive Triton-Ascend trajectory and official benchmark evidence.

## Consequences

- The provider interface can be tested without API keys.
- Generated experiment records can already capture provider, model, prompt
  version, candidate index, retrieved skills, and replay source path.
- Live Codex/OpenAI/local providers can be added behind the same interface
  after the workflow shape is validated.
- Replay generation is not a model-quality claim; it is a pipeline and metadata
  validation step.
