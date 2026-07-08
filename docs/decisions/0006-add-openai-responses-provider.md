# 0006: Add OpenAI Responses Provider Behind The Provider Boundary

Date: 2026-07-08

## Status

Accepted

## Context

The replay provider proves the agent-generation workflow without credentials,
but the project also needs a first live LLM adapter so generated candidates can
come from a real code/reasoning model. The adapter must not make tests depend on
network access or checked-in secrets.

## Decision

Add an `openai` provider that calls the OpenAI Responses API through the same
`ProviderRequest` and `ProviderResponse` interface used by `replay`.

Configuration is environment-driven:

- `OPENAI_API_KEY` for credentials.
- `KERNEL_FORGE_OPENAI_MODEL` or `OPENAI_MODEL` for model selection.
- Optional timeout, max-output-token, temperature, and endpoint overrides.

Unit tests inject a fake HTTP poster to validate request construction and
response parsing without real credentials or network calls.

## Consequences

- Live model experiments can use the existing OpSpec, Sketch, skill retrieval,
  prompt, candidate, submission, and experiment-record pipeline.
- `replay` remains the deterministic regression provider.
- Credentials stay outside repository state and test fixtures.
- Model comparison stays provider-agnostic because model name and provider
  metadata are captured in generated experiment records.
