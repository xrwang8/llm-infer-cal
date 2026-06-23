# ADR-001: ModelScope Integration Strategy

**Status:** Accepted

## Context

`llm-infer-cal` supports both HuggingFace and ModelScope as model metadata
sources. The Rust core only needs metadata operations:

- model metadata
- repository file list and sizes
- `config.json`
- small `safetensors` header reads

## Decision

Use ModelScope's REST endpoints directly through the existing Rust HTTP client
layer. Do not add the ModelScope SDK as a runtime dependency.

## Rationale

- The required surface area is small and metadata-only.
- Direct REST keeps installation lightweight.
- The implementation can map ModelScope errors into the same internal error
  types used by HuggingFace.
- Authentication stays simple through `MODELSCOPE_API_TOKEN` or
  `MODELSCOPE_TOKEN`.

## Consequences

- CLI users choose the source with `--source huggingface|modelscope`.
- Cache keys include the selected source.
- If ModelScope changes response envelope fields, the parser in the Rust
  ModelScope source is the single place to update.
