# Ascend Debug Skill

## Applicability

Use for compile errors, runtime errors, API incompatibilities, shape/dtype
failures, and environment issues on Ascend NPU.

## Error Categories

- syntax or API error
- missing include/import/dependency
- unsupported backend operation
- type mismatch
- shape mismatch
- out-of-bounds access
- runtime launch failure
- environment or version mismatch

## Routing Rules

- Syntax/API/type errors usually go back to Code Agent.
- Shape/broadcast/layout/tile-plan errors usually go back to Sketch Agent.
- Environment errors should be separated from generation failures.
- Numerical errors should involve dtype, boundary, mask, and reduction checks.

## Log Summary Format

```text
error_category:
first_error_line:
likely_owner:
minimal_context:
suggested_fix:
```

## Bad-To-Good Cases

No recorded cases yet.

