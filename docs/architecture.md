# Architecture

ViPanelTR is organized as one installable package whose modules communicate
through explicit Python interfaces.

## Package boundaries

- `vipaneltr.data` owns dataset validation, loading, HTML parsing,
  normalization, and table representation. Other packages do not read raw
  dataset files directly.
- `vipaneltr.system` owns agents, prompts, and the investigation, self-review,
  peer-review, consensus, and answer-formatting phases.
- `vipaneltr.evaluation` owns prediction/reference alignment and all supported
  metrics and grouped analyses.
- `vipaneltr.baseline` owns the zero-shot comparison pipeline. It consumes the
  same data representation used by the main system.
- `vipaneltr.utils` owns logging, retry, JSON parsing, traces, and artifact
  helpers shared across system components.
- `vipaneltr.cli` wires public commands to those packages and contains no
  provider-specific model implementation.

## Data flow

```text
data/open_vitabqa
        |
        v
validate -> load -> parse/normalize -> table representation
                                      |                 |
                                      v                 v
                              multi-agent system   zero-shot baseline
                                      |                 |
                                      +--------+--------+
                                               v
                                    results and usage metadata
                                               |
                                               v
                                alignment -> metrics -> report
```

`prepare-data` validates all identifiers before a paid inference command can
begin. `infer` and `baseline` write isolated run artifacts under `outputs/`.
`evaluate` aligns records by `qa_id`, so prediction order does not affect the
report.

## Configuration and failure behavior

Defaults are resolved from the installed repository root rather than the
caller's current directory. YAML configuration is optional and CLI values take
precedence. Provider keys are read from the environment; missing keys are
reported before model construction without printing secret values.

Dataset schema errors, duplicate IDs, missing table references, invalid
prediction files, and unavailable credentials return a non-zero CLI status with
an actionable message. Automated tests exercise only offline code paths.
