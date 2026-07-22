# ViPanel GitHub Repository Design

**Date:** 2026-07-22

## Objective

Create a self-contained, public-ready Python repository at `ViPanel_github/` by reorganizing the ViPanelTR system, the required Open-ViTabQA data-processing and evaluation code, the zero-shot baseline code, and the official Open-ViTabQA dataset. Preserve the existing source directories and exclude generated experiment results, caches, bytecode, and secrets.

## Confirmed Scope

The new repository will include:

- The ViPanelTR multi-agent inference system.
- Data loading, parsing, normalization, and table-representation code currently located in `Open_ViTabQA`.
- Evaluation metrics and reporting code currently located in `Open_ViTabQA` and ViPanelTR.
- Zero-shot baseline source code, without baseline predictions or evaluation outputs.
- The official Open-ViTabQA dataset files: `table.json`, `qas_train.json`, `qas_dev.json`, `qas_test.json`, the dataset README, guideline PDF, and upstream MIT license.
- Tests, documentation, GitHub Actions, packaging metadata, and safe configuration examples.

The new repository will exclude:

- `ViPanelTR/.env` and all credential values.
- Existing contents of `ViPanelTR/outputs/` and other generated run directories.
- `__pycache__`, `.pytest_cache`, `.pyc`, logs, temporary files, and local editor metadata.
- Unrelated systems and baselines outside the ViPanelTR scope.

## Source Provenance and Licensing

The dataset source is [DuzDao/Open-ViTabQA](https://github.com/DuzDao/Open-ViTabQA). The upstream repository publishes the dataset under the MIT License and identifies the copyright holder as Dao Hoang Dung, 2024.

The repository will:

- Include a root `LICENSE` for ViPanelTR source code using the existing MIT declaration from `ViPanelTR/pyproject.toml`.
- Preserve the upstream dataset license in `data/open_vitabqa/LICENSE`.
- Record the upstream repository URL and exact source commit in `data/open_vitabqa/README.md` during migration.
- Keep code and dataset attribution explicit instead of implying that all artifacts originated in the new repository.

## Repository Architecture

```text
ViPanel_github/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- data/
|   `-- open_vitabqa/
|       |-- LICENSE
|       |-- README.md
|       |-- TableQA_guideline.pdf
|       |-- qas_dev.json
|       |-- qas_test.json
|       |-- qas_train.json
|       `-- table.json
|-- docs/
|   `-- architecture.md
|-- outputs/
|   `-- .gitkeep
|-- scripts/
|-- src/
|   `-- vipaneltr/
|       |-- __init__.py
|       |-- cli.py
|       |-- config.py
|       |-- system/
|       |   |-- agents/
|       |   |-- core/
|       |   `-- prompts/
|       |-- data/
|       |-- evaluation/
|       |-- baseline/
|       `-- utils/
|-- tests/
|   |-- unit/
|   `-- integration/
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- MIGRATION.md
|-- README.md
`-- pyproject.toml
```

### Module Responsibilities

- `vipaneltr.system`: owns the ViPanelTR orchestration phases, agent implementations, prompts, self-review, peer-review, consensus, and answer formatting.
- `vipaneltr.data`: owns dataset schemas, loading, validation, parsing, normalization, and table representation. It contains the required code migrated from `Open_ViTabQA.preprocessing`.
- `vipaneltr.evaluation`: owns prediction/reference alignment, EM, F1, ROUGE, METEOR, answerability metrics, and metrics grouped by table type.
- `vipaneltr.baseline`: owns the zero-shot comparison runner and provider client. It consumes the same data and evaluation interfaces as the main system.
- `vipaneltr.cli`: exposes a single command-line entry point and contains command wiring only; business logic remains in the relevant package.
- `vipaneltr.utils`: contains shared JSON, logging, retry, trace, and artifact helpers that are not owned by a more specific module.

Imports such as `Open_ViTabQA.preprocessing.loader` and `Open_ViTabQA.evaluation` will be replaced by imports from `vipaneltr.data` and `vipaneltr.evaluation`. The installed package must not depend on an adjacent `Open_ViTabQA` directory or mutate `sys.path` to discover it.

## Command-Line Interface

The package will install a `vipaneltr` command with four subcommands:

- `vipaneltr prepare-data`: validate required dataset files, top-level JSON structure, unique QA identifiers, and all QA-to-table references.
- `vipaneltr infer`: run the multi-agent inference pipeline and write artifacts under `outputs/<run-id>/`.
- `vipaneltr evaluate`: align predictions and references, calculate metrics, and write evaluation artifacts to the selected output directory.
- `vipaneltr baseline`: run the zero-shot baseline using the same dataset loader, provider conventions, and evaluation contracts.

The migration may change the old command names and import paths because the selected goal is a modern unified package, not backward-compatible internal imports. `MIGRATION.md` will provide old-to-new command and path mappings.

## Data Flow

1. `prepare-data` loads and validates `table.json` and all QA splits before any paid model call is permitted.
2. `infer` loads the requested QA split and resolves every `table_id` through `vipaneltr.data`.
3. The data layer normalizes and converts raw table content into the representation consumed by `vipaneltr.system`.
4. The multi-agent system performs investigation, self-review, peer-review, consensus, and answer formatting.
5. The artifact layer writes per-question results plus run-level metadata, configuration, token totals, and estimated or provider-reported cost.
6. `evaluate` aligns results to ground truth through stable QA identifiers and writes aggregate and grouped metrics.
7. `baseline` reuses the same data and evaluation layers so baseline and ViPanelTR results are comparable.

## Configuration and Security

- The default dataset path will be `data/open_vitabqa`, resolved explicitly rather than through assumptions about the caller's current working directory.
- Output paths default to `outputs/` and remain configurable.
- YAML configuration remains optional, with explicit CLI arguments taking precedence.
- Provider credentials are read only from environment variables.
- `.env.example` lists supported variable names with empty values and safe comments.
- `.env`, secret-bearing variants, generated outputs, caches, bytecode, and local artifacts are ignored by Git.
- No real provider call is made by automated tests.
- A repository scan must confirm that credential values from the source `.env` do not appear anywhere in the destination.

## Error Handling

The CLI will fail before inference with a non-zero exit code and a concise actionable message when:

- A required dataset file is missing.
- JSON cannot be parsed or does not have the expected top-level structure.
- A QA identifier is duplicated.
- A QA references an unknown `table_id`.
- A selected model provider lacks its required environment credential.
- A predictions file cannot be aligned with the requested reference split.

Partial inference artifacts remain isolated under their run identifier. Existing run data is not silently overwritten unless an existing explicit resume/overwrite option permits it.

## Testing Strategy

### Unit tests

- Dataset file discovery and schema validation.
- QA-to-table reference validation.
- Parser, normalizer, and table representation behavior.
- Prediction/reference alignment and each metric wrapper.
- Configuration precedence and path resolution.
- Token and cost tracking.

### Integration tests

- Run one small in-memory fixture through inference with a fake LLM client.
- Evaluate the fake prediction without network access.
- Run the baseline pipeline with a fake provider response.
- Verify each CLI subcommand's help and representative success/error exit codes.

### Migration and isolation checks

- Search the destination for remaining `Open_ViTabQA` runtime imports and old path bootstrapping.
- Temporarily execute tests without the original sibling directories on `PYTHONPATH`.
- Compare copied dataset file hashes against the selected upstream/local source snapshot.
- Scan the destination for `.env`, Python caches, generated output files, and known source secret values.

## Continuous Integration

GitHub Actions will install the package and development dependencies, then run the offline test suite on Python 3.10, 3.11, and 3.12. CI must not require API credentials or download model weights.

## Documentation

`README.md` will cover:

- Project purpose and architecture.
- Installation with an editable development option.
- Dataset provenance, citation, and license.
- Environment-variable setup without credential values.
- Examples for `prepare-data`, `infer`, `evaluate`, and `baseline`.
- Output layout and reproducibility notes.

`MIGRATION.md` will map source locations and legacy commands to the new package layout. `docs/architecture.md` will describe package boundaries and the end-to-end data flow.

## Git Repository Policy

The migration will initialize a new Git repository inside `ViPanel_github/`. It will create a clean initial commit only after validation passes. Creating a GitHub remote repository or pushing commits is outside the current scope because no destination account or remote URL has been provided.

The existing `ViPanelTR` and `Open_ViTabQA` directories remain unchanged and serve as migration sources.

## Completion Criteria

The migration is complete when:

- `ViPanel_github/` is a standalone Git repository with the documented structure.
- `python -m pip install -e ".[dev]"` succeeds in a clean environment.
- The complete offline test suite passes on the local supported Python version.
- `vipaneltr --help` and all four subcommand help screens succeed.
- No runtime import or path dependency points to the original sibling projects.
- Dataset provenance, upstream MIT license, and source commit are recorded.
- No API key, `.env`, generated experiment output, cache, or bytecode is tracked.
- The original source directories have not been modified.
