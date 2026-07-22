# ViPanelTR

ViPanelTR is a multi-agent system for Vietnamese table question answering. This
repository packages the reasoning pipeline, Open-ViTabQA preprocessing and
evaluation, a zero-shot baseline, and a pinned copy of the official dataset in
one installable Python project.

## Repository layout

```text
src/vipaneltr/system/      Multi-agent investigation and review pipeline
src/vipaneltr/data/        Dataset loading, validation, parsing, normalization
src/vipaneltr/evaluation/  EM, F1, ROUGE-1, METEOR and grouped analyses
src/vipaneltr/baseline/    Zero-shot comparison runner
data/open_vitabqa/         Official pinned dataset and its MIT license
scripts/                   Dataset and metric utilities
tests/                     Offline unit and integration tests
outputs/                   Generated runs (ignored by Git)
```

See [docs/architecture.md](docs/architecture.md) for the package boundaries and
data flow, and [MIGRATION.md](MIGRATION.md) for old-to-new path mappings.

## Installation

Python 3.10 or newer is required. From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

Optional evaluation dependencies can be installed with:

```powershell
python -m pip install -e ".[dev,eval]"
```

## Configuration

Copy `.env.example` to `.env` for local use and set only the providers you
need. Never commit `.env`.

Supported variables include `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `FPT_AI_MARKETPLACE`, and
`SEALION_API_KEY`. CLI arguments override YAML configuration values.

Real inference sends table/question content to the selected model provider and
may incur usage charges. `prepare-data`, evaluation, and the automated tests do
not call paid APIs.

## Validate the dataset

```powershell
vipaneltr prepare-data
```

The command checks the four required JSON files, duplicate identifiers, and
every QA-to-table reference before inference begins.

## Run ViPanelTR inference

```powershell
vipaneltr infer `
  --qas data/open_vitabqa/qas_test.json `
  --model openai/gpt-4o-mini `
  --run-id gpt4o-mini `
  --n 5
```

Artifacts are written below `outputs/<run-id>/`, including results, metadata,
configuration, token totals, and provider-reported or estimated cost.

## Evaluate predictions

```powershell
vipaneltr evaluate `
  --preds outputs/gpt4o-mini/results.json `
  --qas data/open_vitabqa/qas_test.json `
  --output-dir outputs/gpt4o-mini/eval
```

The detailed report is written to `outputs/gpt4o-mini/eval/evaluation.json`.

## Run the zero-shot baseline

```powershell
vipaneltr baseline `
  --qas data/open_vitabqa/qas_test.json `
  --tables data/open_vitabqa/table.json `
  --model openai/gpt-4o-mini `
  --n 5
```

The baseline reuses the same data representation and usage accounting policy as
the multi-agent system.

## Development and testing

```powershell
python -m pytest
python -m compileall -q src/vipaneltr
```

Tests use small local fixtures and fake/offline paths; no provider credential is
required.

## Dataset provenance and citation

The bundled dataset is pinned to
[`DuzDao/Open-ViTabQA`](https://github.com/DuzDao/Open-ViTabQA) commit
`3e027061117d2c98e6131034f91f48c75165fd71`. Its original MIT license and
copyright notice are preserved in `data/open_vitabqa/LICENSE`.

When using the dataset, cite:

> Dung Hoang Dao, Ngan Thi-Kim Huynh, Khanh Quoc Tran, and Kiet Van Nguyen.
> “Open-ViTabQA: A novel benchmark for Vietnamese question answering on open
> domain wikipedia table.” *Knowledge-Based Systems* 330 (2025), 114391.
> <https://doi.org/10.1016/j.knosys.2025.114391>

## License

ViPanelTR source code is distributed under the root MIT License. The bundled
Open-ViTabQA dataset retains its separate upstream MIT notice.
