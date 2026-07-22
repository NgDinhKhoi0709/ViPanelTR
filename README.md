# ViPanelTR

**ViPanelTR: A Multi-Agent Framework for Vietnamese Table Question Answering**

ViPanelTR is a role-specialized multi-agent framework for Vietnamese table
question answering. It coordinates structural analysis, evidence verification,
answer synthesis, logical reasoning, and numerical reasoning through an early
answerability gate, bounded self-review, peer deliberation, consensus, and
answer normalization.

> **MAPR 2026:** The ViPanelTR paper has been accepted at the 2026
> International Conference on Multimedia Analysis and Pattern Recognition
> (MAPR 2026). See the
> [official list of accepted papers](https://mapr.uit.edu.vn/list-accepted-papers-mapr-2026)
> and [read the paper](docs/paper.pdf).

## Architecture

![ViPanelTR architecture](imgs/architecture.png)

The framework contains five role-conditioned agents:

- **Structuralist** reconstructs table structure and relevant regions.
- **Verifier** checks whether candidate claims are supported by table evidence.
- **Synthesizer** integrates findings into a candidate answer.
- **Logician** resolves constraints, comparisons, and logical conditions.
- **Calculator** handles numerical operations and validates calculations.

Each agent first investigates independently. The framework can return `NULL`
early when at least three of five agents identify the question as unsupported.
Otherwise, the agents self-review their drafts, deliberate with their peers,
reach consensus, and normalize the final Vietnamese answer.

## Results

ViPanelTR is evaluated on the official Open-ViTabQA test set using F1, Exact
Match (EM), ROUGE-1 (R1), and METEOR (MET). The complete overall results from
Table I of the paper are reproduced below.

| Category | Model | F1 | EM | R1 | MET |
|---|---|---:|---:|---:|---:|
| Prompt-based | LLaMA 2 | 4.20 | 6.00 | 8.80 | 5.80 |
| Prompt-based | Mistral 7B v0.3 | 34.90 | 35.00 | 44.80 | 32.80 |
| Prompt-based | Mistral Nemo 2407 | 35.50 | 35.60 | 45.20 | 33.50 |
| Prompt-based | LLaMA 3.1 8B | 56.44 | 35.38 | 45.54 | 42.89 |
| Prompt-based | Gemini 1.5 Pro | 59.80 | 60.80 | 71.90 | 49.80 |
| Prompt-based | Gemini 2.0 Flash Exp. | 60.50 | 60.20 | 70.10 | 50.00 |
| Prompt-based | GPT-4o mini | 64.57 | 42.44 | 55.64 | 53.17 |
| Prompt-based | Qwen3 8B | 62.58 | 40.46 | 56.62 | 51.98 |
| Fine-tuned | LLaMA 2 | 5.69 | 7.85 | 9.58 | 6.42 |
| Fine-tuned | TAPAS-Base | 30.62 | 30.29 | 41.10 | 28.27 |
| Fine-tuned | TAPAS-Large | 31.56 | 31.44 | 40.02 | 27.96 |
| Fine-tuned | LLaMA 3.1 | 37.73 | 36.89 | 51.17 | 38.54 |
| Fine-tuned | Mistral 7B v0.3 | 41.28 | 41.83 | 50.76 | 37.08 |
| Fine-tuned | ViT5-Base | 42.35 | 42.24 | 50.35 | 36.81 |
| Fine-tuned | Mistral Nemo 2407 | 43.11 | 42.14 | 53.71 | 40.38 |
| Fine-tuned | ViT5-Large | 45.22 | 45.13 | 51.87 | 37.12 |
| Non-LLM baselines | RGCN-RCI | 18.12 | 17.70 | 23.24 | 16.71 |
| Non-LLM baselines | KorWikiTQ | 46.00 | 46.00 | 52.90 | 37.80 |
| Multi-agent systems | CoAgt | 68.87 | 32.36 | 45.32 | 46.89 |
| Multi-agent systems | CoQ | 73.20 | 57.86 | 66.36 | 67.20 |
| **ViPanelTR (Ours)** | **LLaMA 3.1 8B** | **62.23** | **40.02** | **50.92** | **48.00** |
| **ViPanelTR (Ours)** | **Qwen3 8B** | **80.06** | **64.72** | **74.94** | **72.12** |
| **ViPanelTR (Ours)** | **GPT-4o mini** | **80.66** | **65.52** | **76.01** | **73.36** |

### Results at a glance

- **Best overall performance:** ViPanelTR with GPT-4o mini reaches **80.66
  F1**, **65.52 EM**, **76.01 ROUGE-1**, and **73.36 METEOR**.
- **Matched-backbone gains:** ViPanelTR improves F1 from 64.57 to 80.66 on
  GPT-4o mini (**+16.09**), from 62.58 to 80.06 on Qwen3 8B (**+17.48**), and
  from 56.44 to 62.23 on LLaMA 3.1 8B (**+5.79**).
- **Multi-agent comparison:** On GPT-4o mini, ViPanelTR exceeds CoAgt by 11.79
  F1 and CoQ by 7.46 F1. It also leads CoQ by 7.66 EM, 9.65 ROUGE-1, and 6.16
  METEOR.
- **Reasoning-intensive questions:** The largest consistent improvements occur
  on Yes/No, Calculate, and Multi-conditions questions.
- **Table robustness:** Improvements hold across normal, merged-header, and
  merged-value tables for all three ViPanelTR backbones.
- **Answerability and normalization:** The answerability gate improves
  unsupported-question detection, while answer normalization raises EM by
  11.39 points on GPT-4o mini and 6.55 points on Qwen3 8B.

## Repository layout

```text
src/vipaneltr/system/      Multi-agent investigation and review pipeline
src/vipaneltr/data/        Dataset loading, validation, parsing, normalization
src/vipaneltr/evaluation/  EM, F1, ROUGE-1, METEOR, and grouped analyses
src/vipaneltr/baseline/    Zero-shot comparison runner
data/open_vitabqa/         Open-ViTabQA data and its MIT license
docs/architecture.md       Package boundaries and data flow
docs/paper.pdf             ViPanelTR paper
imgs/architecture.png      Framework architecture
scripts/                   Dataset and metric utilities
tests/                     Offline unit and integration tests
outputs/                   Generated runs (ignored by Git)
```

See [docs/architecture.md](docs/architecture.md) for the package boundaries and
data flow.

## Installation

Python 3.10 or newer is required. From the repository root:

```powershell
python -m pip install -e ".[dev]"
```

Install the optional evaluation dependencies with:

```powershell
python -m pip install -e ".[dev,eval]"
```

## Configuration

Copy `.env.example` to `.env` for local use and set only the providers you
need. Never commit `.env`.

Supported variables include `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `FPT_AI_MARKETPLACE`, and
`SEALION_API_KEY`. CLI arguments override YAML configuration values.

Real inference sends table and question content to the selected model provider
and may incur usage charges. Data preparation, evaluation, and automated tests
do not call paid APIs.

## Usage

### Validate the dataset

```powershell
vipaneltr prepare-data
```

This command checks the four required JSON files, duplicate identifiers, and
every QA-to-table reference before inference begins.

### Run ViPanelTR inference

```powershell
vipaneltr infer `
  --qas data/open_vitabqa/qas_test.json `
  --model openai/gpt-4o-mini `
  --run-id gpt4o-mini `
  --n 5
```

Artifacts are written below `outputs/<run-id>/`, including results, metadata,
configuration, token totals, and provider-reported or estimated cost.

### Evaluate predictions

```powershell
vipaneltr evaluate `
  --preds outputs/gpt4o-mini/results.json `
  --qas data/open_vitabqa/qas_test.json `
  --output-dir outputs/gpt4o-mini/eval
```

The detailed report is written to
`outputs/gpt4o-mini/eval/evaluation.json`.

### Run the zero-shot baseline

```powershell
vipaneltr baseline `
  --qas data/open_vitabqa/qas_test.json `
  --tables data/open_vitabqa/table.json `
  --model openai/gpt-4o-mini `
  --n 5
```

The baseline uses the same data representation and usage-accounting policy as
the multi-agent system.

## Development and testing

```powershell
python -m pytest
python -m compileall -q src/vipaneltr
```

Tests use small local fixtures and offline model paths, so provider credentials
are not required.

## Paper citation

```bibtex
@inproceedings{nguyen2026vipaneltr,
  author    = {Nguyen, Dinh Khoi and Vo, Tuan Kiet and Dang, Van Thin},
  title     = {ViPanelTR: A Multi-Agent Framework for Vietnamese Table Question Answering},
  booktitle = {2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  year      = {2026}
}
```

## Dataset

This repository includes the
[Open-ViTabQA](https://github.com/DuzDao/Open-ViTabQA) benchmark under its MIT
License. When using the dataset, cite:

> Dung Hoang Dao, Ngan Thi-Kim Huynh, Khanh Quoc Tran, and Kiet Van Nguyen.
> “Open-ViTabQA: A novel benchmark for Vietnamese question answering on open
> domain Wikipedia tables.” *Knowledge-Based Systems* 330 (2025), 114391.
> <https://doi.org/10.1016/j.knosys.2025.114391>

## License

ViPanelTR source code is distributed under the root MIT License. Open-ViTabQA
retains the separate MIT notice in `data/open_vitabqa/LICENSE`.
