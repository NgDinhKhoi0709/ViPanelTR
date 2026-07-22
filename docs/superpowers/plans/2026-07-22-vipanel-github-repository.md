# ViPanel GitHub Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, public-ready `ViPanel_github` Git repository containing ViPanelTR, its data-processing and evaluation stack, baseline code, and the licensed Open-ViTabQA dataset.

**Architecture:** Consolidate the current `ViPanelTR` and required `Open_ViTabQA` modules under one `src/vipaneltr` package. Keep the official dataset as a licensed repository asset under `data/open_vitabqa`, expose one four-command CLI, and ensure all tests run offline without the source sibling directories.

**Tech Stack:** Python 3.10+, setuptools/PEP 621, pytest, PyYAML, BeautifulSoup/lxml, provider SDKs already used by ViPanelTR, GitHub Actions.

## Global Constraints

- Create the destination only at `D:\.UIT\KLTN\code\ViPanel_github`.
- Preserve `D:\.UIT\KLTN\code\ViPanelTR` and `D:\.UIT\KLTN\code\Open_ViTabQA` unchanged.
- Use a modern `src/vipaneltr` package; do not retain runtime imports from `Open_ViTabQA` or `paneltr_vitabqa`.
- Include baseline source code but no baseline predictions, evaluation outputs, or other generated experiment results.
- Include the official Open-ViTabQA dataset and preserve its upstream MIT license and attribution.
- Never copy `.env` or any credential value; tests and CI must not call paid APIs.
- Track `outputs/.gitkeep` only; ignore generated contents of `outputs/`.
- Support Python 3.10, 3.11, and 3.12 in CI.
- Do not create or push a GitHub remote without a destination URL/account from the user.
- Use UTF-8 for all text and JSON operations.

---

## File Map

### Source files migrated without behavioral changes first

| Source | Destination |
|---|---|
| `ViPanelTR/paneltr_vitabqa/agents/*.py` | `ViPanel_github/src/vipaneltr/system/agents/` |
| `ViPanelTR/paneltr_vitabqa/core/*.py` | `ViPanel_github/src/vipaneltr/system/core/` |
| `ViPanelTR/paneltr_vitabqa/prompts/*.py` | `ViPanel_github/src/vipaneltr/system/prompts/` |
| `ViPanelTR/paneltr_vitabqa/utils/*.py` | `ViPanel_github/src/vipaneltr/utils/` |
| `ViPanelTR/paneltr_vitabqa/config.py` | `ViPanel_github/src/vipaneltr/config.py` |
| `Open_ViTabQA/preprocessing/*.py` | `ViPanel_github/src/vipaneltr/data/` |
| `Open_ViTabQA/evaluation/*.py` | `ViPanel_github/src/vipaneltr/evaluation/` |
| `ViPanelTR/baseline/*.py` | `ViPanel_github/src/vipaneltr/baseline/` |
| `ViPanelTR/baseline/utils/*.py` | `ViPanel_github/src/vipaneltr/baseline/utils/` |
| `ViPanelTR/paneltr_vitabqa/cli.py` | `ViPanel_github/src/vipaneltr/cli.py` |
| `ViPanelTR/scripts/*.py` | `ViPanel_github/scripts/` |
| `ViPanelTR/utils/create_subset.py` | `ViPanel_github/scripts/create_subset.py` |

### New files with focused responsibility

- `src/vipaneltr/data/validation.py`: dataset discovery and referential-integrity validation.
- `src/vipaneltr/paths.py`: repository/data/output default path resolution.
- `tests/conftest.py`: small offline dataset and fake-provider fixtures.
- `tests/unit/test_data_validation.py`: dataset contract tests.
- `tests/unit/test_evaluation.py`: alignment and metric tests.
- `tests/unit/test_usage_tracking.py`: migrated cost/token tests.
- `tests/integration/test_cli.py`: four-command offline CLI smoke tests.
- `tests/integration/test_standalone.py`: import isolation and source-tree independence.
- `README.md`, `MIGRATION.md`, `docs/architecture.md`: public documentation.
- `.github/workflows/ci.yml`: Python 3.10–3.12 offline CI.

---

### Task 1: Initialize the repository and package contract

**Files:**
- Create: `ViPanel_github/.gitignore`
- Create: `ViPanel_github/.env.example`
- Create: `ViPanel_github/LICENSE`
- Create: `ViPanel_github/pyproject.toml`
- Create: `ViPanel_github/src/vipaneltr/__init__.py`
- Create: `ViPanel_github/src/vipaneltr/paths.py`
- Create: `ViPanel_github/tests/unit/test_package_contract.py`
- Create: `ViPanel_github/outputs/.gitkeep`
- Copy: `docs/superpowers/specs/2026-07-22-vipanel-github-repository-design.md` to `ViPanel_github/docs/superpowers/specs/`
- Copy: this plan to `ViPanel_github/docs/superpowers/plans/`

**Interfaces:**
- Produces: `vipaneltr.paths.repo_root() -> Path`, `dataset_dir() -> Path`, `outputs_dir() -> Path`.
- Produces: console entry point `vipaneltr = vipaneltr.cli:main`.

- [ ] **Step 1: Create the destination directories and initialize Git**

Run:

```powershell
New-Item -ItemType Directory -Force ViPanel_github/src/vipaneltr, ViPanel_github/tests/unit, ViPanel_github/tests/integration, ViPanel_github/outputs, ViPanel_github/docs/superpowers/specs, ViPanel_github/docs/superpowers/plans | Out-Null
git -C ViPanel_github init -b main
```

Expected: `ViPanel_github/.git` exists and the current branch is `main`.

- [ ] **Step 2: Write the failing package-contract test**

Create `tests/unit/test_package_contract.py`:

```python
from pathlib import Path

from vipaneltr.paths import dataset_dir, outputs_dir, repo_root


def test_repository_default_paths_are_stable():
    root = repo_root()
    assert root.name == "ViPanel_github"
    assert dataset_dir() == root / "data" / "open_vitabqa"
    assert outputs_dir() == root / "outputs"
    assert isinstance(root, Path)
```

- [ ] **Step 3: Run the test to verify the package is missing**

Run:

```powershell
$env:PYTHONPATH = "$PWD\ViPanel_github\src"
python -m pytest ViPanel_github/tests/unit/test_package_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vipaneltr.paths'`.

- [ ] **Step 4: Add package metadata, safe defaults, and path implementation**

Create `src/vipaneltr/paths.py`:

```python
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    return repo_root() / "data" / "open_vitabqa"


def outputs_dir() -> Path:
    return repo_root() / "outputs"
```

Create `src/vipaneltr/__init__.py`:

```python
"""ViPanelTR: multi-agent reasoning for Vietnamese table QA."""

__version__ = "1.0.0"
```

Create `pyproject.toml` with:

```toml
[project]
name = "vipaneltr"
version = "1.0.0"
description = "Multi-agent reasoning and evaluation for Vietnamese Table QA"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
dependencies = [
  "anthropic>=0.20.0",
  "beautifulsoup4>=4.12.0",
  "google-generativeai>=0.3.0",
  "lxml>=4.9.0",
  "numpy>=1.24.0",
  "openai>=1.0.0",
  "python-dotenv>=1.0.0",
  "pyvi>=0.1.1",
  "pyyaml>=6.0",
  "requests>=2.31.0",
  "tqdm>=4.65.0",
]

[project.optional-dependencies]
eval = ["nltk>=3.8.0", "rouge-score>=0.1.2"]
dev = ["pytest>=8.0.0", "pytest-cov>=5.0.0"]

[project.scripts]
vipaneltr = "vipaneltr.cli:main"

[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --tb=short"
```

Create `.env.example` with empty `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `FPT_AI_MARKETPLACE`, and `SEALION_API_KEY` entries. Create `.gitignore` rules for `.env`, `.env.*` except `.env.example`, Python caches, virtual environments, build artifacts, logs, editor metadata, and `outputs/*` except `outputs/.gitkeep`. Copy the approved spec and plan into the repository. Add the complete MIT license text at root with `Copyright (c) 2026 ViPanelTR contributors`; keep the dataset's upstream copyright notice separately under `data/open_vitabqa/LICENSE`.

- [ ] **Step 5: Verify the package contract**

Run:

```powershell
$env:PYTHONPATH = "$PWD\ViPanel_github\src"
python -m pytest ViPanel_github/tests/unit/test_package_contract.py -q
git -C ViPanel_github check-ignore .env outputs/example/results.json
```

Expected: one test passes; both unsafe paths are ignored.

- [ ] **Step 6: Commit the scaffold**

```powershell
git -C ViPanel_github add .
git -C ViPanel_github commit -m "chore: initialize unified ViPanelTR repository"
```

---

### Task 2: Migrate and validate the Open-ViTabQA data layer

**Files:**
- Create from source: `src/vipaneltr/data/{loader,normalizer,parser,representation,run,__init__}.py`
- Create: `src/vipaneltr/data/validation.py`
- Create: `tests/unit/test_data_validation.py`
- Create: `tests/conftest.py`
- Copy: `Open_ViTabQA/dataset/*` to `data/open_vitabqa/`
- Copy from upstream: `data/open_vitabqa/LICENSE`

**Interfaces:**
- Produces: `validate_dataset(path: str | Path) -> DatasetValidationReport`.
- Produces: `DatasetValidationReport(table_count: int, split_counts: dict[str, int])`.
- Produces: existing `DatasetLoader` and `create_representation` under `vipaneltr.data`.

- [ ] **Step 1: Write validation tests using a minimal local fixture**

Create `tests/unit/test_data_validation.py`:

```python
import json

import pytest

from vipaneltr.data.validation import DatasetValidationError, validate_dataset


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_dataset(tmp_path):
    write_json(tmp_path / "table.json", {"table": [{"table_id": "t1", "table_html": "<table><tr><td>A</td></tr></table>"}]})
    for split in ("train", "dev", "test"):
        write_json(tmp_path / f"qas_{split}.json", {"qas": [{"qa_id": f"{split}-1", "table_id": "t1", "question": "Q?", "answer": "A", "hints": []}]})
    return tmp_path


def test_validate_dataset_reports_counts(tmp_path):
    report = validate_dataset(make_dataset(tmp_path))
    assert report.table_count == 1
    assert report.split_counts == {"train": 1, "dev": 1, "test": 1}


def test_validate_dataset_rejects_unknown_table(tmp_path):
    root = make_dataset(tmp_path)
    write_json(root / "qas_test.json", {"qas": [{"qa_id": "bad", "table_id": "missing"}]})
    with pytest.raises(DatasetValidationError, match="unknown table_id 'missing'"):
        validate_dataset(root)
```

- [ ] **Step 2: Run tests to verify validation is missing**

Run: `python -m pytest ViPanel_github/tests/unit/test_data_validation.py -q`

Expected: FAIL importing `vipaneltr.data.validation`.

- [ ] **Step 3: Copy the preprocessing modules and implement validation**

Copy the six preprocessing modules according to the file map. Replace the loader import:

```python
from vipaneltr.evaluation.normalization import is_unanswerable_reference
```

Create `validation.py` with `DatasetValidationError`, frozen `DatasetValidationReport`, `_load_object(path)`, and `validate_dataset(path)`. The validator must require all four JSON files, require `table.json` to contain a top-level `table` list and every QA split to contain a top-level `qas` list, reject duplicate `qa_id` values within or across splits, and reject missing table references with the exact message used in the test.

- [ ] **Step 4: Copy the official dataset and preserve attribution**

Copy only:

```powershell
Copy-Item Open_ViTabQA/dataset/table.json ViPanel_github/data/open_vitabqa/
Copy-Item Open_ViTabQA/dataset/qas_train.json ViPanel_github/data/open_vitabqa/
Copy-Item Open_ViTabQA/dataset/qas_dev.json ViPanel_github/data/open_vitabqa/
Copy-Item Open_ViTabQA/dataset/qas_test.json ViPanel_github/data/open_vitabqa/
Copy-Item Open_ViTabQA/dataset/TableQA_guideline.pdf ViPanel_github/data/open_vitabqa/
Copy-Item Open_ViTabQA/dataset/README.md ViPanel_github/data/open_vitabqa/README.md
```

Fetch or copy the complete MIT license from `https://github.com/DuzDao/Open-ViTabQA/blob/main/LICENSE`. Record the exact upstream commit from `git ls-remote https://github.com/DuzDao/Open-ViTabQA.git refs/heads/main`, download that commit's archive to a temporary directory, and compare the four official JSON hashes. If the upstream snapshot differs from the local `Open_ViTabQA/dataset` snapshot, copy the four JSON files from the pinned upstream archive so the recorded commit and committed dataset are identical.

- [ ] **Step 5: Verify fixture and full dataset validation**

Run:

```powershell
python -m pytest ViPanel_github/tests/unit/test_data_validation.py -q
python -c "from vipaneltr.data.validation import validate_dataset; print(validate_dataset('ViPanel_github/data/open_vitabqa'))"
Get-FileHash ViPanel_github/data/open_vitabqa/table.json,ViPanel_github/data/open_vitabqa/qas_train.json,ViPanel_github/data/open_vitabqa/qas_dev.json,ViPanel_github/data/open_vitabqa/qas_test.json -Algorithm SHA256
```

Expected: tests pass; reported counts are non-zero; hashes equal the four recorded source hashes in the approved design session.

- [ ] **Step 6: Commit the data layer**

```powershell
git -C ViPanel_github add src/vipaneltr/data data/open_vitabqa tests
git -C ViPanel_github commit -m "feat: integrate Open-ViTabQA data pipeline"
```

---

### Task 3: Migrate the evaluation package

**Files:**
- Create from source: `src/vipaneltr/evaluation/*.py`
- Replace: `src/vipaneltr/evaluation/evaluator.py` with the ViPanelTR wrapper adapted to local modules
- Create: `tests/unit/test_evaluation.py`

**Interfaces:**
- Produces: `evaluate_files(predictions_path, qas_path, output_path=None) -> dict[str, object]`.
- Produces: `Evaluator.evaluate(predictions, references) -> EvaluationResult`.

- [ ] **Step 1: Write the failing offline evaluation test**

```python
from vipaneltr.evaluation.io import align_records
from vipaneltr.evaluation.run import _core_metrics


def test_perfect_prediction_scores_one():
    predictions = [{"qa_id": "q1", "answer": "Hà Nội"}]
    references = [{"qa_id": "q1", "answer": "Hà Nội", "hints": ["what"]}]
    aligned, coverage = align_records(predictions, references)
    scores = _core_metrics(aligned)
    assert coverage.matched == 1
    assert scores["exact_match"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["rouge1"] == 1.0
```

- [ ] **Step 2: Run the test to verify evaluation modules are missing**

Run: `python -m pytest ViPanel_github/tests/unit/test_evaluation.py -q`

Expected: FAIL importing `vipaneltr.evaluation.io`.

- [ ] **Step 3: Copy all Open-ViTabQA evaluation modules**

Copy `answerability_f1.py`, `contracts.py`, `cost.py`, `exact_match.py`, `exceptions.py`, `f1.py`, `io.py`, `meteor.py`, `metrics_by_table_type.py`, `normalization.py`, `rouge1.py`, `rouge1_by_hint.py`, `run.py`, and `__init__.py`. Preserve their relative imports.

- [ ] **Step 4: Adapt the ViPanelTR evaluator wrapper**

Copy `ViPanelTR/paneltr_vitabqa/evaluation/evaluator.py`, remove its `sys.path` bootstrap, and replace all `Open_ViTabQA.evaluation` imports with relative imports from the new package:

```python
from .answerability_f1 import evaluate_answerability
from .io import align_records
from .rouge1_by_hint import evaluate_by_hint
from .run import _core_metrics
```

- [ ] **Step 5: Run evaluation and import-isolation checks**

Run:

```powershell
python -m pytest ViPanel_github/tests/unit/test_evaluation.py -q
rg -n "Open_ViTabQA|sys\.path" ViPanel_github/src/vipaneltr/evaluation
```

Expected: test passes and `rg` returns no runtime dependency match.

- [ ] **Step 6: Commit evaluation**

```powershell
git -C ViPanel_github add src/vipaneltr/evaluation tests/unit/test_evaluation.py
git -C ViPanel_github commit -m "feat: integrate standalone evaluation metrics"
```

---

### Task 4: Migrate the ViPanelTR multi-agent system

**Files:**
- Create from source: `src/vipaneltr/system/agents/*.py`
- Create from source: `src/vipaneltr/system/core/*.py`
- Create from source: `src/vipaneltr/system/prompts/*.py`
- Create from source: `src/vipaneltr/utils/*.py`
- Create from source: `src/vipaneltr/config.py`
- Create: `src/vipaneltr/system/__init__.py`
- Migrate: `tests/unit/test_usage_tracking.py`

**Interfaces:**
- Produces: `PanelTROrchestrator`, `PanelTRResult`, provider clients, artifact writers, and usage tracking under the `vipaneltr` namespace.
- Consumes: `vipaneltr.data.create_representation` and `vipaneltr.evaluation`.

- [ ] **Step 1: Migrate the usage tests with new imports before code**

Copy `ViPanelTR/tests/test_usage_tracking.py` and replace imports with:

```python
from vipaneltr.system.agents.llm_client import UsageTrackingMixin, _normalize_usage, calculate_call_cost
from vipaneltr.utils.trace import StructuredOutputSaver
from vipaneltr.baseline.llm_client import LLMZeroShotClient, calculate_call_cost as baseline_call_cost
from vipaneltr.baseline.run import _calculate_batch_stats
```

Remove the `sys.path` mutation from the test.

- [ ] **Step 2: Run the usage test to verify system modules are missing**

Run: `python -m pytest ViPanel_github/tests/unit/test_usage_tracking.py -q`

Expected: FAIL importing `vipaneltr.system.agents`.

- [ ] **Step 3: Copy system and utility sources mechanically**

Copy all Python files from the three `paneltr_vitabqa` subpackages and `utils`, excluding every `__pycache__` and `.pyc`. Copy `config.py` to package root.

- [ ] **Step 4: Rewrite imports to the unified namespace**

Apply these exact mappings throughout copied files:

```text
In vipaneltr.system.core:    from ..agents/from ..prompts stay unchanged
In vipaneltr.system.core:    from ..utils -> from ...utils
In vipaneltr.system.core:    from ..config -> from ...config
In vipaneltr.system.agents:  from ..utils -> from ...utils
In vipaneltr.system.agents:  from ..config -> from ...config
Everywhere: Open_ViTabQA.preprocessing.representation -> vipaneltr.data.representation
Everywhere: paneltr_vitabqa.* -> vipaneltr.*
```

Use package-relative imports where the target is within `vipaneltr`; do not add `sys.path` code.

- [ ] **Step 5: Verify compilation and usage tracking**

Run:

```powershell
python -m compileall -q ViPanel_github/src/vipaneltr
python -m pytest ViPanel_github/tests/unit/test_usage_tracking.py -q
rg -n "Open_ViTabQA|paneltr_vitabqa|sys\.path" ViPanel_github/src/vipaneltr/system ViPanel_github/src/vipaneltr/utils ViPanel_github/src/vipaneltr/config.py
```

Expected: compilation succeeds, usage tests pass, and no old runtime import/path bootstrap remains.

- [ ] **Step 6: Commit the system**

```powershell
git -C ViPanel_github add src/vipaneltr/system src/vipaneltr/utils src/vipaneltr/config.py tests/unit/test_usage_tracking.py
git -C ViPanel_github commit -m "feat: migrate ViPanelTR multi-agent system"
```

---

### Task 5: Migrate the zero-shot baseline

**Files:**
- Create from source: `src/vipaneltr/baseline/*.py`
- Create from source: `src/vipaneltr/baseline/utils/*.py`
- Create: `tests/unit/test_baseline.py`

**Interfaces:**
- Produces: `run_batch_zeroshot(...)` and `LLMZeroShotClient`.
- Consumes: `vipaneltr.data.DatasetLoader`, `vipaneltr.data.create_representation`, and `vipaneltr.utils.logging.get_logger`.

- [ ] **Step 1: Write a failing baseline import and cost-policy test**

```python
from vipaneltr.baseline.llm_client import calculate_call_cost
from vipaneltr.baseline.run import _calculate_batch_stats


def test_baseline_cost_and_batch_stats_are_offline():
    assert calculate_call_cost("unknown/model", 1_000_000, 1_000_000) == 0.75
    stats = _calculate_batch_stats([{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3, "cost_usd": 0.2}])
    assert stats["total_tokens"] == 3
    assert stats["total_cost_usd"] == 0.2
```

- [ ] **Step 2: Run the test to verify the baseline package is missing**

Run: `python -m pytest ViPanel_github/tests/unit/test_baseline.py -q`

Expected: FAIL importing `vipaneltr.baseline`.

- [ ] **Step 3: Copy baseline sources and update dependencies**

Copy the baseline package, then replace imports as follows:

```python
from vipaneltr.data import create_representation
from vipaneltr.data.loader import DatasetLoader
from vipaneltr.utils.logging import get_logger
```

Remove unused `sys.path` imports and retain environment-only credential loading.

- [ ] **Step 4: Verify the baseline offline**

Run:

```powershell
python -m pytest ViPanel_github/tests/unit/test_baseline.py ViPanel_github/tests/unit/test_usage_tracking.py -q
rg -n "Open_ViTabQA|from baseline|sys\.path" ViPanel_github/src/vipaneltr/baseline
```

Expected: tests pass and no legacy import/path bootstrap remains.

- [ ] **Step 5: Commit baseline code**

```powershell
git -C ViPanel_github add src/vipaneltr/baseline tests/unit
git -C ViPanel_github commit -m "feat: preserve reproducible zero-shot baseline"
```

---

### Task 6: Build the unified four-command CLI

**Files:**
- Create/adapt: `src/vipaneltr/cli.py`
- Modify: `src/vipaneltr/config.py`
- Create: `tests/integration/test_cli.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`.
- Produces subcommands: `prepare-data`, `infer`, `evaluate`, `baseline`.

- [ ] **Step 1: Write failing CLI contract tests**

```python
import pytest

from vipaneltr.cli import main


@pytest.mark.parametrize("command", ["prepare-data", "infer", "evaluate", "baseline"])
def test_each_command_has_help(command):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0


def test_prepare_data_validates_repository_dataset(capsys):
    assert main(["prepare-data"]) == 0
    assert "Dataset valid" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify the unified CLI is absent**

Run: `python -m pytest ViPanel_github/tests/integration/test_cli.py -q`

Expected: FAIL because `main` does not accept `argv` and the four command names are not registered.

- [ ] **Step 3: Adapt the current ViPanelTR CLI**

Copy the current CLI as the behavioral base. Refactor `main` to:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    return int(args.handler(args) or 0)
```

Register exact command names `prepare-data`, `infer`, `evaluate`, and `baseline`. Wire `prepare-data` to `validate_dataset`, preserve the existing inference/evaluation arguments under their renamed commands, and move the argument definitions from `run_baseline.py` into the baseline subparser. Default dataset/output paths must come from `vipaneltr.paths`.

- [ ] **Step 4: Add early credential validation**

Before constructing a provider client, map provider names to exact variables:

```python
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
```

If the selected provider has no configured key, return a non-zero CLI status with `Missing required environment variable: <NAME>` before starting work.

- [ ] **Step 5: Verify all CLI contracts**

Run:

```powershell
python -m pytest ViPanel_github/tests/integration/test_cli.py -q
python -m vipaneltr.cli --help
python -m vipaneltr.cli prepare-data
```

Expected: tests pass, help lists exactly the four public subcommands, and the full dataset validates.

- [ ] **Step 6: Commit the CLI**

```powershell
git -C ViPanel_github add src/vipaneltr/cli.py src/vipaneltr/config.py tests/integration/test_cli.py
git -C ViPanel_github commit -m "feat: expose unified ViPanelTR command line"
```

---

### Task 7: Migrate utility scripts and public documentation

**Files:**
- Create from source: `scripts/*.py`
- Create: `README.md`
- Create: `MIGRATION.md`
- Create: `docs/architecture.md`
- Modify: `data/open_vitabqa/README.md`
- Create: `tests/integration/test_standalone.py`

**Interfaces:**
- Documents all public commands and old-to-new path mappings.
- Proves imports work with only `ViPanel_github/src` available.

- [ ] **Step 1: Write the standalone import test**

```python
import importlib


def test_public_packages_import_without_source_siblings():
    for name in (
        "vipaneltr.data",
        "vipaneltr.evaluation",
        "vipaneltr.system",
        "vipaneltr.baseline",
        "vipaneltr.cli",
    ):
        assert importlib.import_module(name)
```

- [ ] **Step 2: Copy and repair supported scripts**

Copy `split_qas_json.py`, `answerability_classification_f1.py`, `metrics_by_table_type.py`, and `create_subset.py`. Update imports to `vipaneltr.data` or `vipaneltr.evaluation`. Exclude `aggregate_all_metrics.py` because it imports the nonexistent `evaluation.f1_by_answerability` module and hard-codes obsolete output paths; record this exact exclusion and reason in `MIGRATION.md`.

- [ ] **Step 3: Write documentation with runnable commands**

README examples must use:

```powershell
python -m pip install -e ".[dev]"
vipaneltr prepare-data
vipaneltr infer --qas data/open_vitabqa/qas_test.json --model openai/gpt-4o-mini --run-id gpt4o-mini --n 5
vipaneltr evaluate --preds outputs/gpt4o-mini/results.json --qas data/open_vitabqa/qas_test.json --output-dir outputs/gpt4o-mini/eval
vipaneltr baseline --qas data/open_vitabqa/qas_test.json --tables data/open_vitabqa/table.json --model openai/gpt-4o-mini --n 5
```

Document the dataset source URL, upstream commit, MIT attribution, supported environment variables, output layout, offline test command, and explicit warning that real inference may incur provider costs. `MIGRATION.md` must map every copied source directory to its new location and list exclusions.

- [ ] **Step 4: Verify docs and standalone imports**

Run:

```powershell
python -m pytest ViPanel_github/tests/integration/test_standalone.py -q
rg -n "Open_ViTabQA|paneltr_vitabqa|\.\./Open_ViTabQA" ViPanel_github/README.md ViPanel_github/MIGRATION.md ViPanel_github/scripts
```

Expected: standalone import passes; remaining old names occur only in provenance/migration explanations, not executable imports or example paths.

- [ ] **Step 5: Commit scripts and documentation**

```powershell
git -C ViPanel_github add scripts README.md MIGRATION.md docs data/open_vitabqa/README.md tests/integration/test_standalone.py
git -C ViPanel_github commit -m "docs: document standalone workflows and migration"
```

---

### Task 8: Add CI and perform final security/reproducibility verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Verify and, only if a listed check fails, correct: `.gitignore`, `pyproject.toml`, `src/vipaneltr/`, `tests/`, `README.md`, `MIGRATION.md`

**Interfaces:**
- Produces an offline CI matrix for Python 3.10, 3.11, and 3.12.
- Produces a clean initial public Git history with no secret or generated output.

- [ ] **Step 1: Add the GitHub Actions workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m pytest
      - run: python -m compileall -q src/vipaneltr
```

- [ ] **Step 2: Install from the destination and run the full offline suite**

Run from `ViPanel_github`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall -q src/vipaneltr
vipaneltr --help
vipaneltr prepare-data
```

Expected: installation succeeds; all tests pass; compilation and both CLI commands exit zero.

- [ ] **Step 3: Prove source isolation**

Run from outside the workspace with only the installed editable package active:

```powershell
Push-Location $env:TEMP
python -c "import vipaneltr, vipaneltr.data, vipaneltr.evaluation, vipaneltr.system, vipaneltr.baseline; print(vipaneltr.__version__)"
Pop-Location
```

Expected: prints `1.0.0` without adding the original workspace to `PYTHONPATH`.

- [ ] **Step 4: Run secret, output, cache, and legacy dependency scans**

Read the source `.env` values into memory only, compare each non-empty value against destination files without printing those values, and fail the scan if any match occurs. Also run:

```powershell
git -C ViPanel_github ls-files | rg "(^|/)(\.env$|__pycache__|\.pytest_cache|.*\.pyc$|outputs/.+[^.]$)"
rg -n "Open_ViTabQA\.|paneltr_vitabqa\.|sys\.path.*(Open_ViTabQA|ViPanelTR)" ViPanel_github/src ViPanel_github/scripts ViPanel_github/tests
git -C ViPanel_github status --short
```

Expected: no tracked secret/cache/generated output; no legacy runtime imports; clean working tree after intended changes are committed.

- [ ] **Step 5: Confirm dataset integrity**

Expected SHA-256 hashes copied from the source snapshot:

```text
table.json     6FE1298F8F1673364056BB97A74FEC0BB83E2A2841A959E8C5359D2D961D51B4
qas_train.json 33FA274C0B661535F39A99F6BBDAB6290B349D2DD664F061ABAF59C58E56462A
qas_dev.json   D20C046F7651D30F58BDBC0EE46D9E43F490F25B4D32E792726C5609C5C2CDB4
qas_test.json  A28C4FC382D8EC14C4C2D64EDB86B9E05E735E9DFEE2F03AD7DA055982D6939E
```

- [ ] **Step 6: Commit final CI or verification fixes**

```powershell
git -C ViPanel_github add .github .gitignore pyproject.toml src tests README.md MIGRATION.md docs data scripts outputs/.gitkeep
git -C ViPanel_github commit -m "ci: verify public standalone repository"
git -C ViPanel_github status --short --branch
git -C ViPanel_github log --oneline --decorate -8
```

Expected: clean `main` branch and a concise task-oriented commit history. Do not configure a remote or push.

---

## Plan Self-Review Results

- Every confirmed spec requirement is covered by Tasks 1–8.
- Dataset licensing, attribution, hashes, secret exclusion, and source preservation have explicit verification steps.
- All new behavior begins with a failing test; bulk source migration is followed by compile, import, and behavioral tests.
- Public interfaces and command names are consistent across tasks: `prepare-data`, `infer`, `evaluate`, and `baseline`.
- No GitHub remote creation or push is included.
