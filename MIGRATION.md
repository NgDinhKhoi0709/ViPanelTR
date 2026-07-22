# Migration from the research workspace

This repository was assembled from the standalone `ViPanelTR` and
`Open_ViTabQA` workspace directories. The original directories remain unchanged.

## Path mapping

| Previous location | New location |
|---|---|
| `ViPanelTR/paneltr_vitabqa/agents` | `src/vipaneltr/system/agents` |
| `ViPanelTR/paneltr_vitabqa/core` | `src/vipaneltr/system/core` |
| `ViPanelTR/paneltr_vitabqa/prompts` | `src/vipaneltr/system/prompts` |
| `ViPanelTR/paneltr_vitabqa/utils` | `src/vipaneltr/utils` |
| `ViPanelTR/paneltr_vitabqa/config.py` | `src/vipaneltr/config.py` |
| `Open_ViTabQA/preprocessing` | `src/vipaneltr/data` |
| `Open_ViTabQA/evaluation` | `src/vipaneltr/evaluation` |
| `ViPanelTR/baseline` | `src/vipaneltr/baseline` |
| `Open_ViTabQA/dataset` | `data/open_vitabqa` |
| `ViPanelTR/scripts` and `ViPanelTR/utils/create_subset.py` | `scripts` |

Runtime imports now use `vipaneltr.*`; the installed package does not require an
adjacent `Open_ViTabQA` or `ViPanelTR` directory and does not modify `sys.path`.

## Command mapping

| Previous command | New command |
|---|---|
| `python run_vipaneltr.py run_infer ...` | `vipaneltr infer ...` |
| `python run_vipaneltr.py run_eval ...` | `vipaneltr evaluate ...` |
| `python run_baseline.py ...` | `vipaneltr baseline ...` |
| Dataset checks embedded in loaders | `vipaneltr prepare-data` |

For evaluation, `--preds` replaces `--pred`. `--output-dir DIR` writes
`DIR/evaluation.json`; the exact-file `--output FILE` option remains available.

## Dataset selection

The public repository uses the official upstream dataset at commit
`3e027061117d2c98e6131034f91f48c75165fd71`. The workspace `qas_test.json`
hash differed from that commit, so migration selected the pinned upstream test
split. Table, train, and development files matched upstream exactly.

## Exclusions

- `.env` and every credential value.
- All existing generated results under `ViPanelTR/outputs`.
- Python bytecode, caches, logs, and editor metadata.
- `aggregate_all_metrics.py`, because it imports the nonexistent
  `evaluation.f1_by_answerability` module and hard-codes obsolete experiment
  paths. The supported evaluation package and CLI replace it.
- Unrelated workspace systems and external baselines.

Generated runs belong under `outputs/`, where Git tracks only `.gitkeep`.
