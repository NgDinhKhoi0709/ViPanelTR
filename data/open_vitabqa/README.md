# Open-ViTabQA: A Novel Benchmark for Vietnamese Question Answering on Open-Domain Wikipedia Tables

## Source and license

The dataset in this directory is distributed by
[`DuzDao/Open-ViTabQA`](https://github.com/DuzDao/Open-ViTabQA) under the MIT
License included in this directory.

Open-ViTabQA is a public dataset for Vietnamese table question answering. It is
designed to benchmark language models, including large language models, on
understanding and answering questions grounded in Vietnamese Wikipedia tables.

## Motivation

Tabular data is a rich source of information, and automated information
extraction from tables is an important natural language processing task. Most
table question-answering resources focus on English. Open-ViTabQA provides a
Vietnamese benchmark that captures challenges such as word segmentation,
diverse syntax, implicit information, and irregular table structures.

## Data structure

Each question-answer record contains:

| Field | Description |
|---|---|
| `qa_id` | Unique question-answer identifier |
| `table_id` | Identifier of the associated table |
| `question` | Vietnamese question |
| `answer` | Reference answer |
| `hints` | Question-type annotations |

Each table record contains:

| Field | Description |
|---|---|
| `table_id` | Unique table identifier |
| `table_html` | Table content in HTML format |
| `table_title` | Table title |
| `table_domain` | Table domain |
| `table_type` | Structure labels such as `normal`, `contain_merged_header`, or `contain_merged_value` |
| `table_dict` | Flattened table rows |

## Citation

When using these files, cite the Open-ViTabQA paper and dataset repository. The
complete citation is available in the root project README.

## Contact

For dataset questions, contact `21521972@gm.uit.edu.vn`.
