<div align="center">
  <img
      src="https://raw.githubusercontent.com/insane-group/staged-qwen3.5-scivqa/refs/heads/main/images/staged_vqa_pipeline.svg"
      alt="Staged VQA Pipeline"
    />
</div>

<p align="center">
  <a href="https://github.com/billsioros/staged-qwen3.5-scivqa/actions/workflows/ci.yml">
    <img src="https://github.com/billsioros/staged-qwen3.5-scivqa/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <a href="https://github.com/billsioros/staged-qwen3.5-scivqa/actions/workflows/cd.yml">
    <img src="https://github.com/billsioros/staged-qwen3.5-scivqa/actions/workflows/cd.yml/badge.svg" alt="CD" />
  </a>
  <a href="https://sites.google.com/view/sci-imageminer/">
    <img src="https://img.shields.io/badge/Competition-Sci--ImageMiner-blue" alt="Competition" />
  </a>
  <a href="https://github.com/billsioros/staged-qwen3.5-scivqa/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" />
  </a>
</p>

> Staged multimodal pipeline for scientific figure VQA using Qwen3.5 — summarization, table extraction, and answer-type-specific fine-tuning for the [ICDAR 2026 Sci-ImageMiner competition](https://sites.google.com/view/sci-imageminer/).
>
> **Note:** The code used for the competition is specifically [this commit](https://github.com/insane-group/staged-qwen3.5-scivqa/tree/16316d797c687ae234263ff48f8403044e3490a4).

## :bar_chart: Results

| Task                               | Best Score                                           | Rank |
| ---------------------------------- | ---------------------------------------------------- | ---- |
| **Task 2 — Data Table Extraction** | Weighted=35.07, TEDS=55.2                            | 5th  |
| **Task 3 — Summarization**         | Weighted=0.5340, ROUGE-L=0.2715, BERTScore F1=0.8161 | 6th  |
| **Task 4 — VQA**                   | Weighted=0.26                                        | 5th  |

## :rocket: Getting Started

```shell
# Clone the repository
git clone https://github.com/billsioros/staged-qwen3.5-scivqa
cd staged-qwen3.5-scivqa

# Install dependencies
uv sync --all-groups

# Run unit tests (no GPU needed)
poe test unit

# Run full test suite with coverage
poe coverage
```

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Competition data from the [Sci-ImageMiner download page](https://sites.google.com/view/sci-imageminer/download-the-data)
- [cvc5](https://github.com/cvc5/cvc5) solver at `~/cvc5-Linux-x86_64-shared/bin/cvc5` (optional, for SMT reflection):

  ```shell
  wget https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.3/cvc5-Linux-x86_64-shared.zip
  unzip cvc5-Linux-x86_64-shared.zip -d ~
  rm cvc5-Linux-x86_64-shared.zip
  ```

### CLI Usage

```shell
# Run full pipeline (summary → table → VQA → SMT → reflection)
sci-vqa run --stages summary,table,vqa,smt,reflect --category test [--resume] [--config pipeline.yaml]

# Train individual stages
sci-vqa train summary --category test
sci-vqa train table --category test
sci-vqa train vqa --category test --answer-types factoid,list,paragraph,yes_no

# Run inference
sci-vqa inference vqa --category test --checkpoint-dir ./models/vqa

# SMT pipeline & reflection (requires outlines + cvc5)
sci-vqa smt run --category test [--model-id unsloth/Qwen3.5-9B] [--max-retries 3]
sci-vqa reflect --category test [--model-id unsloth/Qwen3.5-9B]

# Evaluate predictions
sci-vqa eval vqa --predictions data/submission_final.json --category test
sci-vqa eval summary --predictions data/summary_results.json --category test
sci-vqa eval table --predictions data/table_results.json --category test

# HuggingFace Hub integration
sci-vqa hf push ./checkpoint --repo-id user/model
sci-vqa hf pull --repo-id user/model --output ./models/
sci-vqa hf push-dataset ./data/processed --repo-id user/dataset
sci-vqa hf pull-dataset --repo-id user/dataset --output ./data/
```

### Development Tasks

```shell
poe fmt          # Format + fix with ruff
poe lint         # Lint code
poe types        # Type check with mypy
poe hooks        # Run all pre-commit checks
poe test unit    # Unit tests only
poe test all     # Full suite
poe coverage     # Coverage report
```

## :bulb: Why?

Scientific figures encode trends, values, and relationships that are difficult to recover from text alone — especially in Atomic Layer Deposition and Etching (ALD/E) research. This repository implements a **staged multimodal pipeline** that chains summarization and table extraction as auxiliary evidence into a VQA model, with an experimental neurosymbolic reflection path for formal verification.

### Main Techniques

- **QLoRA fine-tuning** of `unsloth/Qwen3.5-9B` (r=16, α=16, 16-bit training, 4-bit inference)
- **Cross-task context injection**: summaries + tables → VQA prompts
- **Answer-type-specific token budgets** tuned to competition data percentiles
- **Answer-type-aware preprocessing**: Unicode resolution, whitespace/punctuation cleanup, format-specific post-processing
- **Neurosymbolic reflection (WIP)**: Grammar-constrained SMT-LIB decoding via cvc5 with answer rewriting

## :open_file_folder: Project Structure

```
staged-qwen3.5-scivqa/
├── notebooks/                    # Jupyter notebooks (primary experimentation)
│   ├── 1. Data loading.py
│   ├── 2. Finetuning Qwen3.5 (submission).py
│   ├── 2. Finetuning Qwen3.5 (Factoid/List/Paragraph/Yes|No) (submission).py
│   ├── 2. Finetuning Qwen3.5 (image+context->summary/table) (submission).py
│   ├── 4. Qwen3.5 Image+Context-to-SMT.py
│   ├── 7. Reflecting on Qwen3.5 answers using SMT (submission).py
│   └── 8. Merge states into submission.py
├── src/staged_qwen3_5_scivqa/    # Production package
│   ├── config.py                 # Constants, prompts, token budgets, SMT grammars
│   ├── data.py                   # Dataset loading
│   ├── preprocessing.py          # Answer cleaning and validation
│   ├── analysis.py               # Token statistics, quality reports
│   ├── context.py                # Paper context extraction
│   ├── conversation.py           # Qwen/Unsloth conversation formatting
│   ├── models/                   # loader, lora, trainer, inference
│   ├── evaluation/               # BERTScore, ROUGE, TEDS, accuracy, set F1
│   └── smt/                      # grammars, solver, pipeline, reflection
├── tests/                        # Unit and integration tests (fully mocked)
├── .github/workflows/            # CI (pytest + coverage) and CD (semantic release)
├── pyproject.toml                # Project metadata, uv/poe/ruff/mypy config
├── .pre-commit-config.yaml       # Pre-commit hooks
├── data/                         # Saved states and outputs (gitignored)
├── models/                       # LoRA checkpoints (gitignored)
└── ALD-E-ImageMiner/             # Competition data (external, gitignored)
```

## :notebook: Notebooks

Notebooks are the primary experimentation interface. Edit the `.py` (percent script) versions, then sync:

```shell
jupytext --sync notebooks/*.py
```

## :bookmark_tabs: Citation

```bibtex
@inproceedings{stagedqwen35scivqa2026sciimage,
  title     = {Staged Qwen3.5 SciVQA: QLoRA Fine-tuning with Neurosymbolic
               Reflection for Scientific Figure VQA},
  author    = {Staged Qwen3.5 SciVQA Contributors},
  booktitle = {Sci-ImageMiner 2026: Scientific Image Mining Challenge at ICDAR 2026},
  year      = {2026},
  publisher = {TIB Open Publishing}
}
```

## :coin: Credits

- [ICDAR 2026 Sci-ImageMiner Competition](https://sites.google.com/view/sci-imageminer/) — organized by TIB, TU Eindhoven, and University of Warwick, supported by [NFDI4DataScience](https://www.nfdi4datascience.de/) (DFG Grant ID: 460234259)
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5) — base vision-language model
- [Unsloth](https://github.com/unslothai/unsloth) — accelerated fine-tuning
- [cvc5](https://github.com/cvc5/cvc5) — SMT solver for neurosymbolic reflection
