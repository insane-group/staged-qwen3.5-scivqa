# AGENTS.md — staged-qwen3.5-scivqa

## Project

ICDAR 2026 Sci-ImageMiner competition: VQA on scientific figures from
ALD/E papers. Staged pipeline: **summarization → table extraction → VQA → (optional) SMT reflection**.
Answer types: Factoid, List, Paragraph, Yes/No, Table/Summary.

## Operating Defaults

- Prefer small, targeted diffs over broad refactors.
- Read the relevant source file before editing — do not guess function signatures or data shapes.
- Preserve existing naming, import style, and file layout unless the task explicitly requires change.
- If a request is ambiguous or could change pipeline behavior, ask before implementing.
- When proceeding with incomplete info, make the smallest safe change and state assumptions.
- Before non-trivial edits, restate the intended change in 1–2 sentences.

## Commands

```bash
uv sync --all-groups              # Install all deps (including dev)
poe fmt && poe lint && poe types  # Format → lint → typecheck (run in order)
poe test unit                     # Unit tests only (no GPU needed)
poe test all                      # Full suite
poe coverage                      # Coverage report (fail_under=60)
poe hooks                         # All pre-commit checks (uses prek, not pre-commit)
```

### CLI (`sci-vqa`)

```bash
# Full pipeline (all stages)
sci-vqa run --stages summary,table,vqa,smt,reflect --category test [--resume] [--config pipeline.yaml]

# Train individual stages
sci-vqa train summary --category test
sci-vqa train table --category test
sci-vqa train vqa --category test --answer-types factoid,list,paragraph,yes_no

# Inference
sci-vqa inference vqa --category test --checkpoint-dir ./models/vqa

# SMT pipeline (requires outlines + cvc5)
sci-vqa smt run --category test [--model-id unsloth/Qwen3.5-9B] [--max-retries 3]

# Reflection (requires FastLanguageModel)
sci-vqa reflect --category test [--model-id unsloth/Qwen3.5-9B]

# Evaluate predictions
sci-vqa eval vqa --predictions data/submission_final.json --category test
sci-vqa eval summary --predictions data/summary_pred.json --category test
sci-vqa eval table --predictions data/table_pred.json --category test

# HuggingFace Hub
sci-vqa hf push ./checkpoint --repo-id user/model
sci-vqa hf pull --repo-id user/model --output ./models/
sci-vqa hf push-dataset ./data/dataset --repo-id user/data
sci-vqa hf pull-dataset --repo-id user/data --output ./data/
```

**Config**: `--config pipeline.yaml` (YAML), env vars (`SCIVQA_*`), or `.env` file.
See `src/staged_qwen3_5_scivqa/settings.py` for all configurable fields.

## Initialization

### First-time setup

```bash
uv sync --all-groups              # Install deps (unsloth/unsloth-zoo from git)
```

### cvc5 (SMT solver — required for SMT/reflection stages)

Not pip-installable. Download and extract to `~/`:

```bash
wget https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.3/cvc5-Linux-x86_64-shared.zip
unzip cvc5-Linux-x86_64-shared.zip -d ~
rm cvc5-Linux-x86_64-shared.zip
```

Binary expected at `~/cvc5-Linux-x86_64-shared/bin/cvc5` (configured in `config.py:CVC5_PATH`).
Tests mock the subprocess call — cvc5 not needed for `poe test unit`.

### Torch / CUDA

`torch==2.8` is pinned in `pyproject.toml`. Unsloth compiles CUDA kernels on first import
(`unsloth_compiled_cache/` in `.gitignore`). If you see compilation errors, delete the cache
and re-run.

### Competition data

Place at `ALD-E-ImageMiner/icdar2026-competition-data/{train,dev,test}/`.
Download from https://sites.google.com/view/sci-imageminer/download-the-data.

## Feature / Fix Workflow

1. **Write tests first** — add `@pytest.mark.unit` to `tests/` (fully mocked, no GPU)
2. **Implement the change** in `src/staged_qwen3_5_scivqa/`
3. **Verify**: `poe fmt && poe lint && poe types && poe test unit`
4. **Commit** with conventional commit message (enforced by commitizen + gitlint)

New test files mirror source structure (`src/foo/bar.py` → `tests/test_foo/test_bar.py`).
Use fixtures from `tests/conftest.py`: `mock_tokenizer`, `sample_annotation`,
`sample_content_json`, `tmp_data_dir`.

## Architecture

**Source:** `src/staged_qwen3_5_scivqa/` — production package.
**Notebooks:** `notebooks/*.ipynb` — experimentation, excluded from most hooks.
**Tests:** `tests/` — fully mocked, no GPU required.
**CLI:** `src/staged_qwen3_5_scivqa/cli/` — Typer entry point (`sci-vqa`).

### Package layout

```
src/staged_qwen3_5_scivqa/
├── config.py            # ALL constants: prompts, token budgets, SMT grammars, cvc5 path
├── settings.py          # Pydantic Settings hierarchy for CLI config
├── data.py              # Dataset loading functions
├── preprocessing.py     # Answer cleaning and validation
├── analysis.py          # Token statistics, quality reports
├── context.py           # Paper context extraction from content.json
├── conversation.py      # Qwen/Unsloth conversation formatting
├── models/              # loader.py, lora.py, trainer.py, inference.py,
│                        # smt_runner.py, reflection_runner.py
├── evaluation/          # metrics.py (BERTScore, ROUGE, TEDS, accuracy, set F1)
├── smt/                 # grammars.py, solver.py, pipeline.py, reflection.py
└── cli/                 # main.py, commands.py, utils.py (Typer CLI)
```

### Data flow

```
Competition JSON + JPG → [Summary LoRA] → data/summary_state.json
                       → [Table LoRA]    → data/extraction_state.json
                       → [VQA LoRA]      → data/finetuning_state.json
                       → [SMT pipeline]  → data/smt_state.json
                       → [Reflection]    → data/submission_final_*.json
```

## Key conventions

- **Notebooks**: only `.ipynb` files currently committed. Jupytext pairing configured in
  `pyproject.toml` (`notebooks/` → `ipynb`, `scripts/` → `py:percent`). `nbstripout` strips
  notebook outputs on commit. Use `poe jupytext` to sync paired files.
- **Notebooks are excluded** from ruff, codespell, debug-statements, editorconfig, and check-ast hooks.
  Do not try to fix lint errors in `notebooks/` — they are research code.
- **Notebooks import from the library**: all utility functions, constants, and prompts come from
  `staged_qwen3_5_scivqa.*` (no `src.` prefix — editable install). Notebook-specific code is limited
  to model loading (`FastVisionModel.*`), training loops, visualizations, and inference loops.
- **Commit messages**: conventional commits enforced by commitizen + gitlint.
- **Version**: bump in `src/staged_qwen3_5_scivqa/__init__.py` (semantic release reads this).
- **All state/submission files saved to `data/`** — not repo root.
- **HF Hub push after training**: enabled by default (`push_checkpoints=True`). Set
  `SCIVQA_HF__TOKEN` env var or configure `hf.token`/`hf.hub_repo_id` to use it.

## Data & paths

- Competition data: `ALD-E-ImageMiner/icdar2026-competition-data/{train,dev,test}/` (external, gitignored)
- LoRA checkpoints: `Sci-ImageMiner-Qwen3.5-*LORA*` (gitignored)
- `.gitignore` excludes: `/data/`, `/models/`, `/logs/`, `/outputs/`, `unsloth_compiled_cache/`, `notebooks/*.json`

## Testing

- All tests are **fully mocked** — no GPU, no network, no cvc5 binary needed.
- `tests/conftest.py` fixtures: `mock_tokenizer`, `sample_annotation`, `sample_content_json`, `tmp_data_dir`.
- Model loading tests mock `unsloth`/`transformers` classes. SMT solver tests mock `subprocess.run`.
- Mark tests with `@pytest.mark.unit` or `@pytest.mark.integration`.
- Coverage source: `src/staged_qwen3_5_scivqa` (fail_under=60).

## External dependencies

- **cvc5**: SMT solver binary at `~/cvc5-Linux-x86_64-shared/bin/cvc5`. Not pip-installable.
  `smt/solver.py` calls it via subprocess; tests mock this.
- **unsloth / unsloth-zoo**: git sources from GitHub (see `[tool.uv.sources]`).
- **torch==2.8**, **transformers==5.2**, **trl==0.22.2**: pinned versions.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): pytest + coverage → Codecov. Triggers on push/PR to `master`
  when `src/**`, `tests/**`, `pyproject.toml`, or `uv.lock` change. Uses `uv sync --group dev`.
- **CD** (`.github/workflows/cd.yml`): semantic release on push to `master`. Creates GitHub releases.
- **Pre-commit hooks**: ruff, ruff-format, codespell, check-ast, check-case-conflict,
  check-docstring-first, check-merge-conflict, detect-private-key, fix-byte-order-marker,
  mixed-line-ending, trailing-whitespace, end-of-file-fixer, check-yaml, check-toml,
  debug-statements, editorconfig-checker, commitizen, gitlint, vulture, bandit,
  pyproject-fmt, yamlfmt, gitleaks, nbstripout. (uses `prek`, not `pre-commit`)

## Gotchas

- `pyproject.toml` `dependencies` must be under `[project]`, NOT after `[project.urls]` (TOML section ordering).
- `poe fmt` runs `ruff check --fix` (not `ruff format`). Ruff handles both linting and formatting.
- `mypy` is configured with `strict = false` and `ignore_missing_imports = true` (heavy ML deps).
- Notebook `.py` files use `# %%` cell markers — they are jupytext percent scripts, not standalone modules.
- Submission JSON format: `[{ "sample_id": str, "vqa": { sub_fig: [ {question_type, answer_type, question, answer} ] } }]`
- `vulture` dead code detection uses `vulture_whitelist.py` for intentionally unused strings (prompts, grammars).
- `pyproject-fmt` hook enforces 2-space indent in `pyproject.toml`.
- `prek` is used instead of `pre-commit` for running hooks (faster Rust implementation).
- `config.py` (flat constants) and `settings.py` (Pydantic Settings) coexist.
  Notebooks import from `config.py`; CLI uses `settings.py`.
- `config.py`, `smt/grammars.py`, `smt/pipeline.py`, and `smt/reflection.py` have E501/C901 per-file ignores
  because prompts and SMT grammar strings are inherently long and complex.
- `vulture_whitelist.py` has B018/F821 ignores — it contains bare names intentionally.
- Codespell ignores: `ans`, `rouge`, `ser` (false positives in ML/SMT context).
- `editorconfig` exempts `config.py` and `grammars.py` from indent rules (grammar DSL uses custom indentation).
