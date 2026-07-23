# CLI-Notebook Training Alignment

## Problem

The CLI training code (`src/staged_qwen3_5_scivqa/cli/commands.py`) has 7 significant discrepancies with the Jupyter notebooks (source of truth). These cause different training behavior and likely degraded model quality when training via CLI.

## Discrepancies

| # | Area | Notebook | CLI | Severity |
|---|------|----------|-----|----------|
| 1 | Model loading | `use_gradient_checkpointing="unsloth"` | Missing | High |
| 2 | Quantization | `load_in_4bit=False` | Default `True` | High |
| 3 | Data collator | `max_seq_length=MAX_SEQUENCE_LENGTH` | Missing | High |
| 4 | Data collator | `resize="max"` | Missing | High |
| 5 | Data filtering | Filters samples exceeding token limits | No filtering | High |
| 6 | Class balancing | Upsamples minority Yes/No class | None | Medium |
| 7 | Category scope | Combines train+dev | Single category | Medium |

## Approach

**Direct code changes** to `commands.py` and `config.py` — notebooks remain source of truth.

## Changes

### 1. `config.py` — Fix default

- Change `ModelConfig.load_in_4bit` default from `True` to `False`

### 2. `commands.py` — Model loading

Add `use_gradient_checkpointing="unsloth"` to `FastVisionModel.from_pretrained()`.

### 3. `commands.py` — SFTConfig

Replace current SFTConfig with notebook-matching defaults:

```python
SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    num_train_epochs=cfg.training.num_train_epochs,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=budget.max_sequence_length,
)
```

### 4. `commands.py` — Data collator

```python
data_collator=UnslothVisionDataCollator(
    model, tokenizer,
    max_seq_length=budget.max_sequence_length,
    resize="max"
)
```

### 5. `commands.py` — Token filtering

Add `_filter_samples(samples, tokenizer, max_seq_length, max_new_tokens)` helper that uses existing `calculate_token_stats()` from `analysis.py`:

```python
def _filter_samples(samples, tokenizer, max_seq_length, max_new_tokens):
    from staged_qwen3_5_scivqa.analysis import calculate_token_stats
    df_tokens = calculate_token_stats(samples, tokenizer)
    original_size = len(samples)
    samples = [
        sample
        for sample, total_tokens, assistant_tokens in zip(
            samples, df_tokens["total_tokens"], df_tokens["assistant_tokens"], strict=False
        )
        if total_tokens <= max_seq_length and assistant_tokens <= max_new_tokens
    ]
    dropped = original_size - len(samples)
    logger.info("Token filtering: %d → %d samples (%d dropped)", original_size, len(samples), dropped)
    return samples
```

Called after dataset loading, before trainer. Uses `TOKEN_BUDGETS[stage_key]` for thresholds.

### 6. `commands.py` — Yes/No class balancing

Add `_balance_yes_no(samples)` helper matching notebook logic exactly:

```python
def _balance_yes_no(samples):
    import random
    yes_samples = [s for s in samples if s["messages"][1]["content"][0]["text"] == "Yes"]
    no_samples = [s for s in samples if s["messages"][1]["content"][0]["text"] == "No"]
    if len(yes_samples) > len(no_samples):
        majority, minority = yes_samples, no_samples
    else:
        majority, minority = no_samples, yes_samples
    diff = len(majority) - len(minority)
    if diff > 0:
        upsampled = random.choices(minority, k=diff)
        samples = majority + minority + upsampled
        random.seed(3407)
        random.shuffle(samples)
    return samples
```

Only applied when `stage == "yes_no"`.

### 7. `settings.py` + `commands.py` — Multi-category

- Change `category` default from `"test"` to `"train,dev"`
- Parse as comma-separated list
- Loop over categories, extend dataset

## Testing

Add `@pytest.mark.unit` tests for:
- `_filter_samples()` — verify samples exceeding limits are removed
- `_balance_yes_no()` — verify minority class is upsampled correctly
- Multi-category loading — verify datasets are combined

## Files Modified

- `src/staged_qwen3_5_scivqa/config.py`
- `src/staged_qwen3_5_scivqa/settings.py`
- `src/staged_qwen3_5_scivqa/cli/commands.py`
- `tests/test_cli/test_commands.py` (new tests)
