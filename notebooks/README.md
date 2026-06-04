# Notebooks

Exploratory and pipeline notebooks for the ICDAR 2026 Sci-ImageMiner competition.
Numbering follows the pipeline stage order.

## Notebook Index

| # | Notebook | Stage | Description |
|---|----------|-------|-------------|
| 1 | `1. Data loading.ipynb` | Data | Explore dataset structure, annotation format, bounding boxes, and paper context |
| 2 | `2. Finetuning Qwen3.5 (Factoid).ipynb` | VQA - Train | LoRA fine-tuning for Factoid answer type |
| 2 | `2. Finetuning Qwen3.5 (List).ipynb` | VQA - Train | LoRA fine-tuning for List answer type |
| 2 | `2. Finetuning Qwen3.5 (Paragraph).ipynb` | VQA - Train | LoRA fine-tuning for Paragraph answer type |
| 2 | `2. Finetuning Qwen3.5 (Yes\|No).ipynb` | VQA - Train | LoRA fine-tuning for Yes/No answer type |
| 2 | `2. Finetuning Qwen3.5 (image+context->summary).ipynb` | Summary - Train | LoRA fine-tuning for summarization |
| 2 | `2. Finetuning Qwen3.5 (image+context->table).ipynb` | Table - Train | LoRA fine-tuning for table extraction |
| 2 | `* (submission).ipynb` variants | VQA - Inference | Inference with trained LoRAs, producing submission state files |
| 3 | `3. Qwen3.5 Image+Context-to-SMT (playground).ipynb` | SMT - Playground | Interactive exploration of grammar-constrained SMT-LIB decoding on individual samples |
| 3 | `3. Qwen3.5 Image+Context-to-SMT (pipeline).ipynb` | SMT - Pipeline | Batch SMT pipeline over full dataset using `run_smt_pipeline()` |
| 4 | `4. Grammar-Constrained Decoding.ipynb` | POC | Proof-of-concept for outlines CFG-constrained decoding (arithmetic examples) |
| 5 | `5. Reflecting on Qwen3.5 answers using SMT (playground).ipynb` | Reflection - Playground | Explore contradictory SMT-verified Yes/No cases and rewrite answers |
| 5 | `5. Reflecting on Qwen3.5 answers using SMT (pipeline).ipynb` | Reflection - Pipeline | Batch reflection pipeline using `run_reflection()` |
| 6 | `6. Merge states into submission.ipynb` | Merge | Merge partial per-answer-type state files into a single submission JSON |
| 7 | `7. Evaluation (pipeline).ipynb` | Eval | Overall evaluation metrics |
| 7 | `7. Evaluation (summarization).ipynb` | Eval | Summarization-specific evaluation |
| 7 | `7. Evaluation (data extraction).ipynb` | Eval | Table extraction-specific evaluation |

## Pipeline Flow

```
1. Data loading
       ↓
2. Finetuning (Summary → Table → VQA per answer type)
       ↓
3. SMT grammar-constrained decoding
       ↓
4. Grammar-constrained decoding POC (standalone)
       ↓
5. Reflection on answers using SMT
       ↓
6. Merge states → submission
       ↓
7. Evaluation
```

## Conventions

- **Playground** notebooks are interactive/exploratory — they sample single items and display verbose output.
- **Pipeline** notebooks run batch processing over entire datasets and produce state files in `data/`.
- All notebooks import from the `staged_qwen3_5_scivqa` library rather than duplicating logic.
- Submission variants under `2.` run inference with pre-trained LoRA adapters.
