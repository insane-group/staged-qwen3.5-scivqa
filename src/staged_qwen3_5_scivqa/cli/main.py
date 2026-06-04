"""CLI entry point for the Sci-ImageMiner VQA pipeline."""

import typer

from staged_qwen3_5_scivqa.cli.commands import (
    dataset_app,
    eval_app,
    hf_app,
    inference_app,
    reflect,
    run_pipeline,
    smt_app,
    train_app,
)

app = typer.Typer(
    name="sci-vqa",
    help=(
        "Sci-ImageMiner VQA pipeline CLI — train, evaluate, and deploy "
        "multimodal models for scientific figure understanding."
    ),
    rich_markup_mode="rich",
    add_completion=False,
)

app.add_typer(train_app, name="train", help="Train LoRA adapters")
app.add_typer(inference_app, name="inference", help="Run inference")
app.add_typer(smt_app, name="smt", help="SMT-LIB decoding")
app.add_typer(eval_app, name="eval", help="Evaluate predictions")
app.add_typer(dataset_app, name="dataset", help="Build/push HF datasets")
app.add_typer(hf_app, name="hf", help="HuggingFace Hub")

app.command("run")(run_pipeline)
app.command("reflect")(reflect)

if __name__ == "__main__":
    app()
