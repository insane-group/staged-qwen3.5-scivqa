"""CLI commands for the Sci-ImageMiner VQA pipeline.

Commands:
    train summary/table/vqa — LoRA fine-tuning per stage
    inference vqa — run inference with trained LoRA
    smt run — grammar-constrained SMT pipeline
    reflect — answer reflection using SMT output
    eval vqa/summary/table — evaluate predictions against ground truth
    run — end-to-end pipeline with resume support
    hf push/pull/push-dataset/pull-dataset — HuggingFace Hub integration
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from staged_qwen3_5_scivqa.cli.utils import (
    console,
    finish_wandb,
    load_json_state,
    print_metrics_table,
    print_skip_header,
    print_stage_header,
    print_submission_summary,
    progress_context,
    pull_dataset,
    pull_from_hub,
    push_dataset,
    push_to_hub,
    setup_wandb,
    stage_has_output,
)
from staged_qwen3_5_scivqa.config import SciVQAConfig, load_config

train_app = typer.Typer(help="Train LoRA adapters for pipeline stages")
inference_app = typer.Typer(help="Run inference with trained models")
smt_app = typer.Typer(help="SMT-LIB grammar-constrained decoding")
eval_app = typer.Typer(help="Evaluate predictions against ground truth")
hf_app = typer.Typer(help="HuggingFace Hub integration")

ConfigArg = Annotated[
    str | None,
    typer.Option("--config", "-c", help="Path to YAML config file"),
]
CategoryArg = Annotated[
    str,
    typer.Option("--category", help="Competition data category (train/dev/test)"),
]
ResumeArg = Annotated[
    bool,
    typer.Option("--resume/--no-resume", help="Resume from existing stage outputs"),
]


# ── Train commands ────────────────────────────────────────────────────


@train_app.command("summary")
def train_summary(
    category: CategoryArg = "test",
    config: ConfigArg = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    epochs: Annotated[int | None, typer.Option("--epochs", "-e")] = None,
    lr: Annotated[float | None, typer.Option("--lr")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", "-b")] = None,
):
    """Train summarization LoRA adapter."""
    cfg = load_config(
        config,
        **_filter_none(
            category=category,
            **{
                "paths.output_dir": output_dir,
                "training.epochs": epochs,
                "training.lr": lr,
                "training.batch_size": batch_size,
            },
        ),
    )
    train_stage(cfg, "summary", cfg.paths.output_dir / "summary")


@train_app.command("table")
def train_table(
    category: CategoryArg = "test",
    config: ConfigArg = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    epochs: Annotated[int | None, typer.Option("--epochs", "-e")] = None,
    lr: Annotated[float | None, typer.Option("--lr")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", "-b")] = None,
):
    """Train table extraction LoRA adapter."""
    cfg = load_config(
        config,
        **_filter_none(
            category=category,
            **{
                "paths.output_dir": output_dir,
                "training.epochs": epochs,
                "training.lr": lr,
                "training.batch_size": batch_size,
            },
        ),
    )
    train_stage(cfg, "table", cfg.paths.output_dir / "table")


@train_app.command("vqa")
def train_vqa(
    category: CategoryArg = "test",
    config: ConfigArg = None,
    answer_types: Annotated[
        str,
        typer.Option(
            "--answer-types",
            "-a",
            help="Comma-separated: factoid,list,paragraph,yes_no",
        ),
    ] = "factoid,list,paragraph,yes_no",
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    epochs: Annotated[int | None, typer.Option("--epochs", "-e")] = None,
    lr: Annotated[float | None, typer.Option("--lr")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", "-b")] = None,
):
    """Train VQA LoRA adapters per answer type (one LoRA per type)."""
    types = [t.strip() for t in answer_types.split(",")]
    valid = {"factoid", "list", "paragraph", "yes_no"}
    invalid = set(types) - valid
    if invalid:
        console.print(f"[red]Invalid answer types: {invalid}[/red]")
        raise typer.Exit(1)

    cfg = load_config(
        config,
        **_filter_none(
            category=category,
            **{
                "paths.output_dir": output_dir,
                "training.epochs": epochs,
                "training.lr": lr,
                "training.batch_size": batch_size,
            },
        ),
    )

    for atype in types:
        checkpoint_dir = cfg.paths.output_dir / f"vqa_{atype}"
        train_stage(cfg, atype, checkpoint_dir)


def train_stage(cfg: SciVQAConfig, stage: str, output_dir: Path):
    """Train a single stage LoRA adapter."""
    setup_wandb(cfg.wandb)

    budget = cfg.get_stage_budget(stage)
    checkpoint_name = cfg.get_lora_checkpoint_name(stage)

    print_stage_header(f"Training {stage}", 1, 1)
    console.print(f"  Model: {cfg.model.model_id}")
    console.print(f"  LoRA: r={cfg.lora.r}, α={cfg.lora.alpha}")
    console.print(f"  Epochs: {cfg.training.epochs}")
    console.print(f"  LR: {cfg.training.lr}")
    console.print(f"  Batch: {cfg.training.batch_size} × {cfg.training.grad_accum}")
    console.print(f"  Max seq: {budget.max_sequence_length}")
    console.print(f"  Max new tokens: {budget.max_new_tokens}")
    console.print(f"  Output: {output_dir}")
    console.print()

    try:
        from unsloth import FastVisionModel

        from staged_qwen3_5_scivqa.data import (
            load_summary_dataset,
            load_table_dataset,
            load_vqa_dataset,
        )
        from staged_qwen3_5_scivqa.models.lora import get_lora_config
        from staged_qwen3_5_scivqa.models.trainer import get_sft_config

        with progress_context("Loading model...") as progress:
            progress.add_task(description="Loading model...", total=None)
            model, tokenizer = FastVisionModel.from_pretrained(
                cfg.model.model_id,
                load_in_4bit=cfg.model.load_in_4bit,
                max_seq_length=budget.max_sequence_length,
            )
            model = FastVisionModel.get_peft_model(
                model, **get_lora_config(**cfg.lora.model_dump())
            )
            FastVisionModel.for_training(model)

        with progress_context("Loading dataset...") as progress:
            progress.add_task(description="Loading dataset...", total=None)
            if stage == "summary":
                samples, _, _ = load_summary_dataset(cfg.category)
            elif stage == "table":
                samples, _, _ = load_table_dataset(cfg.category)
            else:
                answer_types = [stage] if stage != "vqa" else None
                samples, _, _ = load_vqa_dataset(cfg.category, answer_types)

        report_to = "wandb" if cfg.wandb.enabled else "none"
        sft_cfg = get_sft_config(
            max_length=budget.max_sequence_length,
            num_train_epochs=cfg.training.epochs,
            per_device_train_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.grad_accum,
            warmup_ratio=cfg.training.warmup_ratio,
            learning_rate=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            output_dir=str(output_dir),
            report_to=report_to,
        )

        with progress_context("Training...") as progress:
            progress.add_task(description="Training...", total=None)
            from trl import SFTTrainer
            from unsloth import UnslothVisionDataCollator

            trainer = SFTTrainer(  # type: ignore[call-arg]
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model),
                train_dataset=samples,
                args=sft_cfg,
            )
            trainer.train()

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        console.print(f"[green]LoRA saved to {output_dir}[/green]")

        if cfg.hf.push_checkpoints and cfg.hf.hub_repo_id:
            push_to_hub(
                output_dir,
                f"{cfg.hf.hub_repo_id}/{checkpoint_name}",
                token=cfg.hf.token,
            )

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Run: uv sync --all-groups[/yellow]")
    finally:
        finish_wandb()


# ── Inference commands ────────────────────────────────────────────────


@inference_app.command("vqa")
def inference_vqa(
    category: CategoryArg = "test",
    config: ConfigArg = None,
    checkpoint_dir: Annotated[
        str,
        typer.Option("--checkpoint-dir", help="Path to VQA LoRA checkpoint"),
    ] = "models/vqa",
    answer_types: Annotated[
        str,
        typer.Option("--answer-types", "-a", help="Comma-separated answer types"),
    ] = "factoid,list,paragraph,yes_no",
):
    """Run VQA inference with trained LoRA adapters."""
    cfg = load_config(config, category=category)
    types = [t.strip() for t in answer_types.split(",")]
    run_inference_stage(cfg, checkpoint_dir, types)


def run_inference_stage(
    cfg: SciVQAConfig,
    checkpoint_dir: str,
    answer_types: list[str],
) -> dict:
    """Run VQA inference. Returns the state dict."""
    print_stage_header("VQA Inference", 1, 1)
    console.print(f"  Checkpoints: {checkpoint_dir}")
    console.print(f"  Answer types: {', '.join(answer_types)}")
    console.print()

    try:
        from unsloth import FastVisionModel

        from staged_qwen3_5_scivqa.data import load_test_dataset
        from staged_qwen3_5_scivqa.models.inference import (
            run_inference,
            save_submission,
        )

        summary_cache = load_json_state(cfg.get_state_path("summary"))
        extraction_cache = load_json_state(cfg.get_state_path("table"))
        if isinstance(summary_cache, list):
            summary_cache = {}
        if isinstance(extraction_cache, list):
            extraction_cache = {}

        with progress_context("Loading dataset...") as progress:
            progress.add_task(description="Loading dataset...", total=None)
            samples = load_test_dataset(
                cfg.category, summary_cache, extraction_cache, answer_types
            )

        state_path = cfg.get_state_path("vqa")
        budget = cfg.get_stage_budget("yes_no")

        with progress_context("Loading model...") as progress:
            progress.add_task(description="Loading model...", total=None)
            model, tokenizer = FastVisionModel.from_pretrained(
                checkpoint_dir,
                load_in_4bit=cfg.model.load_in_4bit,
                max_seq_length=budget.max_sequence_length,
            )
            FastVisionModel.for_inference(model)

        with progress_context("Running inference...") as progress:
            progress.add_task(description="Running inference...", total=None)
            gen_kwargs = {
                "max_new_tokens": budget.max_new_tokens,
                "use_cache": True,
                "temperature": cfg.inference.temperature,
                "min_p": cfg.inference.min_p,
                "top_p": cfg.inference.top_p,
                "top_k": cfg.inference.top_k,
                "enable_thinking": cfg.inference.enable_thinking,
            }
            state = run_inference(model, tokenizer, samples, state_path, gen_kwargs)

        submission_path = cfg.paths.data_dir / f"submission_vqa_{cfg.category}.json"
        save_submission(state, submission_path)
        print_submission_summary(state, "VQA Inference")
        return state

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        return {}


# ── SMT commands ──────────────────────────────────────────────────────


@smt_app.command("run")
def smt_run(
    category: CategoryArg = "test",
    config: ConfigArg = None,
    vqa_state: Annotated[
        str | None,
        typer.Option("--vqa-state", help="Path to VQA state JSON"),
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="SMT model identifier"),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", help="Max retry attempts per question"),
    ] = None,
):
    """Run SMT-LIB grammar-constrained decoding pipeline."""
    cfg = load_config(
        config,
        **_filter_none(
            category=category,
            **{
                "smt.model_id": model_id,
                "smt.max_retries": max_retries,
            },
        ),
    )
    run_smt_stage(cfg, vqa_state)


def run_smt_stage(cfg: SciVQAConfig, vqa_state: str | None = None) -> None:
    """Run SMT pipeline stage."""
    state_path = Path(vqa_state) if vqa_state else cfg.get_state_path("vqa")
    output_path = cfg.get_state_path("smt")

    if stage_has_output(cfg, "smt"):
        print_skip_header("SMT")
        return

    print_stage_header("SMT Pipeline", 1, 1)
    console.print(f"  VQA state: {state_path}")
    console.print(f"  Model: {cfg.smt.model_id}")
    console.print(f"  Max retries: {cfg.smt.max_retries}")
    console.print(f"  Output: {output_path}")
    console.print()

    try:
        from staged_qwen3_5_scivqa.models.smt_runner import run_smt_pipeline

        summary_cache_path = cfg.get_state_path("summary")
        extraction_cache_path = cfg.get_state_path("table")

        run_smt_pipeline(
            category=cfg.category,
            output_path=output_path,
            summary_cache_path=summary_cache_path,
            extraction_cache_path=extraction_cache_path,
            model_id=cfg.smt.model_id,
            max_new_tokens=cfg.smt.max_new_tokens,
            temperature=cfg.smt.temperature,
            top_p=cfg.smt.top_p,
            top_k=cfg.smt.top_k,
            min_p=cfg.smt.min_p,
            presence_penalty=cfg.smt.presence_penalty,
            repetition_penalty=cfg.smt.repetition_penalty,
            max_retries=cfg.smt.max_retries,
        )

        console.print(f"[green]SMT state saved to {output_path}[/green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Run: uv sync --all-groups[/yellow]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")


# ── Reflect command ───────────────────────────────────────────────────


def reflect(
    category: str = "test",
    config: str | None = None,
    initial_state: str | None = None,
    smt_state: str | None = None,
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="Reflection model identifier"),
    ] = None,
):
    """Reflect on VQA answers using SMT solver output."""
    cfg = load_config(
        config,
        **_filter_none(
            category=category,
            **{
                "reflection.model_id": model_id,
            },
        ),
    )
    initial_path = Path(initial_state) if initial_state else cfg.get_state_path("vqa")
    smt_path = Path(smt_state) if smt_state else cfg.get_state_path("smt")
    submission_path = cfg.get_state_path("submission")
    reflection_path = cfg.get_state_path("reflection")

    if stage_has_output(cfg, "reflection"):
        print_skip_header("Reflection")
        return

    print_stage_header("Answer Reflection", 1, 1)
    console.print(f"  Initial state: {initial_path}")
    console.print(f"  SMT state: {smt_path}")
    console.print(f"  Model: {cfg.reflection.model_id}")
    console.print(f"  Output: {submission_path}")
    console.print()

    try:
        from staged_qwen3_5_scivqa.models.reflection_runner import run_reflection

        run_reflection(
            model_id=cfg.reflection.model_id,
            initial_state_path=initial_path,
            smt_state_path=smt_path,
            reflection_state_path=reflection_path,
            final_submission_path=submission_path,
            max_seq_length=cfg.reflection.max_sequence_length,
            load_in_4bit=cfg.reflection.load_in_4bit,
        )

        console.print(f"[green]Final submission saved to {submission_path}[/green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Run: uv sync --all-groups[/yellow]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")


# ── Eval commands ─────────────────────────────────────────────────────


@eval_app.command("vqa")
def eval_vqa(  # noqa: C901
    predictions: Annotated[
        str,
        typer.Option("--predictions", "-p", help="Path to prediction JSON"),
    ],
    category: CategoryArg = "test",
    config: ConfigArg = None,
):
    """Evaluate VQA predictions against ground truth."""
    cfg = load_config(config, category=category)
    pred_path = Path(predictions)

    print_stage_header("VQA Evaluation", 1, 1)
    console.print(f"  Predictions: {pred_path}")
    console.print()

    try:
        from staged_qwen3_5_scivqa.data import load_vqa_dataset
        from staged_qwen3_5_scivqa.evaluation.metrics import (
            compute_accuracy,
            compute_bert_score,
            compute_rouge,
            compute_set_f1,
        )

        pred_data = load_json_state(pred_path)
        if isinstance(pred_data, dict):
            pred_data = [pred_data]

        load_vqa_dataset(cfg.category)

        by_type: dict[str, dict[str, list]] = {}
        for sample in pred_data:
            if not isinstance(sample, dict):
                continue
            vqa = sample.get("vqa", {})
            for questions in vqa.values():
                for q in questions:
                    atype = q.get("answer_type", "unknown")
                    if atype not in by_type:
                        by_type[atype] = {"preds": [], "refs": []}
                    by_type[atype]["preds"].append(q.get("answer", ""))

        for atype, data in by_type.items():
            preds = data["preds"]
            refs = data["refs"]
            if not preds:
                continue

            metrics: dict[str, float] = {}
            if atype == "Yes/No":
                metrics["accuracy"] = compute_accuracy(preds, refs)
            elif atype == "Factoid":
                metrics["rouge_l"] = compute_rouge(preds, refs)["rougeL"]
            elif atype == "List":
                set_metrics = compute_set_f1(preds, refs)
                metrics.update(set_metrics)
            elif atype == "Paragraph":
                metrics["rouge_l"] = compute_rouge(preds, refs)["rougeL"]
                metrics["bertscore_f1"] = compute_bert_score(preds, refs)["f1"]

            print_metrics_table(metrics, f"{atype} Metrics")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")


@eval_app.command("summary")
def eval_summary(
    predictions: Annotated[
        str,
        typer.Option("--predictions", "-p", help="Path to prediction JSON"),
    ],
    category: CategoryArg = "test",
    config: ConfigArg = None,
):
    """Evaluate summarization predictions."""
    _ = load_config(config, category=category)

    print_stage_header("Summarization Evaluation", 1, 1)
    console.print(f"  Predictions: {predictions}")
    console.print()

    try:
        from staged_qwen3_5_scivqa.evaluation.metrics import (
            compute_bert_score,
            compute_rouge,
        )

        pred_data = load_json_state(Path(predictions))
        if isinstance(pred_data, dict):
            pred_data = [pred_data]
        preds = []
        refs = []
        for sample in pred_data:
            if not isinstance(sample, dict):
                continue
            for answers in sample.get("vqa", {}).values():
                if not isinstance(answers, list):
                    continue
                for a in answers:
                    preds.append(a.get("answer", ""))
                    refs.append(a.get("reference", ""))

        if preds:
            rouge = compute_rouge(preds, refs)
            bert = compute_bert_score(preds, refs)
            metrics = {
                "ROUGE-1": rouge["rouge1"],
                "ROUGE-2": rouge["rouge2"],
                "ROUGE-L": rouge["rougeL"],
                "BERTScore P": bert["precision"],
                "BERTScore R": bert["recall"],
                "BERTScore F1": bert["f1"],
            }
            print_metrics_table(metrics, "Summarization Metrics")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")


@eval_app.command("table")
def eval_table(
    predictions: Annotated[
        str,
        typer.Option("--predictions", "-p", help="Path to prediction JSON"),
    ],
    category: CategoryArg = "test",
    config: ConfigArg = None,
):
    """Evaluate table extraction predictions."""
    _ = load_config(config, category=category)

    print_stage_header("Table Extraction Evaluation", 1, 1)
    console.print(f"  Predictions: {predictions}")
    console.print()

    try:
        from staged_qwen3_5_scivqa.evaluation.metrics import compute_teds

        pred_data = load_json_state(Path(predictions))
        if isinstance(pred_data, dict):
            pred_data = [pred_data]
        preds = []
        refs = []
        for sample in pred_data:
            if not isinstance(sample, dict):
                continue
            for answers in sample.get("vqa", {}).values():
                if not isinstance(answers, list):
                    continue
                for a in answers:
                    preds.append(a.get("answer", ""))
                    refs.append(a.get("reference", ""))

        if preds:
            teds = compute_teds(preds, refs)
            print_metrics_table({"TEDS": teds}, "Table Extraction Metrics")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")


# ── Run command (end-to-end) ──────────────────────────────────────────


def run_pipeline(
    stages: Annotated[
        str,
        typer.Option(
            "--stages",
            "-s",
            help="Comma-separated: summary,table,vqa,smt,reflect",
        ),
    ] = "summary,table,vqa",
    category: CategoryArg = "test",
    config: ConfigArg = None,
    resume: ResumeArg = True,
):
    """Run the full pipeline end-to-end with optional resume."""
    cfg = load_config(config, category=category)
    stage_list = [s.strip() for s in stages.split(",")]
    valid_stages = {"summary", "table", "vqa", "smt", "reflect"}
    invalid = set(stage_list) - valid_stages
    if invalid:
        console.print(f"[red]Invalid stages: {invalid}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold magenta]Pipeline Run[/bold magenta]\n"
            f"Stages: {', '.join(stage_list)}  |  "
            f"Category: {cfg.category}  |  Resume: {resume}",
            border_style="magenta",
        )
    )
    console.print()

    for stage in stage_list:
        if resume and stage_has_output(cfg, stage):
            print_skip_header(stage)
            continue

        if stage == "summary":
            train_stage(cfg, "summary", cfg.paths.output_dir / "summary")
        elif stage == "table":
            train_stage(cfg, "table", cfg.paths.output_dir / "table")
        elif stage == "vqa":
            run_inference_stage(cfg, str(cfg.paths.output_dir / "vqa"), [])
        elif stage == "smt":
            run_smt_stage(cfg)
        elif stage == "reflect":
            reflect(category=cfg.category, config=config)

    console.print(
        Panel(
            "[bold green]Pipeline complete![/bold green]",
            border_style="green",
        )
    )


# ── Dataset commands ──────────────────────────────────────────────────


dataset_app = typer.Typer(help="Build and push cleaned HF datasets")


@dataset_app.command("build")
def dataset_build(
    task: Annotated[
        str,
        typer.Option("--task", "-t", help="Task type: vqa, summary, or table"),
    ],
    categories: Annotated[
        str,
        typer.Option("--categories", "-c", help="Comma-separated: train,dev,test"),
    ] = "train,dev,test",
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", "-r", help="HF dataset repo ID"),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="HF token (or use HF_TOKEN env)"),
    ] = None,
):
    """Build a cleaned DatasetDict from competition data and push to HF."""
    cats = [c.strip() for c in categories.split(",")]
    valid_tasks = {"vqa", "summary", "table"}
    if task not in valid_tasks:
        console.print(f"[red]Invalid task: {task}. Choose from {valid_tasks}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Building {task} dataset for categories: {cats}[/cyan]")
    if repo_id:
        console.print(f"[cyan]Target repo: {repo_id}[/cyan]")

    try:
        if task == "vqa":
            from staged_qwen3_5_scivqa.data import build_vqa_dataset

            build_vqa_dataset(tuple(cats), repo_id=repo_id, token=token)
        elif task == "summary":
            from staged_qwen3_5_scivqa.data import build_summary_dataset

            build_summary_dataset(tuple(cats), repo_id=repo_id, token=token)
        elif task == "table":
            from staged_qwen3_5_scivqa.data import build_table_dataset

            build_table_dataset(tuple(cats), repo_id=repo_id, token=token)

        console.print(f"[green]Dataset {task} built and pushed successfully![/green]")
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Run: uv sync --all-groups[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


# ── HF commands ───────────────────────────────────────────────────────


@hf_app.command("push")
def hf_push(
    path: Annotated[str, typer.Argument(help="Local path to push")],
    repo_id: Annotated[str, typer.Option("--repo-id", "-r", help="HF repo ID")],
    token: Annotated[
        str | None,
        typer.Option("--token", "-t", help="HF token (or use HF_TOKEN env)"),
    ] = None,
    repo_type: Annotated[
        str, typer.Option("--repo-type", help="model or dataset")
    ] = "model",
):
    """Push a checkpoint or directory to HuggingFace Hub."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Path not found: {p}[/red]")
        raise typer.Exit(1)
    if repo_type == "dataset":
        push_dataset(p, repo_id, token)
    else:
        push_to_hub(p, repo_id, token, repo_type)


@hf_app.command("pull")
def hf_pull(
    repo_id: Annotated[str, typer.Option("--repo-id", "-r", help="HF repo ID")],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Local output path")
    ] = "models",
    token: Annotated[str | None, typer.Option("--token", "-t", help="HF token")] = None,
    repo_type: Annotated[
        str, typer.Option("--repo-type", help="model or dataset")
    ] = "model",
):
    """Download a repo from HuggingFace Hub."""
    if repo_type == "dataset":
        pull_dataset(repo_id, Path(output), token)
    else:
        pull_from_hub(repo_id, Path(output), token, repo_type)


@hf_app.command("push-dataset")
def hf_push_dataset(
    path: Annotated[str, typer.Argument(help="Local dataset path")],
    repo_id: Annotated[str, typer.Option("--repo-id", "-r", help="HF dataset ID")],
    token: Annotated[str | None, typer.Option("--token", "-t", help="HF token")] = None,
):
    """Push a processed dataset to HuggingFace Hub."""
    push_dataset(Path(path), repo_id, token)


@hf_app.command("pull-dataset")
def hf_pull_dataset(
    repo_id: Annotated[str, typer.Option("--repo-id", "-r", help="HF dataset ID")],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Local output path")
    ] = "data",
    token: Annotated[str | None, typer.Option("--token", "-t", help="HF token")] = None,
):
    """Download a dataset from HuggingFace Hub."""
    pull_dataset(repo_id, Path(output), token)


# ── Helpers ───────────────────────────────────────────────────────────


def _filter_none(**kwargs) -> dict:
    """Remove None values from kwargs."""
    return {k: v for k, v in kwargs.items() if v is not None}
