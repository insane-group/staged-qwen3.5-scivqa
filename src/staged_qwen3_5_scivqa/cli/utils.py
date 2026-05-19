"""CLI utility helpers: Rich formatting, HF Hub integration, W&B setup."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

if TYPE_CHECKING:
    from staged_qwen3_5_scivqa.settings import PipelineConfig

console = Console()


def print_stage_header(stage: str, index: int, total: int) -> None:
    """Print a Rich panel header for a pipeline stage."""
    console.print(
        Panel(
            f"[bold cyan]Stage {index}/{total}: {stage.upper()}[/bold cyan]",
            border_style="cyan",
        )
    )


def print_skip_header(stage: str, reason: str = "output already exists") -> None:
    """Print a Rich panel indicating a stage is being skipped."""
    console.print(
        Panel(
            f"[bold yellow]Skipping {stage.upper()}[/bold yellow] — {reason}",
            border_style="yellow",
        )
    )


def print_metrics_table(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Print a Rich table of evaluation metrics."""
    table = Table(title=title, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    console.print(table)


def print_submission_summary(state: dict, title: str = "Submission") -> None:
    """Print a Rich summary of submission state."""
    total_samples = len(state)
    total_questions = sum(
        len(answers) for sub_figs in state.values() for answers in sub_figs.values()
    )
    console.print(
        Panel(
            f"[bold green]{title}[/bold green]\n"
            f"Samples: {total_samples}  |  Questions: {total_questions}",
            border_style="green",
        )
    )


def progress_context(description: str) -> Progress:
    """Create a Rich progress context with spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def setup_wandb(config) -> bool:
    """Initialize W&B run if enabled. Returns True if successful."""
    if not config.enabled:
        return False
    try:
        import wandb

        init_kwargs: dict[str, Any] = {
            "project": config.project,
        }
        if config.entity:
            init_kwargs["entity"] = config.entity
        if config.run_name:
            init_kwargs["name"] = config.run_name
        wandb.init(**init_kwargs)
        console.print("[green]W&B logging enabled[/green]")
        return True
    except ImportError:
        console.print("[yellow]W&B not installed. Install with: uv add wandb[/yellow]")
        return False


def finish_wandb() -> None:
    """Finish the current W&B run if active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def push_to_hub(
    path: Path,
    repo_id: str,
    token: str | None = None,
    repo_type: str = "model",
) -> str:
    """Push a directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    console.print(f"[cyan]Pushing {path} to {repo_id} ({repo_type})...[/cyan]")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type=repo_type,
    )
    url = f"https://huggingface.co/{repo_id}"
    console.print(f"[green]Uploaded to {url}[/green]")
    return url


def pull_from_hub(
    repo_id: str,
    output: Path,
    token: str | None = None,
    repo_type: str = "model",
) -> Path:
    """Download a repo from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    console.print(f"[cyan]Downloading {repo_id} to {output}...[/cyan]")
    output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(output),
        repo_type=repo_type,
        token=token,
    )
    console.print(f"[green]Downloaded to {path}[/green]")
    return Path(path)


def push_dataset(
    path: Path,
    repo_id: str,
    token: str | None = None,
) -> str:
    """Push a dataset directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    console.print(f"[cyan]Pushing dataset {path} to {repo_id}...[/cyan]")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    console.print(f"[green]Dataset uploaded to {url}[/green]")
    return url


def pull_dataset(
    repo_id: str,
    output: Path,
    token: str | None = None,
) -> Path:
    """Download a dataset from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    console.print(f"[cyan]Downloading dataset {repo_id} to {output}...[/cyan]")
    output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(output),
        repo_type="dataset",
        token=token,
    )
    console.print(f"[green]Dataset downloaded to {path}[/green]")
    return Path(path)


def stage_has_output(config: "PipelineConfig", stage: str) -> bool:
    """Check if a pipeline stage already has output (for resume)."""
    state_path = config.get_state_path(stage)
    return bool(state_path.exists())


def load_json_state(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    """Load a JSON state file."""
    if not path.exists():
        console.print(f"[red]State file not found: {path}[/red]")
        return {}
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def save_json_state(path: Path, data: dict) -> None:
    """Save a JSON state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Saved state to {path}[/green]")
