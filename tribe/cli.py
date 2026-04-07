"""CLI entry point for Tribe."""

from __future__ import annotations

import sys

import click

from tribe import __version__


@click.group()
@click.version_option(version=__version__, prog_name="tribe")
def main() -> None:
    """Tribe — Neural Content Analysis Engine.

    Analyzes content using TRIBE v2 brain encoding and tells you
    what emotional response it's engineered to trigger.
    """


@main.command()
@click.argument("input_source")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON.")
@click.option("--verbose", is_flag=True, help="Show detailed neural breakdown.")
@click.option("--quiet", is_flag=True, help="Single-line score output.")
def analyze(
    input_source: str,
    output_json: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Analyze content for manipulation.

    INPUT_SOURCE can be a URL, file path, or "-" for stdin.

    Uses TRIBE v2 brain encoding model via the tribev2-infer Rust binary.
    Automatically uses Metal GPU when available, falls back to CPU.

    \b
    Examples:
      tribe analyze https://example.com/article
      tribe analyze article.txt
      cat article.txt | tribe analyze -
      tribe analyze --json https://example.com/article
      tribe analyze --quiet https://example.com/article
    """
    try:
        from tribe.analyze import analyze as run_analysis

        result = run_analysis(input_source)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {type(e).__name__}: {e}", err=True)
        sys.exit(1)

    # Render output
    if output_json:
        from tribe.output.json_output import render_json

        click.echo(render_json(result))
    elif quiet:
        from tribe.output.narrative import render_quiet

        click.echo(render_quiet(result))
    else:
        from tribe.output.narrative import render_narrative

        click.echo(render_narrative(result, verbose=verbose))


@main.command()
def backends() -> None:
    """Show backend status and hardware info."""
    from tribe.backends.router import detect_hardware

    hw = detect_hardware()

    click.echo("Tribe — Backend Status")
    click.echo("\u2500" * 40)
    click.echo()

    # Hardware
    click.echo("Hardware:")
    if hw.has_cuda:
        click.echo(f"  GPU: {hw.cuda_device_name} ({hw.vram_gb:.1f}GB VRAM) \u2713")
        click.echo("  Mode: CUDA GPU (fast)")
    elif hw.has_mps:
        click.echo("  GPU: Apple Silicon (MPS) \u2713")
        click.echo("  Mode: Metal GPU (fast, ~25s)")
    else:
        click.echo("  GPU: none detected")
        click.echo("  Mode: CPU (slower, ~2-5 min)")

    click.echo()

    # TRIBE v2 Rust backend
    click.echo("TRIBE v2 Rust:")
    try:
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        rust_backend = TribeV2RustBackend(hw)
        if rust_backend.is_loaded():
            click.echo("  Status: \u2713 available")
        else:
            click.echo("  Status: \u2717 missing components")
            click.echo()
            click.echo("  Setup:")
            click.echo("    1. git clone https://github.com/eugenehp/tribev2-rs /tmp/tribev2-rs")
            if hw.has_mps:
                click.echo(
                    "    2. cd /tmp/tribev2-rs && cargo build --release "
                    '--bin tribev2-infer --features "default,llama-metal"'
                )
            else:
                click.echo(
                    "    2. cd /tmp/tribev2-rs && cargo build --release "
                    "--bin tribev2-infer --features default"
                )
            click.echo("    3. ollama pull llama3.2")
    except Exception:
        click.echo("  Status: \u2717 module error")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def serve(host: str, port: int) -> None:
    """Start the realtime demo server."""
    import uvicorn

    click.echo(f"Starting Tribe demo server on http://{host}:{port}")
    click.echo("Open http://localhost:8000 in your browser")
    uvicorn.run("tribe.server:app", host=host, port=port, reload=False)


@main.group()
def bench() -> None:
    """Run benchmarks against manipulation datasets."""


@bench.command()
@click.option(
    "--dataset",
    type=click.Choice(["semeval", "mentalmanip", "paired", "all"]),
    default="all",
)
def run(dataset: str) -> None:
    """Run benchmarks. Requires tribev2-infer binary."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from tribe.benchmarks.runner import run_all, run_benchmark

    if dataset == "all":
        results = run_all()
        for name, r in results.items():
            _print_summary(name, r)
    else:
        result = run_benchmark(dataset)
        _print_summary(dataset, result)


@bench.command()
def download() -> None:
    """Download benchmark datasets."""
    from pathlib import Path

    from tribe.benchmarks.datasets.mentalmanip import download as dl_mental
    from tribe.benchmarks.datasets.semeval import download as dl_semeval

    data_dir = Path(__file__).parent / "benchmarks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    click.echo("Downloading SemEval-2020 Task 11...")
    dl_semeval(data_dir)
    click.echo("  Done.")

    click.echo("Downloading MentalManip...")
    dl_mental(data_dir)
    click.echo("  Done.")


@bench.command()
def visualize() -> None:
    """Generate SVG visualizations from existing results."""
    from pathlib import Path

    from tribe.benchmarks.visualize import generate_all

    results_dir = Path(__file__).parent / "benchmarks" / "results"
    output_dir = Path(__file__).parent.parent / "images"
    generate_all(results_dir, output_dir)
    click.echo(f"Visualizations saved to {output_dir}")


@bench.command()
def results() -> None:
    """Show benchmark results summary."""
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent / "benchmarks" / "results"
    for name in ["paired", "mentalmanip", "semeval"]:
        path = results_dir / f"{name}_results.json"
        if path.exists():
            data = json.loads(path.read_text())
            _print_summary(name, data)
        else:
            click.echo(f"\n{name}: no results yet (run: tribe bench run --dataset {name})")


@bench.command(name="collect")
@click.option(
    "--dataset",
    type=click.Choice(["semeval", "mentalmanip", "paired", "qbias"]),
    required=True,
)
def collect_cmd(dataset: str) -> None:
    """Collect raw activation vectors for classifier training."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from tribe.benchmarks.classifier import collect_activations

    X, y, ids = collect_activations(dataset)
    click.echo(f"Collected {len(ids)} activations, shape {X.shape}")
    click.echo(f"  Manipulative: {sum(y)}, Neutral: {len(y) - sum(y)}")


@bench.command(name="train-classifier")
@click.option("--n-components", default=50, help="PCA components")
def train_cmd(n_components: int) -> None:
    """Train manipulation classifier from collected activations."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load all collected activations
    import numpy as np

    from tribe.benchmarks.classifier import CLASSIFIER_DIR, train_classifier

    all_X = []
    all_y = []

    for dataset in ["paired", "semeval", "mentalmanip"]:
        act_dir = CLASSIFIER_DIR / f"{dataset}_activations"
        if not act_dir.exists():
            continue

        from tribe.benchmarks.runner import _item_label, _load_dataset

        items = _load_dataset(dataset, CLASSIFIER_DIR.parent / "data")
        item_labels = {item["id"]: _item_label(item) for item in items}

        count = 0
        for npy in sorted(act_dir.glob("*.npy")):
            item_id = npy.stem
            if item_id in item_labels:
                all_X.append(np.load(str(npy)))
                all_y.append(item_labels[item_id])
                count += 1

        click.echo(f"  {dataset}: {count} activations loaded")

    if not all_X:
        click.echo("No activations found. Run: tribe bench collect --dataset <name>")
        return

    X = np.stack(all_X)
    y = np.array(all_y)
    click.echo(f"Total: {len(X)} samples ({sum(y)} manip, {len(y) - sum(y)} neutral)")

    results = train_classifier(X, y, n_components=n_components)
    click.echo(f"\nCV AUC: {results['cv_auc_mean']:.4f} +/- {results['cv_auc_std']:.4f}")
    click.echo(f"Model saved to: {results['model_path']}")


def _print_summary(name: str, data: dict) -> None:
    """Print a summary of benchmark results."""
    click.echo(f"\n{'=' * 50}")
    click.echo(f"  {name.upper()} -- {data['n_successful']}/{data['n_total']} texts analyzed")
    click.echo(f"{'=' * 50}")
    metrics = data.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, float):
            click.echo(f"  {key}: {value:.4f}")
        else:
            click.echo(f"  {key}: {value}")


@main.command()
def version() -> None:
    """Show version and model info."""
    click.echo(f"Tribe v{__version__}")
    click.echo("Neural Content Analysis Engine")
    click.echo()
    click.echo("Models:")
    click.echo("  Neural: eugenehp/tribev2 + tribev2-infer Rust binary")
    click.echo("  Text features: LLaMA 3.2 3B (via llama.cpp, GGUF)")
    click.echo("  Atlas: Yeo2011 7-Network Parcellation (fsaverage5)")


# Allow running as `python -m tribe`
if __name__ == "__main__":
    main()
