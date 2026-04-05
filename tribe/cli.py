"""CLI entry point for Tribe."""

from __future__ import annotations

import sys

import click

from tribe import __version__


@click.group()
@click.version_option(version=__version__, prog_name="tribe")
def main() -> None:
    """Tribe — Content Manipulation Awareness Engine.

    Analyzes content and tells you what emotional response
    it's engineered to trigger.
    """


@main.command()
@click.argument("input_source")
@click.option(
    "--backend",
    type=click.Choice(["tribe", "rust", "cls", "auto"]),
    default="auto",
    help="Force analysis backend. auto = detect GPU.",
)
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON.")
@click.option("--verbose", is_flag=True, help="Show detailed technique and neural breakdown.")
@click.option("--quiet", is_flag=True, help="Single-line score output.")
def analyze(
    input_source: str,
    backend: str,
    output_json: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Analyze content for manipulation.

    INPUT_SOURCE can be a URL, file path, or "-" for stdin.

    \b
    Examples:
      tribe analyze https://example.com/article
      tribe analyze article.txt
      cat article.txt | tribe analyze -
      tribe analyze --json https://example.com/article
      tribe analyze --quiet https://example.com/article
    """
    force_backend = None if backend == "auto" else backend

    try:
        from tribe.analyze import analyze as run_analysis

        result = run_analysis(input_source, force_backend=force_backend)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ImportError as e:
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
    """Show available backends and hardware info."""
    from tribe.backends.router import detect_hardware

    hw = detect_hardware()

    click.echo("Tribe — Backend Status")
    click.echo("\u2500" * 40)
    click.echo()

    # Hardware
    click.echo("Hardware:")
    if hw.has_cuda:
        click.echo(f"  GPU: {hw.cuda_device_name} ({hw.vram_gb:.1f}GB VRAM) \u2713")
        click.echo(f"  CUDA: available \u2713")
    elif hw.has_mps:
        click.echo("  GPU: Apple Silicon (MPS) \u2713")
    else:
        click.echo("  GPU: none detected")

    click.echo()

    # Backends
    click.echo("Backends:")

    # Classifier is always available (just needs transformers)
    try:
        import transformers  # noqa: F401
        click.echo(
            "  Classifier (QCRI 18-technique + DistilRoBERTa emotion): "
            "\u2713 available"
        )
    except ImportError:
        click.echo("  Classifier: \u2717 transformers package not installed")

    # TRIBE v2 (official)
    if hw.can_run_tribe_v2:
        try:
            import tribev2  # noqa: F401
            click.echo("  TRIBE v2 (neural prediction): \u2713 available")
        except ImportError:
            click.echo(
                "  TRIBE v2: \u2717 tribev2 package not installed "
                "(pip install tribev2)"
            )
    else:
        click.echo("  TRIBE v2: \u2717 requires GPU with \u226510GB VRAM")

    # TRIBE v2 Rust (tribev2-rs binary + GGUF)
    try:
        from tribe.backends.tribe_v2_rust import TribeV2RustBackend

        rust_backend = TribeV2RustBackend(hw)
        if rust_backend.is_loaded():
            click.echo(
                "  TRIBE v2 Rust (llama-cpp Metal + eugenehp/tribev2): \u2713 available"
            )
        else:
            click.echo(
                "  TRIBE v2 Rust: \u2717 missing dependencies "
                "(tribev2-infer binary or LLaMA GGUF)"
            )
    except ImportError:
        click.echo("  TRIBE v2 Rust: \u2717 tribe_v2_rust module not found")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def serve(host: str, port: int) -> None:
    """Start the realtime demo server."""
    import uvicorn

    click.echo(f"Starting Tribe demo server on http://{host}:{port}")
    click.echo("Open http://localhost:8000 in your browser")
    uvicorn.run("tribe.server:app", host=host, port=port, reload=False)


@main.command()
def setup() -> None:
    """Download models and verify setup."""
    click.echo("Tribe \u2014 Content Manipulation Awareness Engine")
    click.echo("\u2500" * 46)
    click.echo()

    # Check hardware
    click.echo("Checking hardware...")
    from tribe.backends.router import detect_hardware

    hw = detect_hardware()
    if hw.has_cuda:
        click.echo(f"  GPU: {hw.cuda_device_name} ({hw.vram_gb:.1f}GB VRAM) \u2713")
    elif hw.has_mps:
        click.echo("  GPU: Apple Silicon (MPS) \u2713")
    else:
        click.echo("  GPU: none (classifier-only mode)")

    click.echo()
    click.echo("Verifying classifier models...")

    # Verify models by loading them directly (avoids pipeline tokenizer issues)
    try:
        from tribe.backends.classifier import ClassifierBackend

        click.echo("  [1/2] Technique detector (QCRI BERT, 18-class)...", nl=False)
        backend = ClassifierBackend()
        # _ensure_loaded() loads both the QCRI technique model and emotion pipeline
        backend._ensure_loaded()
        click.echo(" \u2713")

        click.echo("  [2/2] Emotion classifier (DistilRoBERTa)...", nl=False)
        click.echo(" \u2713")
    except Exception as e:
        click.echo(f" \u2717 Error: {e}", err=True)
        sys.exit(1)

    click.echo()

    # Quick self-test using hardcoded fixture text
    click.echo("Running self-test...")
    try:
        from tribe.backends.classifier import ClassifierBackend

        fixture = (
            "The government has FAILED to protect our children "
            "from this deadly threat. Act NOW before it's too late!"
        )
        backend = ClassifierBackend()
        result = backend.analyze_text(fixture)
        click.echo(
            f"  Classifier backend: \u2713 "
            f"({result.processing_time_ms}ms, "
            f"score: {result.manipulation_score}/10)"
        )
    except Exception as e:
        click.echo(f"  Classifier backend: \u2717 {e}", err=True)

    click.echo()

    # Check tribev2-rs binary
    click.echo("Checking tribev2-rs Rust binary (TRIBE v2 Rust backend)...")
    try:
        from tribe.backends.tribe_v2_rust import (
            _find_eugenehp_model_files,
            _find_llama_gguf,
            _find_rust_binary,
        )

        binary_path = _find_rust_binary()
        model_files = _find_eugenehp_model_files()
        gguf_path = _find_llama_gguf()

        if binary_path and model_files and gguf_path:
            click.echo(
                f"  tribev2-infer: \u2713 {binary_path.name} | "
                f"model: {model_files['weights'].name} | "
                f"gguf: {gguf_path.name}"
            )
        else:
            missing = []
            if not binary_path:
                missing.append("tribev2-infer binary")
            if not model_files:
                missing.append("eugenehp/tribev2 model")
            if not gguf_path:
                missing.append("LLaMA 3.2 3B GGUF (run: ollama pull llama3.2)")
            click.echo(f"  TRIBE v2 Rust: \u2717 missing: {', '.join(missing)}")
            if not binary_path:
                click.echo(
                    "    Build: cargo build --release --bin tribev2-infer "
                    "--features 'default,llama-metal' -p tribev2-rs"
                )
    except ImportError:
        pass

    click.echo()
    click.echo("Checking atlas files...")
    from pathlib import Path

    atlas_dir = Path(__file__).parent / "interpretation" / "atlas"
    lh_path = atlas_dir / "lh.Yeo2011_7Networks_N1000.annot"
    rh_path = atlas_dir / "rh.Yeo2011_7Networks_N1000.annot"
    if lh_path.exists() and rh_path.exists():
        import nibabel as nib

        try:
            nib.freesurfer.read_annot(str(lh_path))
            nib.freesurfer.read_annot(str(rh_path))
            size_kb = lh_path.stat().st_size // 1024
            click.echo(
                f"  Yeo 7-network atlas: \u2713 "
                f"(lh + rh, {size_kb}KB each)"
            )
        except Exception as e:
            click.echo(f"  Yeo 7-network atlas: \u2717 {e}", err=True)
    else:
        click.echo(
            "  Yeo 7-network atlas: \u2717 not found. "
            "Download from https://surfer.nmr.mgh.harvard.edu/fswiki/"
            "CorticalParcellation_Yeo2011"
        )

    click.echo()
    click.echo("Setup complete. Run `tribe analyze <url>` to analyze content.")


@main.command()
def version() -> None:
    """Show version and model info."""
    click.echo(f"Tribe v{__version__}")
    click.echo("Content Manipulation Awareness Engine")
    click.echo()
    click.echo("Models:")
    click.echo("  Technique: QCRI/PropagandaTechniquesAnalysis-en-BERT (18-class)")
    click.echo("  Emotion: j-hartmann/emotion-english-distilroberta-base")
    click.echo("  Neural: facebook/tribev2 (requires GPU)")
    click.echo(
        "  Neural Rust: eugenehp/tribev2 + llama-cpp-4 Metal (no approval needed)"
    )
    click.echo("  Atlas: Yeo2011 7-Network Parcellation (fsaverage5)")


# Allow running as `python -m tribe`
if __name__ == "__main__":
    main()
