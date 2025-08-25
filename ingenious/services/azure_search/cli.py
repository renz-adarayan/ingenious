"""CLI for Advanced Azure AI Search.

This module provides a Typer-based command-line interface that orchestrates a
multi-stage search pipeline over an Azure AI Search index. It supports classic
retrieval, result fusion (DAT), semantic reranking, and generative answering
(RAG). The default command is `run`, so users can execute queries directly
(e.g., `azure-search "my query"`). Key entry points are the `run_search`
command and the lazy-loaded `build_search_pipeline` shim used by tests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import click
import typer
from pydantic import SecretStr, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typer.core import TyperGroup

# ── Local imports ────────────────────────────────────────────────────────────
from .config import DEFAULT_DAT_PROMPT, SearchConfig

# ── Module constants (replace magic values) ──────────────────────────────────
APP_NAME = "azure-search"
APP_HELP = "CLI interface for the Ingenious Advanced Azure AI Search service."
RUN_COMMAND_NAME = "run"
DEFAULT_MAX_CONTENT_WIDTH = 200
STATUS_SPINNER_NAME = "dots"
CONTENT_SAMPLE_PREVIEW_LEN = 250
DEFAULT_OPENAI_API_VERSION = "2024-02-01"
DEFAULT_LOGGING_LEVEL = logging.WARNING

# ── Lazy loader for the heavy pipeline ───────────────────────────────────────
_build_pipeline_impl: Callable[..., Any] | None = None


def _get_build_pipeline_impl() -> Callable[..., Any]:
    """Import and return the pipeline factory lazily.

    Avoids importing heavy ML deps (e.g., torch/transformers) when users only
    need help/usage. This keeps the CLI snappy for `--help` and similar flows.
    """
    global _build_pipeline_impl
    if _build_pipeline_impl is None:
        from .components.pipeline import build_search_pipeline as _impl

        _build_pipeline_impl = _impl
    return _build_pipeline_impl


def build_search_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Shim around the pipeline factory for test patching.

    Tests patch this symbol directly without interfering with lazy-loading.
    The logic delegates to the lazily imported implementation.

    Args:
        *args: Positional arguments for the pipeline factory.
        **kwargs: Keyword arguments for the pipeline factory.

    Returns:
        Any: The constructed pipeline instance.
    """
    return _get_build_pipeline_impl()(*args, **kwargs)


class DefaultToRunTyperGroup(TyperGroup):
    """TyperGroup that forwards to the 'run' command by default.

    Improves UX by allowing `azure-search "query"` to call the `run` command
    implicitly. Explicit subcommands and group-level options behave normally.
    """

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve command, defaulting to 'run' if none is found.

        The logic checks for group options (e.g., --help) or explicit
        subcommands first. If none, it forwards remaining args to `run`.

        Args:
            ctx: Click context.
            args: Raw CLI arguments after the executable.

        Returns:
            Tuple of (command name, command object, remaining args).
        """
        # Group option like --help: let the group handle it.
        if args and args[0].startswith("-"):
            return super().resolve_command(ctx, args)

        # Explicit subcommand present: use it.
        if args:
            maybe_cmd = self.get_command(ctx, args[0])
            if maybe_cmd is not None:
                return args[0], maybe_cmd, args[1:]

        # Otherwise, forward to 'run' with remaining args.
        cmd = self.get_command(ctx, RUN_COMMAND_NAME)
        if cmd is not None:
            return RUN_COMMAND_NAME, cmd, args

        # Fallback (should not occur since 'run' is defined).
        return super().resolve_command(ctx, args)


# Initialize Typer app (backed by our custom Typer group) and Rich console
app = typer.Typer(
    name=APP_NAME,
    help=APP_HELP,
    cls=DefaultToRunTyperGroup,  # subclass of TyperGroup (satisfies tests)
    context_settings={
        "max_content_width": DEFAULT_MAX_CONTENT_WIDTH,
        # Route `azure-search q --opts` → `run q --opts` safely.
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
console = Console()

# Configure basic logging (overridden by `setup_logging`)
logging.basicConfig(level=DEFAULT_LOGGING_LEVEL)


def setup_logging(verbose: bool) -> None:
    """Configure logging levels for application components.

    Sets DEBUG if verbose is True, otherwise INFO, on key modules used in the
    search service. Keeps root logger aligned when verbose is enabled.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging.
    """
    level = logging.DEBUG if verbose else logging.INFO

    loggers = [
        "ingenious.services.azure_search.pipeline",
        "ingenious.services.azure_search.components.retrieval",
        "ingenious.services.azure_search.components.fusion",
        "ingenious.services.azure_search.components.generation",
        __name__,  # Include CLI logger itself
    ]

    for logger_name in loggers:
        try:
            logging.getLogger(logger_name).setLevel(level)
        except Exception:
            # Package structure differences or missing loggers are non-fatal.
            pass

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def _run_search_pipeline(config: SearchConfig, query: str, verbose: bool) -> None:
    """Create an event loop and execute the search pipeline.

    Encapsulates the async run to ensure proper initialization and cleanup of
    resources. Displays the answer and relevant source chunks using Rich.

    Args:
        config: Validated search configuration model.
        query: Natural language search query string.
        verbose: Whether to display full exception tracebacks.
    """

    async def _async_run() -> None:
        """Build, run, and clean up the search pipeline asynchronously.

        Handles configuration errors explicitly and runtime errors generically.
        Always attempts to close pipeline clients on exit.
        """
        pipeline: Any | None = None
        try:
            # Build the pipeline using the factory
            pipeline = build_search_pipeline(config)

            # Status line visible in captured output for tests
            console.print("Executing Advanced Search Pipeline", markup=False)

            # Execute the pipeline
            result: dict[str, Any]
            status_text = (
                "[bold green]Executing Advanced Search Pipeline "
                "(L1 -> DAT -> L2 -> RAG)..."
            )
            with console.status(status_text, spinner=STATUS_SPINNER_NAME):
                result = await pipeline.get_answer(query)

            # Display Results
            answer: str = result.get("answer", "No answer generated.")
            sources: list[dict[str, Any]] = result.get("source_chunks", [])

            console.print(
                Panel(
                    Markdown(answer),
                    title="[bold green]:robot: Answer[/bold green]",
                    border_style="green",
                )
            )

            # Display Sources
            console.print(f"\n[bold]Sources Used ({len(sources)}):[/bold]")
            for i, source in enumerate(sources):
                score: float | str = source.get("_final_score", "N/A")
                content_field = config.content_field
                content_text: str = source.get(content_field, "")
                content_sample = (
                    content_text[:CONTENT_SAMPLE_PREVIEW_LEN] + "..."
                    if content_text
                    else ""
                )

                score_display = (
                    f"{score:.4f}" if isinstance(score, float) else str(score)
                )

                console.print(
                    Panel(
                        content_sample,
                        title=(
                            f"[bold cyan]Chunk {i + 1} "
                            f"(Score: {score_display} | "
                            f"Type: {source.get('_retrieval_type', 'N/A')})[/bold cyan]"
                        ),
                        border_style="cyan",
                        expand=False,
                    )
                )

        except ValueError as ve:
            console.print(
                Panel(
                    f"Configuration failed: {ve}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
            # Tests expect exit(1) on configuration errors
            raise typer.Exit(code=1)
        except Exception as e:
            # Handle runtime errors
            if verbose:
                console.print_exception(show_locals=True)
            console.print(
                Panel(
                    (
                        f"Pipeline execution failed: {e}\n"
                        "[dim]Run with --verbose for details.[/dim]"
                    ),
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
        finally:
            # Ensure clients are closed
            if pipeline:
                await pipeline.close()
                logging.info("Pipeline clients closed.")

    asyncio.run(_async_run())


@app.callback()
def _callback() -> None:
    """Entry point for the Typer application group.

    Ensures that the group-level help message is displayed correctly. The
    default-to-run behavior is implemented by DefaultToRunTyperGroup.
    """
    # Default-to-run is implemented in DefaultToRunTyperGroup.resolve_command
    return None


@app.command(
    name=RUN_COMMAND_NAME,
    # Accept options after positional args, e.g.: run "q" --search-endpoint ...
    context_settings={"allow_interspersed_args": True},
)
def run_search(
    # ── Azure AI Search Configuration ─────────────────────────────────────────
    search_endpoint: str = typer.Option(
        ...,
        "--search-endpoint",
        "-se",
        envvar="AZURE_SEARCH_ENDPOINT",
        help="Azure AI Search Endpoint URL.",
    ),
    search_key: str = typer.Option(
        ...,
        "--search-key",
        "-sk",
        envvar="AZURE_SEARCH_KEY",
        help="Azure AI Search API Key.",
        prompt=True,
        hide_input=True,
    ),
    search_index_name: str | None = typer.Option(
        None,
        "--search-index-name",
        "-si",
        help="Azure AI Search index name to use.",
        envvar="AZURE_SEARCH_INDEX_NAME",
        show_envvar=True,
    ),
    # ── Azure OpenAI Configuration ────────────────────────────────────────────
    openai_endpoint: str = typer.Option(
        ...,
        "--openai-endpoint",
        "-oe",
        envvar="AZURE_OPENAI_ENDPOINT",
        help="Azure OpenAI Endpoint URL.",
    ),
    openai_key: str = typer.Option(
        ...,
        "--openai-key",
        "-ok",
        envvar="AZURE_OPENAI_KEY",
        help="Azure OpenAI API Key.",
        prompt=True,
        hide_input=True,
    ),
    embedding_deployment: str = typer.Option(
        ...,
        "--embedding-deployment",
        "-ed",
        envvar="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        help="Embedding model deployment name.",
    ),
    generation_deployment: str = typer.Option(
        ...,
        "--generation-deployment",
        "-gd",
        envvar="AZURE_OPENAI_GENERATION_DEPLOYMENT",
        help="Generation model deployment name (used for DAT and RAG).",
    ),
    # ── Pipeline Behavior Configuration ───────────────────────────────────────
    top_k_retrieval: int = typer.Option(
        20,
        "--top-k-retrieval",
        "-k",
        help="Number of initial results to fetch (K).",
    ),
    use_semantic_ranking: bool = typer.Option(
        True,
        "--semantic-ranking/--no-semantic-ranking",
        help="Enable/Disable Azure Semantic Ranking (L2).",
    ),
    semantic_config_name: str | None = typer.Option(
        None,
        "--semantic-config",
        "-sc",
        envvar="AZURE_SEARCH_SEMANTIC_CONFIG",
        help="Semantic configuration name (required if using semantic ranking).",
    ),
    top_n_final: int = typer.Option(
        5,
        "--top-n-final",
        "-n",
        help="Number of final chunks for generation (N).",
    ),
    openai_version: str = typer.Option(
        DEFAULT_OPENAI_API_VERSION,
        "--openai-version",
        "-ov",
        help="Azure OpenAI API Version.",
    ),
    dat_prompt_file: str | None = typer.Option(
        None,
        "--dat-prompt-file",
        "-dp",
        help="Path to a custom DAT prompt file (overrides default).",
    ),
    generate: bool = typer.Option(
        False,
        "--generate/--no-generate",
        envvar="AZURE_SEARCH_ENABLE_GENERATION",
        help="Enable/disable final answer generation (default: disabled).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
    # ── Positional query (must be last when invoking) ─────────────────────────
    query: str = typer.Argument(..., help="The search query string."),
) -> None:
    """Execute the advanced AI search pipeline.

    Orchestrates retrieval, Dynamic Alternating Transformation (DAT) for query
    fusion, semantic reranking, and final answer generation. Gathers config
    from options/env, validates it, and runs the pipeline.

    Args:
        search_endpoint: Azure AI Search endpoint URL.
        search_key: Azure AI Search API key (prompted if not provided).
        search_index_name: Azure AI Search index name.
        openai_endpoint: Azure OpenAI endpoint URL.
        openai_key: Azure OpenAI API key (prompted if not provided).
        embedding_deployment: Embedding model deployment name.
        generation_deployment: Generation model deployment name.
        top_k_retrieval: Initial retrieval depth (K).
        use_semantic_ranking: Toggle Azure semantic ranking (L2).
        semantic_config_name: Semantic config name when using ranking.
        top_n_final: Number of chunks used for generation (N).
        openai_version: Azure OpenAI API version.
        dat_prompt_file: Path to custom DAT prompt file.
        generate: Toggle final answer generation (RAG).
        verbose: Toggle verbose logging (DEBUG).
        query: The search query string.

    Raises:
        typer.Exit: On configuration/validation errors (exit code 1).
    """
    setup_logging(verbose)

    console.print(f"\nStarting search for: '[bold]{query}[/bold]'\n", markup=False)

    # Guardrail for tests: if semantic ranking is enabled, name must be supplied
    if use_semantic_ranking and not semantic_config_name:
        typer.echo(
            "Error: Semantic ranking is enabled but no semantic configuration name "
            "was provided.\nSupply --semantic-config or set "
            "AZURE_SEARCH_SEMANTIC_CONFIG."
        )
        raise typer.Exit(code=1)

    # Handle DAT prompt loading
    dat_prompt: str = DEFAULT_DAT_PROMPT
    if dat_prompt_file:
        try:
            with open(dat_prompt_file, "r") as f:
                dat_prompt = f.read()
            logging.info(f"Loaded custom DAT prompt from {dat_prompt_file}")
        except FileNotFoundError:
            # Plain, stable message for tests to assert reliably
            typer.echo("Error: DAT prompt file not found")
            raise typer.Exit(code=1)

    # Build the configuration object
    try:
        config = SearchConfig(
            search_endpoint=search_endpoint,
            search_key=SecretStr(search_key),
            search_index_name=search_index_name,
            semantic_configuration_name=semantic_config_name,
            openai_endpoint=openai_endpoint,
            openai_key=SecretStr(openai_key),
            openai_version=openai_version,
            embedding_deployment_name=embedding_deployment,
            generation_deployment_name=generation_deployment,
            top_k_retrieval=top_k_retrieval,
            use_semantic_ranking=use_semantic_ranking,
            top_n_final=top_n_final,
            dat_prompt=dat_prompt,
            enable_answer_generation=generate,
        )
    except ValidationError as e:
        console.print(f"[bold red]Configuration Validation Error:[/bold red]\n{e}")
        raise typer.Exit(code=1)

    _run_search_pipeline(config, query, verbose)
