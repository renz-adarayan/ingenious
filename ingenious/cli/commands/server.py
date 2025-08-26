"""
Server-related CLI commands for Insight Ingenious.

This module contains commands for starting and managing the API server.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ingenious.cli.base import BaseCommand, CommandError, ExitCode
from ingenious.cli.utilities import ValidationUtils

if TYPE_CHECKING:
    import typer
    from rich.console import Console


class ServeCommand(BaseCommand):
    """Start the API server with web interface."""

    def execute(
        self,
        config: Optional[str] = None,
        profile: Optional[str] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        no_prompt_tuner: bool = False,
        **kwargs: object,
    ) -> None:
        """
        Start the Insight Ingenious API server with web interface.

        Args:
            config: Path to config.yml file
            profile: Path to profiles.yml file
            host: Host to bind the server
            port: Port to bind the server
            no_prompt_tuner: Whether to disable the prompt tuner interface
        """
        # Resolve port
        if port is None:
            port = int(os.getenv("WEB_PORT", "80"))

        # Validate port
        is_valid_port, port_error = ValidationUtils.validate_port(port)
        if not is_valid_port:
            raise CommandError(f"Invalid port: {port_error}", ExitCode.VALIDATION_ERROR)

        # Validate configuration paths
        try:
            self.validate_config_paths(config, profile)
        except CommandError:
            self.print_error("Configuration validation failed")
            self._show_config_help()
            raise

        self.start_progress("Starting API server...")

        try:
            # Import and start the server
            import uvicorn

            from ingenious.config.main_settings import IngeniousSettings
            from ingenious.main import create_app

            # Load settings
            settings = IngeniousSettings()
            app = create_app(settings)

            self.stop_progress()
            self.print_success(f"Starting server on {host}:{port}")

            # Start the server
            uvicorn.run(app, host=host, port=port)

        except ImportError as e:
            self.stop_progress()
            raise CommandError(
                f"Failed to import server dependencies: {e}",
                ExitCode.MISSING_DEPENDENCY,
            )
        except Exception as e:
            self.stop_progress()
            raise CommandError(f"Failed to start server: {e}", ExitCode.GENERAL_ERROR)

    def _show_config_help(self) -> None:
        """Show configuration help for server startup."""
        self.console.print(
            "\n[bold yellow]💡 Configuration Requirements:[/bold yellow]"
        )
        self.console.print("1. Set environment variables:")
        self.console.print("   export INGENIOUS_PROJECT_PATH=$(pwd)/config.yml")
        self.console.print("   export INGENIOUS_PROFILE_PATH=$(pwd)/profiles.yml")
        self.console.print("2. Ensure config files exist:")
        self.console.print("   Run: ingen init")
        self.console.print("3. Configure Azure OpenAI credentials in .env")


# Backward compatibility
def register_commands(app: typer.Typer, console: Console) -> None:
    """Register server-related commands with the typer app."""
    from typing_extensions import Annotated

    @app.command(name="serve", help="Start the API server with web interface")
    def serve(
        config: Annotated[
            Optional[str],
            typer.Option(
                "--config",
                "-c",
                help="Path to config.yml file (default: ./config.yml or $INGENIOUS_PROJECT_PATH)",
            ),
        ] = None,
        profile: Annotated[
            Optional[str],
            typer.Option(
                "--profile",
                "-p",
                help="Path to profiles.yml file (default: ./profiles.yml or $INGENIOUS_PROFILE_PATH)",
            ),
        ] = None,
        host: Annotated[
            str,
            typer.Option(
                "--host", "-h", help="Host to bind the server (default: 0.0.0.0)"
            ),
        ] = "127.0.0.1",
        port: Annotated[
            int,
            typer.Option(
                "--port", help="Port to bind the server (default: 80 or $WEB_PORT)"
            ),
        ] = int(os.getenv("WEB_PORT", "80")),
        no_prompt_tuner: Annotated[
            bool,
            typer.Option(
                "--no-prompt-tuner", help="Disable the prompt tuner interface"
            ),
        ] = False,
    ) -> None:
        """Start the Insight Ingenious API server with web interface."""
        cmd = ServeCommand(console)
        cmd.run(
            config=config,
            profile=profile,
            host=host,
            port=port,
            no_prompt_tuner=no_prompt_tuner,
        )
