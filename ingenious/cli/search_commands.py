"""Command group registration for the application's CLI.

This module is responsible for aggregating and mounting various command groups
(sub-applications) onto the main Typer application. It employs lazy loading
for command modules to ensure a fast CLI startup time, only importing command
logic when it's explicitly invoked.

Key entry point:
- register_commands: Attaches all defined subcommands to the main app.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer
    from rich.console import Console


_AZURE_SEARCH_COMMAND_NAME: str = "azure-search"


def register_commands(app: "typer.Typer", console: "Console") -> None:
    """Register and mount the Azure Search commands.

    This function lazily imports the Azure Search CLI module and adds its
    Typer application as a subcommand to the main application instance.

    Args:
        app: The main Typer application object to which commands are added.
        console: The Rich console instance, maintained for a consistent
            registration interface but unused in this specific function.
    """
    from ingenious.services.azure_search import cli as search_cli

    app.add_typer(search_cli.app, name=_AZURE_SEARCH_COMMAND_NAME)
