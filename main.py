"""PPB Vision Module — entry point.

Delegates to the Typer CLI defined in cli.py.
Run with: python main.py <command> [options]
"""

from cli import app

if __name__ == "__main__":
    app()
