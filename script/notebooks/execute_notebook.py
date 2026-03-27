from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path, *, timeout: int = 3600) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    execute_notebook(args.notebook.resolve(), timeout=args.timeout)
    print(f"Executed {args.notebook}")


if __name__ == "__main__":
    main()
