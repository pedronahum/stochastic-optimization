"""Execute every notebook in notebooks/ against the LOCAL repo, headless.

The notebooks are Google Colab bootstraps: their first code cell does
`!pip install ...` and `git clone`s the published GitHub repo, then chdirs into
it. Run verbatim that would (a) risk overwriting the CUDA jaxlib with a CPU
wheel and (b) test the *remote* repo, not this working copy. So we replace that
bootstrap cell with a no-op and execute from the repo root, where `problems`/
`core` are importable from the installed editable package.

Usage:  python benchmarks/run_notebooks.py
Exit code is non-zero if any notebook fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "notebooks"

# Neutralised replacement for the Colab bootstrap cell.
BOOTSTRAP_STUB = (
    "# (local run) Colab bootstrap neutralised: deps are preinstalled and we\n"
    "# already run from the repo root, so no pip-install / git-clone needed.\n"
    "print('✓ Setup complete (local)')"
)


def _is_bootstrap(src: str) -> bool:
    return ("pip install" in src and "git clone" in src) or "os.chdir(" in src


def run_one(path: Path) -> tuple[str, str]:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and _is_bootstrap("".join(cell.source)):
            cell.source = BOOTSTRAP_STUB
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO)}},
    )
    try:
        client.execute()
        return ("PASS", "")
    except CellExecutionError as e:
        return ("FAIL", str(e).strip().splitlines()[-1][:200])
    except Exception as e:  # noqa: BLE001
        return ("ERROR", f"{type(e).__name__}: {e}".splitlines()[0][:200])


def main() -> int:
    nbs = sorted(NB_DIR.glob("*.ipynb"))
    failures = 0
    for nb in nbs:
        status, detail = run_one(nb)
        mark = "✓" if status == "PASS" else "✗"
        print(f"{mark} {nb.name:42s} {status}  {detail}")
        if status != "PASS":
            failures += 1
    print(f"\n{len(nbs) - failures}/{len(nbs)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
