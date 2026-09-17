"""Execute a capstone lab notebook headlessly and save it with its outputs.

Usage:  ../.venv/bin/python run_lab.py <notebook.ipynb> [--timeout SECONDS]

The labs begin with `%pip install numpy==1.26 tensorflow==2.19 ...` cells that would
downgrade the project venv, so those cells (and the cell that inspects their captured
output) are skipped; every other code cell runs top to bottom in one kernel.

Each notebook executes in its own `.lab_runs/<notebook>/` directory (dataset link, downloaded
and trained model files land there); only the executed notebook is written back in place.
"""
import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Keep TensorFlow's C++ startup chatter out of the saved cell outputs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def is_install_cell(cell):
    return cell.cell_type == "code" and ("pip install" in cell.source or "captured_output" in cell.source)


def collapse_progress_bars(cell):
    """Store stream output the way a Jupyter frontend displays it.

    tqdm redraws its bar with carriage returns and every redraw is recorded as a separate
    stream chunk. Merge adjacent chunks of the same stream, then keep only the text after the
    last carriage return on each line, i.e. the final state of each bar.
    """
    merged = []
    for out in cell.outputs:
        if (out.output_type == "stream" and merged and merged[-1].output_type == "stream"
                and merged[-1].name == out.name):
            merged[-1].text += out.text
        else:
            merged.append(out)
    for out in merged:
        if out.output_type == "stream" and "\r" in out.text:
            out.text = "\n".join(line.rsplit("\r", 1)[-1] for line in out.text.split("\n"))
    cell.outputs = merged


def normalize(nb):
    """Fix schema quirks in the IBM originals so the saved notebook validates."""
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
        elif cell.cell_type == "code":  # skipped cells may lack these required keys
            cell.setdefault("execution_count", None)
            cell.setdefault("outputs", [])
            collapse_progress_bars(cell)
    if all("id" in cell for cell in nb.cells):
        nb.nbformat_minor = max(nb.nbformat_minor, 5)  # cell ids need nbformat >= 4.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="per-cell timeout in seconds")
    args = parser.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    skip = {i for i, cell in enumerate(nb.cells) if is_install_cell(cell)}
    print(f"skipping install cells: {sorted(skip)}", flush=True)

    # Run in a private directory. Every lab's download cell deletes and recreates
    # ./images_dataSAT, so two notebooks sharing one directory (e.g. this run and one started
    # from an IDE) can pull the dataset out from under each other mid-training.
    nb_dir = os.path.dirname(os.path.abspath(args.notebook))
    stem = os.path.splitext(os.path.basename(args.notebook))[0].replace(" ", "_")
    workdir = os.path.join(nb_dir, ".lab_runs", stem)
    os.makedirs(workdir, exist_ok=True)
    print(f"working directory: {workdir}", flush=True)

    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": workdir}})
    with client.setup_kernel():
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or idx in skip:
                continue
            start = time.time()
            try:
                client.execute_cell(cell, idx)
            except CellExecutionError as err:
                print(f"\nFAILED at cell {idx} -- notebook NOT saved\n{str(err)[-3000:]}", flush=True)
                sys.exit(1)
            print(f"cell {idx:3d} ok  exec_count={cell.execution_count}  "
                  f"outputs={len(cell.outputs)}  {time.time() - start:7.1f}s", flush=True)

    normalize(nb)
    nbformat.validate(nb)
    nbformat.write(nb, args.notebook)

    stderr_cells = [i for i, cell in enumerate(nb.cells) if cell.cell_type == "code"
                    for out in cell.outputs if out.output_type == "stream" and out.name == "stderr"]
    print(f"saved {args.notebook} (nbformat {nb.nbformat}.{nb.nbformat_minor}); "
          f"cells with stderr output: {sorted(set(stderr_cells))}", flush=True)


if __name__ == "__main__":
    main()
