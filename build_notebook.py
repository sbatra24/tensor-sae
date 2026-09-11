"""Convert tensor_sae_colab.py (with `# %%` cell markers) into TensorSAE_reproduction.ipynb."""
import pathlib
import re
from typing import List

import nbformat

SOURCE = pathlib.Path(__file__).with_name("tensor_sae_colab.py")
TARGET = pathlib.Path(__file__).with_name("TensorSAE_reproduction.ipynb")
MARKER = re.compile(r"^# %%( \[markdown\])?\s*$")


def split_cells(text: str) -> List[nbformat.NotebookNode]:
    """Split the script at cell markers; markdown cells have their leading '# ' stripped."""
    cells: List[nbformat.NotebookNode] = []
    kind, lines = None, []

    def flush() -> None:
        body = "\n".join(lines).strip("\n")
        if kind is None or not body:
            return
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        match = MARKER.match(line)
        if match:
            flush()
            kind, lines = ("markdown" if match.group(1) else "code"), []
        else:
            lines.append(line)
    flush()
    return cells


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = split_cells(SOURCE.read_text())
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"name": "TensorSAE_reproduction.ipynb", "provenance": [], "gpuType": "T4"},
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, TARGET)
    print(f"wrote {TARGET} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
