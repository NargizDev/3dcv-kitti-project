from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'code' / 'joint' / '01_j1_j2_integration.ipynb'


def main():
    with NOTEBOOK.open('r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
    )
    client.execute()

    with NOTEBOOK.open('w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Finished {NOTEBOOK.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
