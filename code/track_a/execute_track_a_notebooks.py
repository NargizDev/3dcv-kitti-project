from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    ROOT / 'code' / 'track_a' / '01_kitti_setup.ipynb',
    ROOT / 'code' / 'track_a' / '02_yolov8_inference.ipynb',
    ROOT / 'code' / 'track_a' / '03_lift_to_3d.ipynb',
    ROOT / 'code' / 'track_a' / '04_metrics_3d.ipynb',
    ROOT / 'code' / 'track_a' / '05_visualizations.ipynb',
]


def execute_notebook(path):
    print(f'Executing {path.relative_to(ROOT)}')
    with path.open('r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
    )
    client.execute()

    with path.open('w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Finished {path.relative_to(ROOT)}')


def main():
    for path in NOTEBOOKS:
        execute_notebook(path)


if __name__ == '__main__':
    main()
