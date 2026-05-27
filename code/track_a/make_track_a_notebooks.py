from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
TRACK_A_DIR = ROOT / 'code' / 'track_a'


COMMON_SETUP = """\
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'track_a'))

from track_a_pipeline import (
    run_a1,
    run_a2,
    run_a3,
    run_a4,
    run_a5,
)
"""


NOTEBOOKS = {
    '01_kitti_setup.ipynb': {
        'title': 'A1 - KITTI setup and loader validation',
        'intro': (
            'Локальная проверка структуры KITTI, `KITTILoader`, intrinsics, '
            'парсинга label-файлов и сохранение `A1_kitti_samples.png`.'
        ),
        'call': 'a1_summary = run_a1()\na1_summary',
    },
    '02_yolov8_inference.ipynb': {
        'title': 'A2 - YOLOv8 inference on KITTI',
        'intro': (
            'Запуск pre-trained YOLOv8 на локальных KITTI image_2, сохранение '
            'JSON-предсказаний и 2D-метрик P/R/F1/AP50. По умолчанию используется '
            '`yolov8n.pt`, потому что локальная машина работает на CPU. Для Colab '
            'можно задать `TRACK_A_YOLO_MODEL=yolov8m.pt`.'
        ),
        'call': 'a2_summary = run_a2()\na2_summary',
    },
    '03_lift_to_3d.ipynb': {
        'title': 'A3 - Lift-to-3D validation',
        'intro': (
            'Проверка back-projection на искусственной карте глубины, где внутри '
            'GT bbox подставляется GT Z из KITTI.'
        ),
        'call': 'a3_summary = run_a3()\na3_summary',
    },
    '04_metrics_3d.ipynb': {
        'title': 'A4 - 3D localization metrics',
        'intro': (
            'Unit-тесты метрик и Track A sanity-check 3D-локализации с oracle '
            'depth из matched GT объектов.'
        ),
        'call': 'a4_summary = run_a4()\na4_summary',
    },
    '05_visualizations.ipynb': {
        'title': 'A5 - Visualization demo',
        'intro': (
            'Демо функций визуализации: 2D bbox, oracle-depth overlay и BEV '
            'для Lift-to-3D sanity-check.'
        ),
        'call': 'a5_summary = run_a5()\na5_summary',
    },
}


def build_notebook(title, intro, call):
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell(f'# {title}\n\n{intro}'),
        nbf.v4.new_code_cell(COMMON_SETUP),
        nbf.v4.new_code_cell(call),
    ]
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.9',
        },
    }
    return nb


def main():
    for filename, spec in NOTEBOOKS.items():
        nb = build_notebook(spec['title'], spec['intro'], spec['call'])
        path = TRACK_A_DIR / filename
        nbf.write(nb, path)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
