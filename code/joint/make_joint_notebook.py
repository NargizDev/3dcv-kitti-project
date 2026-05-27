from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
JOINT_DIR = ROOT / 'code' / 'joint'


def main():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell(
            '# J1-J2 - Track A + Track B integration\n\n'
            'Проверяет совместимость артефактов и запускает первый полный pipeline: '
            'YOLO bbox + Track B Depth Anything + calibrated Lift-to-3D.'
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'joint'))\n\n"
            "from joint_pipeline import run_j1_j2\n"
        ),
        nbf.v4.new_code_cell(
            "result = run_j1_j2()\n"
            "result['J2']['metric_summary']"
        ),
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
    out_path = JOINT_DIR / '01_j1_j2_integration.ipynb'
    nbf.write(nb, out_path)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
