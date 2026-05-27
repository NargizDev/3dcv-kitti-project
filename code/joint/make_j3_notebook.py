from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
JOINT_DIR = ROOT / 'code' / 'joint'


def main():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell(
            '# J3 - Final calibration experiments\n\n'
            'Compares `real_only`, `synth_only`, and `mixed_50_50` calibration strategies '
            'on the J2 KITTI validation split.'
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'joint'))\n\n"
            "from j3_final_experiments import run_j3_final_experiments\n"
        ),
        nbf.v4.new_code_cell(
            "result = run_j3_final_experiments()\n"
            "result['metrics']"
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
    out_path = JOINT_DIR / '02_j3_final_experiments.ipynb'
    nbf.write(nb, out_path)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
