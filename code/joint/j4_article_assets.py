import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / 'code' / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

RESULTS_DIR = PROJECT_ROOT / 'results' / 'joint'
FIGURES_DIR = RESULTS_DIR / 'figures'


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding='utf-8')


def _round_numeric(df, digits=3):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def build_track_a_tables():
    a2 = pd.read_csv(PROJECT_ROOT / 'results' / 'track_a' / 'A2_2d_metrics.csv')
    a4 = pd.read_csv(PROJECT_ROOT / 'results' / 'track_a' / 'metrics_3d.csv')
    a3_summary = json.loads((PROJECT_ROOT / 'results' / 'track_a' / 'A3_lift_to_3d_summary.json').read_text(encoding='utf-8'))

    detection = a2[['class', 'precision', 'recall', 'f1', 'ap50', 'gt_count', 'prediction_count']].copy()
    oracle = a4[['class', 'matched_count', 'mean_3d_error', 'median_3d_error', 'localization_acc_2m', 'localization_acc_4m']].copy()
    table = detection.merge(oracle, on='class', how='left')
    out_path = RESULTS_DIR / 'J4_table_track_a_summary.csv'
    _round_numeric(table).to_csv(out_path, index=False)
    return {
        'path': str(out_path),
        'anchor_decision': a3_summary.get('decision', {}),
        'rows': len(table),
    }


def build_track_b_tables():
    b3 = pd.read_csv(PROJECT_ROOT / 'results' / 'track_b' / 'B3_depth_metrics_vkitti.csv')
    metric_cols = ['AbsRel', 'SqRel', 'RMSE', 'δ < 1.25', 'δ < 1.25²', 'δ < 1.25³']
    scene_table = b3.groupby('scene', as_index=False)[metric_cols].mean()
    total_row = {'scene': 'ALL'}
    total_row.update({col: b3[col].mean() for col in metric_cols})
    scene_table = pd.concat([scene_table, pd.DataFrame([total_row])], ignore_index=True)
    out_path = RESULTS_DIR / 'J4_table_track_b_depth_by_scene.csv'
    _round_numeric(scene_table).to_csv(out_path, index=False)
    return {
        'path': str(out_path),
        'rows': len(scene_table),
        'frame_count': int(len(b3)),
    }


def build_joint_table():
    j3 = pd.read_csv(RESULTS_DIR / 'J3_final_metrics_table.csv')
    car = j3[j3['class'] == 'Car'].copy()
    keep = [
        'experiment',
        'matched_count',
        'mean_3d_error',
        'median_3d_error',
        'mean_depth_error',
        'localization_acc_2m',
        'localization_acc_4m',
        'scale',
        'shift',
        'calibration_pair_count',
    ]
    car = car[keep].sort_values('median_3d_error')
    out_path = RESULTS_DIR / 'J4_table_joint_car_metrics.csv'
    _round_numeric(car).to_csv(out_path, index=False)
    return {
        'path': str(out_path),
        'best_strategy': str(car.iloc[0]['experiment']) if len(car) else None,
        'rows': len(car),
    }


def plot_joint_car_errors():
    car = pd.read_csv(RESULTS_DIR / 'J4_table_joint_car_metrics.csv')
    order = ['real_only', 'synth_only', 'mixed_50_50']
    car['experiment'] = pd.Categorical(car['experiment'], categories=order, ordered=True)
    car = car.sort_values('experiment')

    x = np.arange(len(car))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - width / 2, car['mean_3d_error'], width, label='Mean 3D error', color='#3B82F6')
    ax.bar(x + width / 2, car['median_3d_error'], width, label='Median 3D error', color='#F97316')
    ax.set_xticks(x)
    ax.set_xticklabels(car['experiment'])
    ax.set_ylabel('Ошибка, м')
    ax.set_title('J3: сравнение стратегий калибровки для класса Car')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    for idx, row in enumerate(car.itertuples(index=False)):
        ax.text(idx + width / 2, row.median_3d_error + 0.15, f'{row.median_3d_error:.2f}', ha='center', fontsize=9)
    fig.tight_layout()
    out_path = FIGURES_DIR / 'J4_joint_car_mean_median_errors.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def plot_j3_error_boxplot():
    per_match = pd.read_csv(RESULTS_DIR / 'J3_metrics_3d_per_match.csv')
    car = per_match[per_match['class'] == 'Car'].copy()
    order = ['real_only', 'synth_only', 'mixed_50_50']
    data = [car[car['experiment'] == exp]['euclidean_3d_error'].dropna().values for exp in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.boxplot(data, tick_labels=order, showfliers=True, patch_artist=True)
    ax.set_ylabel('3D error, м')
    ax.set_title('J3: распределение 3D-ошибок для matched Car')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    out_path = FIGURES_DIR / 'J4_j3_car_error_boxplot.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def plot_qualitative_overview():
    inputs = [
        ('Track A: 2D bbox + oracle-depth sanity check', PROJECT_ROOT / 'results' / 'track_a' / 'A5_visualization_demo.png'),
        ('Track B: Depth Anything failure cases', PROJECT_ROOT / 'results' / 'track_b' / 'B5_failure_cases.png'),
        ('J3: final calibration comparison', FIGURES_DIR / 'J4_joint_car_mean_median_errors.png'),
    ]
    fig, axes = plt.subplots(len(inputs), 1, figsize=(10, 12))
    for ax, (title, path) in zip(axes, inputs):
        ax.set_title(title, loc='left', fontsize=11)
        if path.exists():
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f'Не найден файл: {path.name}', ha='center', va='center')
        ax.axis('off')
    fig.tight_layout()
    out_path = FIGURES_DIR / 'J4_qualitative_overview.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def run_j4_article_assets():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    track_a = build_track_a_tables()
    track_b = build_track_b_tables()
    joint = build_joint_table()
    figures = [
        plot_joint_car_errors(),
        plot_j3_error_boxplot(),
        plot_qualitative_overview(),
    ]
    summary = {
        'status': 'ok',
        'tables': {
            'track_a': track_a,
            'track_b': track_b,
            'joint_car': joint,
        },
        'figures': figures,
        'notes': [
            'J4 uses existing Track A, Track B, and J3 result files; it does not rerun inference.',
            'Joint table is sorted by Car median 3D error.',
        ],
    }
    write_json(RESULTS_DIR / 'J4_article_assets_summary.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(run_j4_article_assets(), ensure_ascii=False, indent=2))
