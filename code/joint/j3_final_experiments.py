import json
import os
import random
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / 'code' / 'shared'
JOINT_DIR = PROJECT_ROOT / 'code' / 'joint'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
if str(JOINT_DIR) not in sys.path:
    sys.path.insert(0, str(JOINT_DIR))

from lift_to_3d import lift_bbox_to_3d
from metrics import depth_error, euclidean_3d_error, match_predictions_to_gt, relative_depth_error, summarize_errors
from joint_pipeline import (
    AGGREGATION,
    ANCHOR,
    BBOX_SHRINK,
    IOU_THRESHOLD,
    TARGET_CLASSES,
    RESULTS_DIR,
    artifact_compatibility,
    calibrate_depth,
    collect_calibration_pairs,
    fit_scale_shift,
    make_kitti_loader,
    read_json,
    resize_depth_to_image,
    split_frame_ids,
    write_json,
    _depth_path,
    _load_depth,
    _load_yolo_detections,
)


def resolve_vkitti_root():
    env_root = os.environ.get('VKITTI_ROOT')
    candidates = [
        Path(env_root) if env_root else None,
        PROJECT_ROOT / 'data' / 'data' / 'vkitti2',
        PROJECT_ROOT / 'data' / 'vkitti2',
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    tried = ', '.join(str(c) for c in candidates if c)
    raise FileNotFoundError(f'VKITTI2 root not found. Tried: {tried}')


VKITTI_ROOT = resolve_vkitti_root()
VKITTI_PRED_DIR = PROJECT_ROOT / 'results' / 'track_b' / 'depth_pred_vkitti'
SYNC_ZIP_PATH = PROJECT_ROOT / f'3dcv-project_sync_{date.today().isoformat()}.zip'
RANDOM_SEED = 42
MAX_SYNTH_FRAMES = 250
SYNTH_PIXELS_PER_FRAME = 40
MIN_CLONE_SYNTH_PAIRS = 100


def _load_vkitti_depth_meters(path):
    depth_cm = np.asarray(Image.open(path))
    depth_m = depth_cm.astype(np.float32) / 100.0
    depth_m[depth_cm == 65535] = 0.0
    return depth_m


def _load_vkitti_prediction(scene, frame_id, target_shape):
    pred = np.load(VKITTI_PRED_DIR / scene / f'{frame_id}.npy').astype(np.float32)
    if pred.shape == tuple(target_shape):
        return pred, False
    resized = Image.fromarray(pred.astype(np.float32), mode='F').resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32), True


def _find_matching_vkitti_depth_files(prefer_clone=True):
    candidates = []
    for scene_dir in sorted(p for p in VKITTI_ROOT.iterdir() if p.is_dir()):
        scene = scene_dir.name
        pred_ids = {p.stem for p in (VKITTI_PRED_DIR / scene).glob('*.npy')}
        if not pred_ids:
            continue
        variation_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir())
        if prefer_clone:
            variation_dirs = sorted(variation_dirs, key=lambda p: 0 if p.name == 'clone' else 1)
        for variation_dir in variation_dirs:
            for depth_path in sorted((variation_dir / 'frames' / 'depth' / 'Camera_0').glob('depth_*.png')):
                frame_id = depth_path.stem.split('_')[-1]
                if frame_id in pred_ids:
                    candidates.append({
                        'scene': scene,
                        'variation': variation_dir.name,
                        'frame_id': frame_id,
                        'depth_path': depth_path,
                        'is_clone': variation_dir.name == 'clone',
                    })
    return candidates


def collect_synth_calibration_pairs():
    rng = random.Random(RANDOM_SEED)
    all_candidates = _find_matching_vkitti_depth_files(prefer_clone=True)
    clone_candidates = [item for item in all_candidates if item['is_clone']]
    if len(clone_candidates) >= MIN_CLONE_SYNTH_PAIRS:
        selected_source = 'vkitti2_clone'
        candidates = clone_candidates
    else:
        selected_source = 'vkitti2_available_variations_clone_fallback'
        candidates = all_candidates

    if not candidates:
        raise RuntimeError('No VKITTI2 GT depth files match Track B VKITTI predictions')

    rng.shuffle(candidates)
    candidates = candidates[:MAX_SYNTH_FRAMES]

    rows = []
    for item in candidates:
        gt_depth = _load_vkitti_depth_meters(item['depth_path'])
        pred_rel, resized = _load_vkitti_prediction(item['scene'], item['frame_id'], gt_depth.shape)
        mask = np.isfinite(gt_depth) & np.isfinite(pred_rel) & (gt_depth > 0.1) & (gt_depth < 80)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        sample_count = min(SYNTH_PIXELS_PER_FRAME, len(xs))
        indices = rng.sample(range(len(xs)), sample_count)
        for idx in indices:
            y = int(ys[idx])
            x = int(xs[idx])
            rows.append({
                'source': selected_source,
                'scene': item['scene'],
                'variation': item['variation'],
                'frame_id': item['frame_id'],
                'pixel_x': x,
                'pixel_y': y,
                'relative_depth': float(pred_rel[y, x]),
                'gt_depth': float(gt_depth[y, x]),
                'pred_was_resized': bool(resized),
            })

    df = pd.DataFrame(rows)
    if len(df) < 2:
        raise RuntimeError(f'Not enough synthetic calibration pairs: {len(df)}')
    return df, {
        'selected_source': selected_source,
        'candidate_count': len(candidates),
        'all_matching_depth_files': len(all_candidates),
        'clone_matching_depth_files': len(clone_candidates),
        'pixels_per_frame': SYNTH_PIXELS_PER_FRAME,
        'max_frames': MAX_SYNTH_FRAMES,
    }


def _strategy_calibrations(real_pairs, synth_pairs):
    n_mix = min(len(real_pairs), len(synth_pairs))
    mixed = pd.concat([
        real_pairs.sample(n=n_mix, random_state=RANDOM_SEED),
        synth_pairs.sample(n=n_mix, random_state=RANDOM_SEED),
    ], ignore_index=True)
    mixed['source'] = mixed.get('source', 'mixed_50_50')
    return {
        'real_only': real_pairs.copy(),
        'synth_only': synth_pairs.copy(),
        'mixed_50_50': mixed,
    }


def _evaluate_strategy(strategy_name, calibration, train_ids, val_ids):
    kitti = make_kitti_loader()
    per_match_rows = []
    eval_ids = list(val_ids)
    for frame_id in eval_ids:
        image = kitti.load_image(frame_id)
        relative_depth, depth_was_resized = resize_depth_to_image(_load_depth(frame_id), image.shape)
        metric_depth = calibrate_depth(relative_depth, calibration)
        detections = _load_yolo_detections(frame_id)
        gt_objects = [obj for obj in kitti.load_labels(frame_id) if obj.type in TARGET_CLASSES]
        intrinsics = kitti.get_intrinsics(frame_id)

        matches_by_class = []
        for target_class in TARGET_CLASSES:
            matches_by_class.extend(match_predictions_to_gt(detections, gt_objects, target_class=target_class, iou_threshold=IOU_THRESHOLD))

        lifted_cache = {}
        for det_idx, det in enumerate(detections):
            lifted_cache[det_idx] = lift_bbox_to_3d(
                det['bbox_2d'],
                metric_depth,
                intrinsics,
                anchor=ANCHOR,
                aggregation=AGGREGATION,
                bbox_shrink=BBOX_SHRINK,
                min_depth=0.1,
                max_depth=100.0,
                min_valid_pixels=10,
            )

        for match_idx, match in enumerate(matches_by_class):
            pred = match['prediction']
            pred_idx = next((idx for idx, det in enumerate(detections) if det is pred), None)
            if pred_idx is None or lifted_cache[pred_idx] is None:
                continue
            gt = match['gt']
            pred_loc = np.asarray(lifted_cache[pred_idx]['location_3d'], dtype=np.float32)
            per_match_rows.append({
                'experiment': strategy_name,
                'frame_id': frame_id,
                'split': 'val',
                'class': pred.get('class'),
                'match_index': match_idx,
                'det_index': pred_idx,
                'iou_2d': float(match['iou']),
                'confidence': float(pred.get('confidence', 1.0)),
                'euclidean_3d_error': euclidean_3d_error(pred_loc, gt.location),
                'depth_error': depth_error(pred_loc, gt.location),
                'relative_depth_error': relative_depth_error(pred_loc, gt.location),
                'pred_x': float(pred_loc[0]),
                'pred_y': float(pred_loc[1]),
                'pred_z': float(pred_loc[2]),
                'gt_x': float(gt.location[0]),
                'gt_y': float(gt.location[1]),
                'gt_z': float(gt.location[2]),
                'depth_was_resized': bool(depth_was_resized),
                'anchor': ANCHOR,
                'aggregation': AGGREGATION,
            })
    return per_match_rows


def _summary_rows(per_match_rows, calibrations):
    rows = []
    for experiment, calibration in calibrations.items():
        experiment_rows = [row for row in per_match_rows if row['experiment'] == experiment]
        for target_class in TARGET_CLASSES:
            class_rows = [row for row in experiment_rows if row['class'] == target_class]
            errors_3d = [row['euclidean_3d_error'] for row in class_rows]
            errors_depth = [row['depth_error'] for row in class_rows]
            summary = summarize_errors(errors_3d, errors_depth)
            rows.append({
                'experiment': experiment,
                'class': target_class,
                'matched_count': summary['count'],
                'mean_3d_error': summary['mean_3d_error'],
                'median_3d_error': summary['median_3d_error'],
                'std_3d_error': summary['std_3d_error'],
                'mean_depth_error': summary['mean_depth_error'],
                'median_depth_error': summary['median_depth_error'],
                'std_depth_error': summary['std_depth_error'],
                'localization_acc_2m': summary['localization_acc_2m'],
                'localization_acc_4m': summary['localization_acc_4m'],
                'scale': calibration['scale'],
                'shift': calibration['shift'],
                'calibration_pair_count': calibration['pair_count'],
                'calibration_train_rmse_m': calibration['train_rmse_m'],
                'calibration_train_mae_m': calibration['train_mae_m'],
                'calibration_correlation': calibration['correlation'],
            })
    return rows


def create_lena_brief(metrics_df, experiment_summary):
    car_rows = metrics_df[metrics_df['class'] == 'Car'].copy()
    car_rows = car_rows.sort_values('median_3d_error')
    best = car_rows.iloc[0].to_dict() if len(car_rows) else {}
    text = f"""# Handoff для Лены после J3

## Текущий статус

- Track A завершён.
- J1-J2 интеграция завершена.
- J3 финальные эксперименты завершены.
- Лучшая стратегия для Car по median 3D error: `{best.get('experiment', 'n/a')}`.

## Результаты, на которые ссылаться

- `results/joint/J3_final_metrics_table.csv`
- `results/joint/J3_experiment_summary.json`
- `results/joint/J4_table_joint_car_metrics.csv`
- `results/joint/figures/J4_joint_car_mean_median_errors.png`
- `results/track_b/B3_depth_metrics_vkitti.csv`
- `results/track_b/B5_absrel_by_scene.png`
- `results/track_b/B5_failure_cases.png`

## Распределение текста

**Наргиз**

- Метод: YOLOv8, Depth Anything depth, calibration, Lift-to-3D.
- Track A: KITTI loader, 2D detection, lift-to-3D, 3D localization metrics.
- J1-J4: интеграция, финальные таблицы/графики, заключение.

**Лена**

- Обзор литературы: synthetic data, sim-to-real/domain gap, monocular depth.
- Раздел данных: Virtual KITTI2 и Track B setup.
- Depth Anything inference setup и VKITTI2 depth metrics.
- Domain gap discussion на основе B5 figures и J3/J4 результатов.

**Совместно**

- Аннотация RU/EN.
- Описание experiment design.
- Финальная интерпретация: synthetic calibration помогла на текущем val split, но вывод ограничен размером выборки.

## Важные ограничения для обсуждения

- J2/J3 KITTI depth predictions ведут себя как inverse или relative depth, поэтому negative scale ожидаем.
- Локальный VKITTI2 `clone` GT depth неполный; J3 фиксирует synthetic source и использует доступные VKITTI2 GT depth files, совпадающие с Track B predictions.
- J3 оценивается на существующем J2 KITTI validation split: {experiment_summary['val_frame_count']} кадров.
"""
    path = RESULTS_DIR / 'J3_lena_brief.md'
    path.write_text(text, encoding='utf-8')
    return path


def create_writing_plan():
    text = """# План написания статьи

## Общий статус

- Техническая часть Track A, J1-J2 и J3 завершена.
- J4 должен дать финальные таблицы и графики для вставки в статью.
- Основной результат J3: на текущем KITTI val split лучшая стратегия по Car median 3D error — `synth_only`.

## Наргиз

- Введение: проблема 3D-аннотаций, цель работы, вклад проекта.
- Метод: YOLOv8, Depth Anything, affine depth calibration, Lift-to-3D через P2/KITTI intrinsics.
- Track A: KITTI loader, YOLOv8 inference, oracle-depth sanity-check, выбор `anchor='bottom'`, `aggregation='median'`.
- Joint J1-J3: compatibility check, integrated pipeline, real/synth/mixed calibration experiments.
- Результаты: финальные таблицы J3, интерпретация Car metrics, ограничения split.
- Заключение: что показал эксперимент и что улучшать дальше.

## Лена

- Обзор литературы: synthetic data, sim-to-real/domain gap, monocular depth estimation.
- Данные: Virtual KITTI2, вариации сцен, формат RGB/depth, отличие от KITTI.
- Track B: Depth Anything v2 setup, VKITTI2 depth metrics, B3/B5 результаты.
- Domain gap analysis: объяснить failure cases, различия синтетики и реальных KITTI кадров.
- Обсуждение ограничений synthetic calibration: неполный clone GT depth локально, fallback на доступные VKITTI2 variation files.

## Совместно

- Аннотация RU/EN и ключевые слова.
- Описание дизайна экспериментов `real_only`, `synth_only`, `mixed_50_50`.
- Финальная интерпретация: почему синтетическая калибровка могла улучшить median Car 3D error на малом val split.
- Вычитка, подписи таблиц/рисунков, список литературы.

## Артефакты для цитирования

- Track A: `results/track_a/A2_2d_metrics.csv`, `results/track_a/A3_lift_to_3d_summary.json`, `results/track_a/metrics_3d.csv`.
- Track B: `results/track_b/B3_depth_metrics_vkitti.csv`, `results/track_b/B5_absrel_by_scene.png`, `results/track_b/B5_failure_cases.png`.
- Joint: `results/joint/J3_final_metrics_table.csv`, `results/joint/J3_experiment_summary.json`, `results/joint/figures/`.

## Ключевые числа

- J3 KITTI val split: 11 кадров, 22 matched Car.
- Car median 3D error: `real_only` — 6.12 м, `synth_only` — 4.23 м, `mixed_50_50` — 5.40 м.
- Car Acc@4m: `real_only` — 0.273, `synth_only` — 0.500, `mixed_50_50` — 0.318.
- Depth Anything predictions ведут себя как inverse/relative depth: scale отрицательный во всех стратегиях.
"""
    path = PROJECT_ROOT / 'paper' / 'writing_plan.md'
    path.write_text(text, encoding='utf-8')
    return path


def create_sync_zip():
    include_dirs = [
        PROJECT_ROOT / 'code' / 'shared',
        PROJECT_ROOT / 'code' / 'track_a',
        PROJECT_ROOT / 'code' / 'joint',
        PROJECT_ROOT / 'results' / 'track_a',
        PROJECT_ROOT / 'results' / 'joint',
        PROJECT_ROOT / 'paper',
    ]
    include_files = [
        PROJECT_ROOT / 'README.md',
        PROJECT_ROOT / 'coordination.ipynb',
        PROJECT_ROOT / 'Joint_Integration_Tasks.md',
        PROJECT_ROOT / 'Track A - Наргиз.md',
        PROJECT_ROOT / 'Track B - Лена.md',
    ]
    excluded_parts = {'__pycache__', 'yolov8_predictions', 'J2_integrated_predictions'}
    excluded_suffixes = {'.npy', '.zip', '.pyc'}

    if SYNC_ZIP_PATH.exists():
        SYNC_ZIP_PATH.unlink()

    with zipfile.ZipFile(SYNC_ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in include_files:
            if file_path.exists():
                zf.write(file_path, file_path.relative_to(PROJECT_ROOT))
        for directory in include_dirs:
            if not directory.exists():
                continue
            for file_path in directory.rglob('*'):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(PROJECT_ROOT)
                if any(part in excluded_parts for part in rel.parts):
                    continue
                if file_path.suffix.lower() in excluded_suffixes:
                    continue
                zf.write(file_path, rel)
    return SYNC_ZIP_PATH


def run_j3_final_experiments():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    j1 = artifact_compatibility()
    common_ids = j1['common_frame_ids']
    train_ids, val_ids = split_frame_ids(common_ids, train_ratio=0.8)
    kitti = make_kitti_loader()

    real_pairs = collect_calibration_pairs(kitti, train_ids)
    real_pairs['source'] = 'kitti_object_depth'
    synth_pairs, synth_meta = collect_synth_calibration_pairs()

    strategies = _strategy_calibrations(real_pairs, synth_pairs)
    calibrations = {}
    for strategy, pairs in strategies.items():
        out_path = RESULTS_DIR / f'J3_calibration_{strategy}.csv'
        pairs.to_csv(out_path, index=False)
        calibrations[strategy] = fit_scale_shift(pairs)

    all_match_rows = []
    for strategy, calibration in calibrations.items():
        all_match_rows.extend(_evaluate_strategy(strategy, calibration, train_ids, val_ids))

    per_match_path = RESULTS_DIR / 'J3_metrics_3d_per_match.csv'
    pd.DataFrame(all_match_rows).to_csv(per_match_path, index=False)

    metric_rows = _summary_rows(all_match_rows, calibrations)
    metrics_path = RESULTS_DIR / 'J3_final_metrics_table.csv'
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)

    experiment_summary = {
        'status': 'ok',
        'common_frame_count': len(common_ids),
        'train_frame_count': len(train_ids),
        'val_frame_count': len(val_ids),
        'train_frame_ids': train_ids,
        'val_frame_ids': val_ids,
        'real_pair_count': int(len(real_pairs)),
        'synth_pair_count': int(len(synth_pairs)),
        'synth_meta': synth_meta,
        'calibrations': calibrations,
        'metrics_table': str(metrics_path),
        'per_match_metrics': str(per_match_path),
        'settings': {
            'anchor': ANCHOR,
            'aggregation': AGGREGATION,
            'bbox_shrink': BBOX_SHRINK,
            'iou_threshold': IOU_THRESHOLD,
            'random_seed': RANDOM_SEED,
        },
    }
    write_json(RESULTS_DIR / 'J3_experiment_summary.json', experiment_summary)

    lena_brief_path = create_lena_brief(metrics_df, experiment_summary)
    writing_plan_path = create_writing_plan()
    sync_zip_path = create_sync_zip()
    sync_manifest = {
        'status': 'ok',
        'zip_path': str(sync_zip_path),
        'zip_size_bytes': sync_zip_path.stat().st_size,
        'excluded': ['data/', '*.npy', '*.zip', '__pycache__/', 'results/track_a/yolov8_predictions/', 'results/joint/J2_integrated_predictions/'],
        'drive_gaps_observed': [
            'Drive results/joint is missing J1-J3 local outputs',
            'Drive code is missing code/joint',
            'Drive coordination.ipynb is older than local',
            'Drive Track A notebooks/results are older than local',
        ],
    }
    write_json(RESULTS_DIR / 'J3_drive_sync_manifest.json', sync_manifest)

    return {
        'metrics': metric_rows,
        'summary': experiment_summary,
        'lena_brief': str(lena_brief_path),
        'writing_plan': str(writing_plan_path),
        'sync_zip': sync_manifest,
    }


if __name__ == '__main__':
    result = run_j3_final_experiments()
    print(json.dumps({
        'status': result['summary']['status'],
        'metrics_table': result['summary']['metrics_table'],
        'lena_brief': result['lena_brief'],
        'sync_zip': result['sync_zip'],
    }, ensure_ascii=False, indent=2))
