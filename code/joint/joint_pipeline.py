import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / 'code' / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from kitti_loader import KITTILoader
from lift_to_3d import extract_bbox_depth, lift_bbox_to_3d
from metrics import depth_error, euclidean_3d_error, match_predictions_to_gt, relative_depth_error, summarize_errors


TARGET_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
ANCHOR = 'bottom'
AGGREGATION = 'median'
BBOX_SHRINK = 0.2
IOU_THRESHOLD = 0.5

RESULTS_DIR = PROJECT_ROOT / 'results' / 'joint'
PRED_3D_DIR = RESULTS_DIR / 'J2_integrated_predictions'
YOLO_DIR = PROJECT_ROOT / 'results' / 'track_a' / 'yolov8_predictions'
DEPTH_DIR = PROJECT_ROOT / 'results' / 'track_b' / 'depth_pred_kitti'


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def find_kitti_root():
    candidates = [
        PROJECT_ROOT / 'data' / 'kitti' / 'training',
        PROJECT_ROOT / 'data' / 'kitti',
    ]
    for candidate in candidates:
        if (candidate / 'training' / 'image_2').is_dir():
            return candidate
    raise FileNotFoundError('Не найдена локальная структура KITTI training/image_2')


def make_kitti_loader():
    return KITTILoader(str(find_kitti_root()), split='training')


def _ids_from_files(path, suffix):
    path = Path(path)
    if not path.exists():
        return set()
    return {p.stem for p in path.glob(f'*{suffix}')}


def _yolo_path(frame_id):
    return YOLO_DIR / f'{frame_id}.json'


def _depth_path(frame_id):
    return DEPTH_DIR / f'{frame_id}.npy'


def _integrated_path(frame_id):
    return PRED_3D_DIR / f'{frame_id}.json'


def _load_yolo_detections(frame_id):
    path = _yolo_path(frame_id)
    if not path.exists():
        return []
    return read_json(path).get('detections', [])


def _load_depth(frame_id):
    return np.load(_depth_path(frame_id)).astype(np.float32)


def resize_depth_to_image(depth, image_shape):
    image_h, image_w = image_shape[:2]
    if depth.shape == (image_h, image_w):
        return depth, False
    resized = Image.fromarray(depth.astype(np.float32), mode='F').resize((image_w, image_h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32), True


def artifact_compatibility():
    kitti = make_kitti_loader()
    image_ids = _ids_from_files(kitti.image_dir, '.png')
    label_ids = _ids_from_files(kitti.label_dir, '.txt')
    calib_ids = _ids_from_files(kitti.calib_dir, '.txt')
    yolo_ids = _ids_from_files(YOLO_DIR, '.json')
    depth_ids = _ids_from_files(DEPTH_DIR, '.npy')

    common_ids = sorted(image_ids & label_ids & calib_ids & yolo_ids & depth_ids)
    shape_checks = []
    for frame_id in common_ids:
        image = kitti.load_image(frame_id)
        depth = _load_depth(frame_id)
        shape_checks.append({
            'frame_id': frame_id,
            'image_shape': list(image.shape[:2]),
            'depth_shape': list(depth.shape),
            'shape_matches': tuple(depth.shape) == tuple(image.shape[:2]),
            'will_resize': tuple(depth.shape) != tuple(image.shape[:2]),
        })

    report = {
        'status': 'ok' if common_ids else 'error',
        'counts': {
            'kitti_images': len(image_ids),
            'kitti_labels': len(label_ids),
            'kitti_calibs': len(calib_ids),
            'track_a_yolo_json': len(yolo_ids),
            'track_b_depth_npy': len(depth_ids),
            'common_required_frames': len(common_ids),
        },
        'common_frame_ids': common_ids,
        'missing_examples': {
            'depth_without_yolo': sorted((depth_ids & image_ids & label_ids & calib_ids) - yolo_ids)[:20],
            'yolo_without_depth': sorted((yolo_ids & image_ids & label_ids & calib_ids) - depth_ids)[:20],
            'depth_without_local_labels': sorted(depth_ids - label_ids)[:20],
        },
        'shape_checks': shape_checks,
        'shape_mismatch_count': sum(1 for item in shape_checks if not item['shape_matches']),
        'policy': {
            'shape_mismatch': 'resize_depth_to_image_with_bilinear_float32',
            'required_inputs': ['KITTI image', 'KITTI label', 'KITTI calib', 'Track A YOLO JSON', 'Track B KITTI depth npy'],
        },
    }
    write_json(RESULTS_DIR / 'J1_artifact_compatibility.json', report)
    if not common_ids:
        raise RuntimeError('J1 failed: no common labelled frames with YOLO and depth artifacts')
    return report


def split_frame_ids(frame_ids, train_ratio=0.8):
    frame_ids = sorted(frame_ids)
    split_idx = max(1, int(len(frame_ids) * train_ratio))
    if split_idx >= len(frame_ids) and len(frame_ids) > 1:
        split_idx = len(frame_ids) - 1
    return frame_ids[:split_idx], frame_ids[split_idx:]


def collect_calibration_pairs(kitti, frame_ids):
    rows = []
    for frame_id in frame_ids:
        image = kitti.load_image(frame_id)
        rel_depth, was_resized = resize_depth_to_image(_load_depth(frame_id), image.shape)
        for obj_idx, obj in enumerate(kitti.load_labels(frame_id)):
            if obj.type not in TARGET_CLASSES:
                continue
            rel_value = extract_bbox_depth(
                obj.bbox_2d,
                rel_depth,
                aggregation=AGGREGATION,
                bbox_shrink=BBOX_SHRINK,
                min_depth=-1e-6,
                max_depth=1.000001,
                min_valid_pixels=10,
            )
            if rel_value is None or not np.isfinite(rel_value):
                continue
            rows.append({
                'frame_id': frame_id,
                'object_index': obj_idx,
                'class': obj.type,
                'relative_depth': float(rel_value),
                'gt_depth': float(obj.depth),
                'depth_was_resized': bool(was_resized),
            })
    return pd.DataFrame(rows)


def fit_scale_shift(pairs_df):
    if len(pairs_df) < 2:
        raise RuntimeError(f'Need at least 2 calibration pairs, got {len(pairs_df)}')
    rel = pairs_df['relative_depth'].to_numpy(dtype=np.float32)
    gt = pairs_df['gt_depth'].to_numpy(dtype=np.float32)
    valid = np.isfinite(rel) & np.isfinite(gt)
    rel = rel[valid]
    gt = gt[valid]
    if rel.size < 2:
        raise RuntimeError(f'Need at least 2 finite calibration pairs, got {rel.size}')
    a = np.stack([rel, np.ones_like(rel)], axis=1)
    scale, shift = np.linalg.lstsq(a, gt, rcond=None)[0]
    pred = scale * rel + shift
    residual = pred - gt
    corr = float(np.corrcoef(rel, gt)[0, 1]) if rel.size > 1 and np.std(rel) > 0 and np.std(gt) > 0 else None
    return {
        'scale': float(scale),
        'shift': float(shift),
        'pair_count': int(rel.size),
        'relative_depth_min': float(rel.min()),
        'relative_depth_max': float(rel.max()),
        'gt_depth_min': float(gt.min()),
        'gt_depth_max': float(gt.max()),
        'train_rmse_m': float(np.sqrt(np.mean(residual ** 2))),
        'train_mae_m': float(np.mean(np.abs(residual))),
        'correlation': corr,
    }


def calibrate_depth(relative_depth, calibration):
    depth = calibration['scale'] * relative_depth.astype(np.float32) + calibration['shift']
    return depth.astype(np.float32)


def _summary_for_rows(rows, split_name, target_class):
    class_rows = [row for row in rows if row['split'] == split_name and row['class'] == target_class]
    errors_3d = [row['euclidean_3d_error'] for row in class_rows]
    errors_depth = [row['depth_error'] for row in class_rows]
    summary = summarize_errors(errors_3d, errors_depth)
    return {
        'split': split_name,
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
    }


def run_integrated_pipeline(frame_ids, train_frame_ids, val_frame_ids, calibration, smoke_limit=None):
    kitti = make_kitti_loader()
    PRED_3D_DIR.mkdir(parents=True, exist_ok=True)
    train_set = set(train_frame_ids)
    val_set = set(val_frame_ids)
    selected_frame_ids = list(frame_ids[:smoke_limit]) if smoke_limit else list(frame_ids)
    per_match_rows = []
    frame_summaries = []

    for frame_id in selected_frame_ids:
        image = kitti.load_image(frame_id)
        relative_depth, depth_was_resized = resize_depth_to_image(_load_depth(frame_id), image.shape)
        metric_depth = calibrate_depth(relative_depth, calibration)
        detections = _load_yolo_detections(frame_id)
        gt_objects = [obj for obj in kitti.load_labels(frame_id) if obj.type in TARGET_CLASSES]
        intrinsics = kitti.get_intrinsics(frame_id)
        split = 'train' if frame_id in train_set else 'val' if frame_id in val_set else 'unknown'

        lifted_detections = []
        matches_by_class = []
        for target_class in TARGET_CLASSES:
            matches_by_class.extend(match_predictions_to_gt(detections, gt_objects, target_class=target_class, iou_threshold=IOU_THRESHOLD))

        matched_prediction_ids = {id(match['prediction']) for match in matches_by_class}
        for det_idx, det in enumerate(detections):
            lifted = lift_bbox_to_3d(
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
            output_det = dict(det)
            output_det['det_index'] = det_idx
            output_det['location_3d'] = lifted['location_3d'] if lifted else None
            output_det['depth'] = lifted['depth'] if lifted else None
            output_det['pixel'] = lifted['pixel'] if lifted else None
            output_det['anchor'] = ANCHOR
            output_det['aggregation'] = AGGREGATION
            output_det['depth_source'] = 'track_b_depth_anything_calibrated'
            output_det['calibration'] = {
                'scale': calibration['scale'],
                'shift': calibration['shift'],
            }
            output_det['lift_status'] = 'ok' if lifted else 'no_valid_depth'
            output_det['is_2d_matched_to_gt'] = id(det) in matched_prediction_ids
            lifted_detections.append(output_det)

        for match_idx, match in enumerate(matches_by_class):
            pred = match['prediction']
            pred_idx = next((idx for idx, det in enumerate(detections) if det is pred), None)
            if pred_idx is None:
                continue
            lifted = lifted_detections[pred_idx]
            if lifted['location_3d'] is None:
                continue
            gt = match['gt']
            pred_loc = np.asarray(lifted['location_3d'], dtype=np.float32)
            err_3d = euclidean_3d_error(pred_loc, gt.location)
            err_depth = depth_error(pred_loc, gt.location)
            per_match_rows.append({
                'frame_id': frame_id,
                'split': split,
                'class': pred.get('class'),
                'match_index': match_idx,
                'det_index': pred_idx,
                'iou_2d': float(match['iou']),
                'confidence': float(pred.get('confidence', 1.0)),
                'euclidean_3d_error': err_3d,
                'depth_error': err_depth,
                'relative_depth_error': relative_depth_error(pred_loc, gt.location),
                'pred_x': float(pred_loc[0]),
                'pred_y': float(pred_loc[1]),
                'pred_z': float(pred_loc[2]),
                'gt_x': float(gt.location[0]),
                'gt_y': float(gt.location[1]),
                'gt_z': float(gt.location[2]),
                'depth_source': 'track_b_depth_anything_calibrated',
                'anchor': ANCHOR,
                'aggregation': AGGREGATION,
            })

        write_json(_integrated_path(frame_id), {
            'frame_id': frame_id,
            'split': split,
            'image_shape': list(image.shape[:2]),
            'relative_depth_shape': list(relative_depth.shape),
            'depth_was_resized': bool(depth_was_resized),
            'depth_source': 'track_b_depth_anything_calibrated',
            'calibration': {
                'scale': calibration['scale'],
                'shift': calibration['shift'],
                'pair_count': calibration['pair_count'],
            },
            'anchor': ANCHOR,
            'aggregation': AGGREGATION,
            'detections': lifted_detections,
        })
        frame_summaries.append({
            'frame_id': frame_id,
            'split': split,
            'detections': len(detections),
            'gt_objects': len(gt_objects),
            'matched_2d': len(matches_by_class),
            'matched_3d': sum(1 for row in per_match_rows if row['frame_id'] == frame_id),
            'depth_was_resized': bool(depth_was_resized),
        })

    return per_match_rows, frame_summaries


def run_j1_j2():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    j1 = artifact_compatibility()
    common_ids = j1['common_frame_ids']
    train_ids, val_ids = split_frame_ids(common_ids, train_ratio=0.8)
    kitti = make_kitti_loader()

    calibration_pairs = collect_calibration_pairs(kitti, train_ids)
    calibration_pairs_path = RESULTS_DIR / 'J2_calibration_pairs.csv'
    calibration_pairs.to_csv(calibration_pairs_path, index=False)
    calibration = fit_scale_shift(calibration_pairs)

    smoke_rows, smoke_frames = run_integrated_pipeline(common_ids, train_ids, val_ids, calibration, smoke_limit=5)
    smoke_path = RESULTS_DIR / 'J2_smoke_summary.json'
    write_json(smoke_path, {
        'status': 'ok',
        'processed_frames': len(smoke_frames),
        'matched_3d_rows': len(smoke_rows),
        'required_output_fields': ['frame_id', 'detections', 'location_3d', 'depth', 'depth_source', 'calibration'],
        'frames': smoke_frames,
    })
    if len(smoke_frames) != min(5, len(common_ids)):
        raise RuntimeError('J2 smoke did not process the expected number of frames')
    for frame in smoke_frames:
        data = read_json(_integrated_path(frame['frame_id']))
        if 'frame_id' not in data or 'detections' not in data or 'calibration' not in data:
            raise RuntimeError(f'Missing top-level fields in smoke output for {frame["frame_id"]}')
        if data['detections']:
            det = data['detections'][0]
            for key in ['location_3d', 'depth', 'depth_source', 'calibration']:
                if key not in det:
                    raise RuntimeError(f'Missing detection field {key} in smoke output for {frame["frame_id"]}')

    per_match_rows, frame_summaries = run_integrated_pipeline(common_ids, train_ids, val_ids, calibration)
    per_match_path = RESULTS_DIR / 'J2_metrics_3d_per_match.csv'
    pd.DataFrame(per_match_rows).to_csv(per_match_path, index=False)

    metric_rows = []
    for split in ['all', 'train', 'val']:
        source_rows = per_match_rows if split == 'all' else [row for row in per_match_rows if row['split'] == split]
        for target_class in TARGET_CLASSES:
            class_rows = source_rows if split == 'all' else per_match_rows
            if split == 'all':
                errors_3d = [row['euclidean_3d_error'] for row in source_rows if row['class'] == target_class]
                errors_depth = [row['depth_error'] for row in source_rows if row['class'] == target_class]
                summary = summarize_errors(errors_3d, errors_depth)
                metric_rows.append({
                    'split': split,
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
                })
            else:
                metric_rows.append(_summary_for_rows(class_rows, split, target_class))

    metrics_path = RESULTS_DIR / 'J2_metrics_3d.csv'
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    pipeline_summary = {
        'status': 'ok',
        'j1_report': str(RESULTS_DIR / 'J1_artifact_compatibility.json'),
        'common_frame_count': len(common_ids),
        'train_frame_count': len(train_ids),
        'val_frame_count': len(val_ids),
        'train_frame_ids': train_ids,
        'val_frame_ids': val_ids,
        'calibration': calibration,
        'calibration_pairs_csv': str(calibration_pairs_path),
        'smoke_summary': str(smoke_path),
        'integrated_predictions_dir': str(PRED_3D_DIR),
        'integrated_prediction_count': len(list(PRED_3D_DIR.glob('*.json'))),
        'per_match_metrics': str(per_match_path),
        'metrics_3d': str(metrics_path),
        'frame_summaries': frame_summaries,
        'metric_summary': metric_rows,
        'settings': {
            'anchor': ANCHOR,
            'aggregation': AGGREGATION,
            'bbox_shrink': BBOX_SHRINK,
            'iou_threshold': IOU_THRESHOLD,
            'depth_source': 'track_b_depth_anything_calibrated',
        },
    }
    write_json(RESULTS_DIR / 'J2_pipeline_summary.json', pipeline_summary)
    return {
        'J1': j1,
        'J2': pipeline_summary,
    }


if __name__ == '__main__':
    result = run_j1_j2()
    print(json.dumps({
        'status': result['J2']['status'],
        'common_frame_count': result['J2']['common_frame_count'],
        'train_frame_count': result['J2']['train_frame_count'],
        'val_frame_count': result['J2']['val_frame_count'],
        'calibration': result['J2']['calibration'],
        'metrics_3d': result['J2']['metrics_3d'],
    }, ensure_ascii=False, indent=2))
