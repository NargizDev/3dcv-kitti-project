import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / 'code' / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from kitti_loader import KITTILoader
from lift_to_3d import bbox_anchor_point, lift_2d_to_3d, lift_bbox_to_3d
from metrics import (
    compute_iou_2d,
    depth_error,
    euclidean_3d_error,
    evaluate_matched_locations,
    localization_accuracy,
    match_predictions_to_gt,
    relative_depth_error,
    summarize_errors,
)
from visualization import draw_2d_bboxes, draw_birds_eye_view, draw_depth_overlay


RESULTS_DIR = PROJECT_ROOT / 'results' / 'track_a'
PRED_DIR = RESULTS_DIR / 'yolov8_predictions'
LIFT_DIR = RESULTS_DIR / 'lift3d_predictions'
TARGET_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
COCO_TO_KITTI = {
    2: 'Car',
    0: 'Pedestrian',
    1: 'Cyclist',
    7: 'Truck',
}


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


def make_kitti_loader(only_valid=False):
    kitti = KITTILoader(str(find_kitti_root()), split='training')
    if only_valid:
        kitti.frame_ids = valid_frame_ids(kitti)
    return kitti


def valid_frame_ids(kitti=None):
    kitti = kitti or make_kitti_loader(only_valid=False)
    image_ids = {Path(p).stem for p in os.listdir(kitti.image_dir) if p.endswith('.png')}
    label_ids = {Path(p).stem for p in os.listdir(kitti.label_dir) if p.endswith('.txt')} if os.path.isdir(kitti.label_dir) else set()
    calib_ids = {Path(p).stem for p in os.listdir(kitti.calib_dir) if p.endswith('.txt')}
    return sorted(image_ids & label_ids & calib_ids)


def _class_objects(objects, target_class):
    return [obj for obj in objects if obj.type == target_class]


def _frame_image_path(kitti, frame_id):
    return Path(kitti.image_dir) / f'{frame_id}.png'


def _prediction_path(frame_id):
    return PRED_DIR / f'{frame_id}.json'


def _lift_path(frame_id):
    return LIFT_DIR / f'{frame_id}.json'


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def run_a1(seed=42):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kitti_all = make_kitti_loader(only_valid=False)
    valid_ids = valid_frame_ids(kitti_all)
    kitti = make_kitti_loader(only_valid=True)

    sample_id = valid_ids[0] if valid_ids else kitti_all.frame_ids[0]
    image = kitti_all.load_image(sample_id)
    objects = kitti_all.load_labels(sample_id)
    calib = kitti_all.load_calib(sample_id)
    intrinsics = kitti_all.get_intrinsics(sample_id)

    summary = {
        'status': 'ok',
        'kitti_root': str(find_kitti_root()),
        'image_count': len(kitti_all.frame_ids),
        'valid_image_label_calib_count': len(valid_ids),
        'sample_frame_id': sample_id,
        'sample_image_shape': list(image.shape),
        'sample_object_count': len(objects),
        'sample_p2_shape': list(calib['P2'].shape),
        'sample_intrinsics': {
            'fx': float(intrinsics[0]),
            'fy': float(intrinsics[1]),
            'cx': float(intrinsics[2]),
            'cy': float(intrinsics[3]),
        },
        'classes_in_sample': sorted({obj.type for obj in objects}),
    }
    write_json(RESULTS_DIR / 'A1_dataset_summary.json', summary)

    rng = random.Random(seed)
    sample_ids = rng.sample(valid_ids, min(4, len(valid_ids)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, frame_id in zip(axes.reshape(-1), sample_ids):
        img = kitti.load_image(frame_id)
        objs = [obj for obj in kitti.load_labels(frame_id) if obj.type in TARGET_CLASSES]
        draw_2d_bboxes(ax, img, detections=[], gt_objects=objs[:20], title=f'KITTI {frame_id}: GT boxes')
    for ax in axes.reshape(-1)[len(sample_ids):]:
        ax.axis('off')
    fig.tight_layout()
    out_path = RESULTS_DIR / 'A1_kitti_samples.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    summary['sample_visualization'] = str(out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _prediction_needs_refresh(path, model_name):
    if not path.exists():
        return True
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return True
    return data.get('model') != model_name


def run_yolo_inference(model_name=None, conf=None, max_frames=None, overwrite=False):
    from ultralytics import YOLO

    model_name = model_name or os.environ.get('TRACK_A_YOLO_MODEL', 'yolov8n.pt')
    conf = float(conf if conf is not None else os.environ.get('TRACK_A_YOLO_CONF', '0.25'))
    env_max = os.environ.get('TRACK_A_MAX_YOLO_FRAMES')
    if max_frames is None and env_max:
        max_frames = int(env_max)

    kitti = make_kitti_loader(only_valid=False)
    frame_ids = list(kitti.frame_ids)
    if max_frames:
        frame_ids = frame_ids[:int(max_frames)]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)

    written = 0
    skipped = 0
    for idx, frame_id in enumerate(frame_ids, 1):
        save_path = _prediction_path(frame_id)
        if not overwrite and not _prediction_needs_refresh(save_path, model_name):
            skipped += 1
            continue

        result = model(str(_frame_image_path(kitti, frame_id)), conf=conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            coco_cls = int(box.cls[0])
            if coco_cls not in COCO_TO_KITTI:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            detections.append({
                'class': COCO_TO_KITTI[coco_cls],
                'bbox_2d': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(box.conf[0].detach().cpu().item()),
                'coco_class_id': coco_cls,
            })

        write_json(save_path, {
            'frame_id': frame_id,
            'model': model_name,
            'confidence_threshold': conf,
            'source': 'local_kitti_training_image_2',
            'detections': detections,
        })
        written += 1

        if idx == 1 or idx % 100 == 0 or idx == len(frame_ids):
            print(f'YOLO progress: {idx}/{len(frame_ids)} frames, written={written}, skipped={skipped}')

    manifest = {
        'status': 'ok',
        'model': model_name,
        'confidence_threshold': conf,
        'requested_frames': len(frame_ids),
        'written': written,
        'skipped': skipped,
        'prediction_dir': str(PRED_DIR),
    }
    write_json(RESULTS_DIR / 'A2_yolo_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _load_detections(frame_id):
    path = _prediction_path(frame_id)
    if not path.exists():
        return []
    return read_json(path).get('detections', [])


def evaluate_2d_class(kitti, frame_ids, target_class, iou_threshold=0.5):
    gt_by_frame = {}
    total_gt = 0
    predictions = []

    for frame_id in frame_ids:
        gt = _class_objects(kitti.load_labels(frame_id), target_class)
        gt_by_frame[frame_id] = gt
        total_gt += len(gt)
        for det in _load_detections(frame_id):
            if det.get('class') == target_class:
                predictions.append({
                    'frame_id': frame_id,
                    'bbox_2d': det['bbox_2d'],
                    'confidence': float(det.get('confidence', 1.0)),
                })

    predictions.sort(key=lambda item: -item['confidence'])
    used_gt = {frame_id: set() for frame_id in frame_ids}
    tp_values = []
    fp_values = []

    tp = 0
    fp = 0
    for pred in predictions:
        frame_id = pred['frame_id']
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gt_by_frame[frame_id]):
            if idx in used_gt[frame_id]:
                continue
            iou = compute_iou_2d(pred['bbox_2d'], gt.bbox_2d)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            used_gt[frame_id].add(best_idx)
            tp_values.append(1)
            fp_values.append(0)
        else:
            fp += 1
            tp_values.append(0)
            fp_values.append(1)

    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap50 = _average_precision(tp_values, fp_values, total_gt)

    return {
        'class': target_class,
        'iou_threshold': iou_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap50': ap50,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt_count': total_gt,
        'prediction_count': len(predictions),
    }


def _average_precision(tp_values, fp_values, total_gt):
    if total_gt == 0 or not tp_values:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp_values, dtype=np.float32))
    fp_cum = np.cumsum(np.asarray(fp_values, dtype=np.float32))
    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def run_a2(model_name=None, conf=None, max_frames=None, overwrite=False):
    manifest = run_yolo_inference(model_name=model_name, conf=conf, max_frames=max_frames, overwrite=overwrite)

    kitti = make_kitti_loader(only_valid=True)
    frame_ids = list(kitti.frame_ids)
    rows = [evaluate_2d_class(kitti, frame_ids, cls) for cls in TARGET_CLASSES]
    metrics_df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / 'A2_2d_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)

    sample_frame = next((fid for fid in frame_ids if _load_detections(fid)), frame_ids[0])
    image = kitti.load_image(sample_frame)
    detections = _load_detections(sample_frame)
    gt_objects = [obj for obj in kitti.load_labels(sample_frame) if obj.type in TARGET_CLASSES]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    draw_2d_bboxes(ax, image, detections=detections, gt_objects=gt_objects, title=f'YOLO detections vs GT: {sample_frame}')
    sample_path = RESULTS_DIR / 'A2_yolo_sample.png'
    fig.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = {
        **manifest,
        'evaluated_frames': len(frame_ids),
        'metrics_path': str(metrics_path),
        'sample_visualization': str(sample_path),
        'metrics': rows,
    }
    write_json(RESULTS_DIR / 'A2_2d_metrics_summary.json', summary)
    print(metrics_df.to_string(index=False))
    return summary


def _fake_depth_from_bbox(image_shape, bbox_2d, depth):
    fake_depth = np.zeros(image_shape[:2], dtype=np.float32)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_2d]
    h, w = fake_depth.shape
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 > x1 and y2 > y1:
        fake_depth[y1:y2, x1:x2] = float(depth)
    return fake_depth


def run_a3(max_frames=50):
    kitti = make_kitti_loader(only_valid=True)
    rows = []
    frame_count = 0
    for frame_id in kitti.frame_ids:
        objects = [obj for obj in kitti.load_labels(frame_id) if obj.type == 'Car']
        if not objects:
            continue
        frame_count += 1
        image = kitti.load_image(frame_id)
        intrinsics = kitti.get_intrinsics(frame_id)
        for object_index, obj in enumerate(objects):
            for anchor in ['center', 'bottom']:
                fake_depth = _fake_depth_from_bbox(image.shape, obj.bbox_2d, obj.depth)
                lifted = lift_bbox_to_3d(
                    obj.bbox_2d,
                    fake_depth,
                    intrinsics,
                    anchor=anchor,
                    aggregation='median',
                    bbox_shrink=0.0,
                    min_valid_pixels=1,
                )
                if lifted is None:
                    continue
                pred = lifted['location_3d']
                gt = obj.location
                rows.append({
                    'frame_id': frame_id,
                    'object_index': object_index,
                    'class': obj.type,
                    'anchor': anchor,
                    'pred_x': float(pred[0]),
                    'pred_y': float(pred[1]),
                    'pred_z': float(pred[2]),
                    'gt_x': float(gt[0]),
                    'gt_y': float(gt[1]),
                    'gt_z': float(gt[2]),
                    'abs_x_error': float(abs(pred[0] - gt[0])),
                    'abs_y_error': float(abs(pred[1] - gt[1])),
                    'abs_z_error': float(abs(pred[2] - gt[2])),
                    'euclidean_error': float(np.linalg.norm(pred - gt)),
                })
        if frame_count >= max_frames:
            break

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / 'A3_lift_to_3d_checks.csv'
    df.to_csv(csv_path, index=False)

    summary_rows = []
    for anchor, group in df.groupby('anchor'):
        summary_rows.append({
            'anchor': anchor,
            'count': int(len(group)),
            'mean_abs_x_error': _safe_float(group['abs_x_error'].mean()),
            'median_abs_x_error': _safe_float(group['abs_x_error'].median()),
            'mean_abs_y_error': _safe_float(group['abs_y_error'].mean()),
            'median_abs_y_error': _safe_float(group['abs_y_error'].median()),
            'mean_abs_z_error': _safe_float(group['abs_z_error'].mean()),
            'median_abs_z_error': _safe_float(group['abs_z_error'].median()),
            'mean_euclidean_error': _safe_float(group['euclidean_error'].mean()),
            'median_euclidean_error': _safe_float(group['euclidean_error'].median()),
        })
    summary = {
        'status': 'ok',
        'checked_frames_with_cars': frame_count,
        'checks_path': str(csv_path),
        'summary_by_anchor': summary_rows,
        'decision': {
            'depth_aggregation': 'median',
            'integration_anchor': 'bottom',
            'reason': 'KITTI location describes the object location near the bottom on the road plane; bottom anchor better matches Y than bbox center.',
        },
    }
    write_json(RESULTS_DIR / 'A3_lift_to_3d_summary.json', summary)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return summary


def run_a4():
    _run_metric_unit_tests()
    per_match_rows, summary_rows = _run_oracle_depth_3d_metrics()

    unit_path = RESULTS_DIR / 'A4_metrics_smoke_test.json'
    write_json(unit_path, {
        'status': 'ok',
        'tests': [
            'compute_iou_2d',
            'euclidean_3d_error',
            'depth_error',
            'relative_depth_error',
            'localization_accuracy',
            'match_predictions_to_gt',
            'evaluate_matched_locations',
            'summarize_errors_empty_input',
        ],
    })

    per_match_path = RESULTS_DIR / 'A4_metrics_3d_per_match.csv'
    summary_path = RESULTS_DIR / 'metrics_3d.csv'
    pd.DataFrame(per_match_rows).to_csv(per_match_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    summary = {
        'status': 'ok',
        'unit_test_report': str(unit_path),
        'per_match_metrics': str(per_match_path),
        'summary_metrics': str(summary_path),
        'depth_source': 'matched_gt_oracle_depth_for_track_a_validation',
        'summary': summary_rows,
    }
    write_json(RESULTS_DIR / 'A4_metrics_3d_summary.json', summary)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return summary


def _run_metric_unit_tests():
    assert compute_iou_2d([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert compute_iou_2d([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert round(compute_iou_2d([0, 0, 10, 10], [5, 5, 15, 15]), 6) == round(25 / 175, 6)

    assert euclidean_3d_error([0, 0, 10], [0, 0, 10]) == 0.0
    assert depth_error([0, 0, 12], [0, 0, 10]) == 2.0
    assert relative_depth_error([0, 0, 12], [0, 0, 10]) == 0.2
    assert abs(localization_accuracy([1.0, 2.5, 4.5], 4.0) - (2 / 3)) < 1e-9

    predictions = [
        {'class': 'Car', 'bbox_2d': [0, 0, 10, 10], 'location_3d': [0, 0, 10], 'confidence': 0.9},
        {'class': 'Car', 'bbox_2d': [50, 50, 60, 60], 'location_3d': [0, 0, 20], 'confidence': 0.8},
        {'class': 'Pedestrian', 'bbox_2d': [0, 0, 10, 10], 'location_3d': [0, 0, 10], 'confidence': 0.7},
    ]
    gt_objects = [
        {'class': 'Car', 'bbox_2d': [1, 1, 11, 11], 'location_3d': [0, 0, 11]},
        {'class': 'Car', 'bbox_2d': [100, 100, 110, 110], 'location_3d': [0, 0, 30]},
    ]

    matches = match_predictions_to_gt(predictions, gt_objects, target_class='Car', iou_threshold=0.5)
    assert len(matches) == 1

    rows, summary = evaluate_matched_locations(matches)
    assert summary['count'] == 1
    assert rows[0]['depth_error'] == 1.0

    empty_summary = summarize_errors([], [])
    assert empty_summary['count'] == 0
    assert empty_summary['mean_3d_error'] is None
    assert empty_summary['localization_acc_2m'] == 0.0


def _run_oracle_depth_3d_metrics():
    kitti = make_kitti_loader(only_valid=True)
    LIFT_DIR.mkdir(parents=True, exist_ok=True)
    per_match_rows = []
    by_class = {cls: {'errors_3d': [], 'errors_depth': [], 'matches': 0} for cls in TARGET_CLASSES}

    for frame_id in kitti.frame_ids:
        detections = _load_detections(frame_id)
        if not detections:
            continue
        gt_objects = [obj for obj in kitti.load_labels(frame_id) if obj.type in TARGET_CLASSES]
        if not gt_objects:
            continue
        intrinsics = kitti.get_intrinsics(frame_id)
        lifted_detections = []

        for target_class in TARGET_CLASSES:
            matches = match_predictions_to_gt(detections, gt_objects, target_class=target_class, iou_threshold=0.5)
            for match_idx, match in enumerate(matches):
                pred = dict(match['prediction'])
                gt = match['gt']
                u, v = bbox_anchor_point(pred['bbox_2d'], anchor='bottom')
                pred_loc = lift_2d_to_3d(u, v, gt.depth, *intrinsics)
                pred['location_3d'] = pred_loc
                pred['depth'] = float(gt.depth)
                pred['depth_source'] = 'matched_gt_oracle'
                pred['anchor'] = 'bottom'
                pred['matched_iou_2d'] = float(match['iou'])
                pred['matched_gt_location'] = gt.location
                pred['matched_gt_bbox_2d'] = gt.bbox_2d
                lifted_detections.append(pred)

                err_3d = euclidean_3d_error(pred_loc, gt.location)
                err_depth = depth_error(pred_loc, gt.location)
                rel_depth = relative_depth_error(pred_loc, gt.location)
                by_class[target_class]['errors_3d'].append(err_3d)
                by_class[target_class]['errors_depth'].append(err_depth)
                by_class[target_class]['matches'] += 1
                per_match_rows.append({
                    'frame_id': frame_id,
                    'class': target_class,
                    'match_index': match_idx,
                    'iou_2d': float(match['iou']),
                    'confidence': float(pred.get('confidence', 1.0)),
                    'euclidean_3d_error': err_3d,
                    'depth_error': err_depth,
                    'relative_depth_error': rel_depth,
                    'pred_x': float(pred_loc[0]),
                    'pred_y': float(pred_loc[1]),
                    'pred_z': float(pred_loc[2]),
                    'gt_x': float(gt.location[0]),
                    'gt_y': float(gt.location[1]),
                    'gt_z': float(gt.location[2]),
                    'depth_source': 'matched_gt_oracle',
                    'anchor': 'bottom',
                })

        if lifted_detections:
            write_json(_lift_path(frame_id), {
                'frame_id': frame_id,
                'depth_source': 'matched_gt_oracle',
                'anchor': 'bottom',
                'detections': lifted_detections,
            })

    summary_rows = []
    for target_class, values in by_class.items():
        errors_3d = np.asarray(values['errors_3d'], dtype=np.float32)
        errors_depth = np.asarray(values['errors_depth'], dtype=np.float32)
        summary = summarize_errors(errors_3d, errors_depth)
        summary_rows.append({
            'class': target_class,
            'depth_source': 'matched_gt_oracle',
            'anchor': 'bottom',
            'matched_count': int(values['matches']),
            'mean_3d_error': summary['mean_3d_error'],
            'median_3d_error': summary['median_3d_error'],
            'std_3d_error': summary['std_3d_error'],
            'mean_depth_error': summary['mean_depth_error'],
            'median_depth_error': summary['median_depth_error'],
            'localization_acc_2m': summary['localization_acc_2m'],
            'localization_acc_4m': summary['localization_acc_4m'],
        })
    return per_match_rows, summary_rows


def run_a5():
    kitti = make_kitti_loader(only_valid=True)
    per_match_path = RESULTS_DIR / 'A4_metrics_3d_per_match.csv'
    if not per_match_path.exists():
        run_a4()
    per_match = pd.read_csv(per_match_path)
    if per_match.empty:
        raise RuntimeError('Нет matched 3D строк для визуализации A5')

    frame_id = str(per_match.groupby('frame_id').size().sort_values(ascending=False).index[0]).zfill(6)
    image = kitti.load_image(frame_id)
    gt_objects = [obj for obj in kitti.load_labels(frame_id) if obj.type in TARGET_CLASSES]
    detections = _load_detections(frame_id)
    rows = per_match[per_match['frame_id'].astype(str).str.zfill(6) == frame_id]
    pred_locs = rows[['pred_x', 'pred_y', 'pred_z']].to_numpy(dtype=np.float32)
    gt_locs = rows[['gt_x', 'gt_y', 'gt_z']].to_numpy(dtype=np.float32)

    fake_depth = np.zeros(image.shape[:2], dtype=np.float32)
    for obj in gt_objects:
        fake_depth = np.maximum(fake_depth, _fake_depth_from_bbox(image.shape, obj.bbox_2d, obj.depth))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    draw_2d_bboxes(axes[0], image, detections=detections, gt_objects=gt_objects, title=f'KITTI {frame_id}: YOLO vs GT')
    im = draw_depth_overlay(axes[1], image, fake_depth, title='Oracle depth inside GT boxes', alpha=0.55, vmax=80)
    draw_birds_eye_view(axes[2], pred_locs=pred_locs, gt_locs=gt_locs, title='Lift-to-3D sanity check', range_m=80)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Depth, m')
    fig.tight_layout()
    out_path = RESULTS_DIR / 'A5_visualization_demo.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = {
        'status': 'ok',
        'frame_id': frame_id,
        'visualization': str(out_path),
        'detections': len(detections),
        'gt_objects': len(gt_objects),
        'matched_3d_objects': int(len(rows)),
    }
    write_json(RESULTS_DIR / 'A5_visualization_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_all_track_a():
    return {
        'A1': run_a1(),
        'A2': run_a2(),
        'A3': run_a3(),
        'A4': run_a4(),
        'A5': run_a5(),
    }


if __name__ == '__main__':
    run_all_track_a()
