import numpy as np


def compute_iou_2d(box_a, box_b):
    """Посчитать IoU для bbox в формате [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def euclidean_3d_error(pred_loc, gt_loc):
    """Евклидова ошибка между предсказанной и GT 3D-локациями."""
    return float(np.linalg.norm(np.asarray(pred_loc, dtype=np.float32) - np.asarray(gt_loc, dtype=np.float32)))


def depth_error(pred_loc, gt_loc):
    """Абсолютная ошибка по оси Z в метрах."""
    return float(abs(float(pred_loc[2]) - float(gt_loc[2])))


def relative_depth_error(pred_loc, gt_loc):
    """Относительная ошибка по глубине Z."""
    gt_z = float(gt_loc[2])
    if gt_z == 0:
        return None
    return float(abs(float(pred_loc[2]) - gt_z) / abs(gt_z))


def localization_accuracy(errors, threshold_m):
    """Доля matched-предсказаний с 3D-ошибкой меньше threshold_m."""
    errors = np.asarray(errors, dtype=np.float32)
    if errors.size == 0:
        return 0.0
    return float(np.mean(errors < float(threshold_m)))


def summarize_errors(errors_3d, errors_depth):
    """Вернуть summary-статистику ошибок с учетом числа matched-объектов."""
    errors_3d = np.asarray(errors_3d, dtype=np.float32)
    errors_depth = np.asarray(errors_depth, dtype=np.float32)

    def stat(values, fn):
        return float(fn(values)) if values.size else None

    return {
        'count': int(errors_3d.size),
        'mean_3d_error': stat(errors_3d, np.mean),
        'median_3d_error': stat(errors_3d, np.median),
        'std_3d_error': stat(errors_3d, np.std),
        'mean_depth_error': stat(errors_depth, np.mean),
        'median_depth_error': stat(errors_depth, np.median),
        'std_depth_error': stat(errors_depth, np.std),
        'localization_acc_2m': localization_accuracy(errors_3d, 2.0),
        'localization_acc_4m': localization_accuracy(errors_3d, 4.0),
    }


def _gt_class(gt):
    if isinstance(gt, dict):
        return gt.get('class')
    return getattr(gt, 'type')


def _gt_bbox(gt):
    if isinstance(gt, dict):
        return gt.get('bbox_2d')
    return getattr(gt, 'bbox_2d')


def _gt_location(gt):
    if isinstance(gt, dict):
        return gt.get('location_3d', gt.get('location'))
    return getattr(gt, 'location')


def match_predictions_to_gt(predictions, gt_objects, target_class='Car', iou_threshold=0.5):
    """
    Жадное one-to-one сопоставление предсказаний и GT по 2D IoU.

    Предсказания обрабатываются по убыванию confidence. GT может быть
    KITTIObject или словарем с полями class/bbox/location.
    """
    filtered_preds = [
        pred for pred in predictions
        if pred.get('class') == target_class and 'bbox_2d' in pred
    ]
    filtered_preds = sorted(filtered_preds, key=lambda p: -float(p.get('confidence', 1.0)))

    filtered_gts = [
        (idx, gt) for idx, gt in enumerate(gt_objects)
        if _gt_class(gt) == target_class
    ]

    matches = []
    used_gt = set()

    for pred in filtered_preds:
        best_iou = 0.0
        best_idx = None
        best_gt = None
        for idx, gt in filtered_gts:
            if idx in used_gt:
                continue
            iou = compute_iou_2d(pred['bbox_2d'], _gt_bbox(gt))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
                best_gt = gt

        if best_gt is not None and best_iou >= iou_threshold:
            matches.append({'prediction': pred, 'gt': best_gt, 'iou': float(best_iou)})
            used_gt.add(best_idx)

    return matches


def evaluate_matched_locations(matches):
    """Посчитать ошибки для каждого match и итоговые метрики локализации."""
    rows = []
    errors_3d = []
    errors_depth = []

    for match in matches:
        pred = match['prediction']
        gt = match['gt']
        pred_loc = pred['location_3d']
        gt_loc = _gt_location(gt)

        err_3d = euclidean_3d_error(pred_loc, gt_loc)
        err_depth = depth_error(pred_loc, gt_loc)
        errors_3d.append(err_3d)
        errors_depth.append(err_depth)
        rows.append({
            'iou_2d': float(match['iou']),
            'euclidean_3d_error': err_3d,
            'depth_error': err_depth,
            'pred_depth': float(pred_loc[2]),
            'gt_depth': float(gt_loc[2]),
        })

    return rows, summarize_errors(errors_3d, errors_depth)
