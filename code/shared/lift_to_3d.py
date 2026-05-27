import numpy as np


def lift_2d_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Восстановить 3D-точку из пикселя изображения и метрической глубины.

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
    """
    if depth is None or not np.isfinite(depth) or depth <= 0:
        raise ValueError(f'Глубина должна быть положительным конечным числом, получено {depth}')
    if fx == 0 or fy == 0:
        raise ValueError('Фокусные расстояния камеры fx и fy не должны быть нулевыми')

    z = float(depth)
    x = (float(u) - float(cx)) * z / float(fx)
    y = (float(v) - float(cy)) * z / float(fy)
    return np.array([x, y, z], dtype=np.float32)


def _clip_bbox(bbox_2d, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def extract_bbox_depth(
    bbox_2d,
    depth_map,
    aggregation='median',
    bbox_shrink=0.2,
    min_depth=0.1,
    max_depth=100.0,
    min_valid_pixels=10,
):
    """
    Агрегировать валидные значения метрической глубины внутри 2D bbox.

    bbox_shrink срезает края bbox, чтобы уменьшить вклад фона. Возвращается
    глубина в метрах или None, если валидных пикселей слишком мало.
    """
    if depth_map.ndim != 2:
        raise ValueError(f'Карта глубины должна быть 2D, получена форма {depth_map.shape}')
    if not 0 <= bbox_shrink < 1:
        raise ValueError('bbox_shrink должен быть в диапазоне [0, 1)')

    x1, y1, x2, y2 = _clip_bbox(bbox_2d, depth_map.shape)
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    dx = width * bbox_shrink / 2.0
    dy = height * bbox_shrink / 2.0

    xs1 = int(np.floor(x1 + dx))
    ys1 = int(np.floor(y1 + dy))
    xs2 = int(np.ceil(x2 - dx))
    ys2 = int(np.ceil(y2 - dy))

    h, w = depth_map.shape
    xs1 = max(0, min(w, xs1))
    xs2 = max(0, min(w, xs2))
    ys1 = max(0, min(h, ys1))
    ys2 = max(0, min(h, ys2))

    if xs2 <= xs1 or ys2 <= ys1:
        return None

    values = depth_map[ys1:ys2, xs1:xs2].astype(np.float32).reshape(-1)
    valid = values[np.isfinite(values) & (values > min_depth) & (values < max_depth)]
    if valid.size < min_valid_pixels:
        return None

    if aggregation == 'median':
        return float(np.median(valid))
    if aggregation == 'mean':
        return float(np.mean(valid))
    if aggregation == 'percentile_30':
        return float(np.percentile(valid, 30))
    raise ValueError(f'Неизвестный способ агрегации глубины: {aggregation}')


def bbox_anchor_point(bbox_2d, anchor='center'):
    """Вернуть точку изображения, через которую выполняется lift-to-3D."""
    x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    u = (x1 + x2) / 2.0
    if anchor == 'center':
        v = (y1 + y2) / 2.0
    elif anchor == 'bottom':
        v = y2
    else:
        raise ValueError(f'Неизвестная точка-якорь: {anchor}')
    return u, v


def lift_bbox_to_3d(
    bbox_2d,
    depth_map,
    intrinsics,
    anchor='center',
    aggregation='median',
    bbox_shrink=0.2,
    min_depth=0.1,
    max_depth=100.0,
    min_valid_pixels=10,
):
    """
    Преобразовать 2D-детекцию и метрическую карту глубины в 3D-локацию.

    Возвращает словарь с location, anchor, aggregation и depth или None,
    если внутри bbox недостаточно валидных пикселей глубины.
    """
    depth = extract_bbox_depth(
        bbox_2d=bbox_2d,
        depth_map=depth_map,
        aggregation=aggregation,
        bbox_shrink=bbox_shrink,
        min_depth=min_depth,
        max_depth=max_depth,
        min_valid_pixels=min_valid_pixels,
    )
    if depth is None:
        return None

    fx, fy, cx, cy = intrinsics
    u, v = bbox_anchor_point(bbox_2d, anchor=anchor)
    location = lift_2d_to_3d(u, v, depth, fx, fy, cx, cy)
    return {
        'location_3d': location,
        'depth': float(depth),
        'anchor': anchor,
        'aggregation': aggregation,
        'pixel': (float(u), float(v)),
    }


def calibrate_relative_depth(relative_depth, gt_depth, mask):
    """
    Подобрать scale и shift для перевода относительной глубины в метры.

    Решается least squares задача:
        gt_depth ~= scale * relative_depth + shift
    """
    relative_depth = np.asarray(relative_depth, dtype=np.float32)
    gt_depth = np.asarray(gt_depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    if relative_depth.shape != gt_depth.shape or relative_depth.shape != mask.shape:
        raise ValueError(
            'relative_depth, gt_depth и mask должны иметь одинаковую форму: '
            f'{relative_depth.shape}, {gt_depth.shape}, {mask.shape}'
        )

    rel = relative_depth[mask]
    gt = gt_depth[mask]
    valid = np.isfinite(rel) & np.isfinite(gt)
    rel = rel[valid]
    gt = gt[valid]

    if rel.size < 2:
        raise ValueError('Для калибровки нужно минимум 2 валидные пары depth')

    a = np.stack([rel, np.ones_like(rel)], axis=1)
    scale, shift = np.linalg.lstsq(a, gt, rcond=None)[0]
    return float(scale), float(shift)
