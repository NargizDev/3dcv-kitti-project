import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


CLASS_COLORS = {
    'Car': 'lime',
    'Pedestrian': 'cyan',
    'Cyclist': 'yellow',
    'Van': 'orange',
    'Truck': 'magenta',
}


def draw_2d_bboxes(ax, image, detections=None, gt_objects=None, title=None):
    """Нарисовать RGB, предсказанные bbox и опционально пунктирные GT bbox."""
    ax.imshow(image)
    detections = detections or []
    gt_objects = gt_objects or []

    for det in detections:
        cls = det.get('class', 'object')
        color = CLASS_COLORS.get(cls, 'lime')
        x1, y1, x2, y2 = det['bbox_2d']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        conf = det.get('confidence')
        label = f'{cls} {conf:.2f}' if conf is not None else cls
        ax.text(x1, max(0, y1 - 4), label, color=color, fontsize=8, weight='bold')

    for gt in gt_objects:
        if isinstance(gt, dict):
            cls = gt.get('class', 'GT')
            bbox = gt.get('bbox_2d')
        else:
            cls = getattr(gt, 'type', 'GT')
            bbox = getattr(gt, 'bbox_2d', None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='white',
                                 facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, min(image.shape[0] - 1, y2 + 10), f'GT {cls}', color='white', fontsize=8)

    if title:
        ax.set_title(title)
    ax.axis('off')


def draw_birds_eye_view(ax, pred_locs=None, gt_locs=None, title='Вид сверху', range_m=80):
    """Нарисовать простое сравнение предсказаний и GT в плоскости X/Z."""
    pred_locs = np.asarray(pred_locs if pred_locs is not None else [], dtype=np.float32)
    gt_locs = np.asarray(gt_locs if gt_locs is not None else [], dtype=np.float32)

    ax.scatter([0], [0], marker='^', color='tab:blue', s=120, label='Камера')

    if gt_locs.size:
        gt_locs = gt_locs.reshape(-1, 3)
        ax.scatter(gt_locs[:, 0], gt_locs[:, 2], color='white', edgecolor='black', marker='s', s=55, label='GT')

    if pred_locs.size:
        pred_locs = pred_locs.reshape(-1, 3)
        ax.scatter(pred_locs[:, 0], pred_locs[:, 2], color='lime', edgecolor='black', marker='o', s=45,
                   label='Предсказание')

    if pred_locs.size and gt_locs.size and len(pred_locs) == len(gt_locs):
        for pred, gt in zip(pred_locs, gt_locs):
            ax.plot([pred[0], gt[0]], [pred[2], gt[2]], color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlim(-range_m / 2, range_m / 2)
    ax.set_ylim(0, range_m)
    ax.set_xlabel('X, м')
    ax.set_ylabel('Z, м')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')


def draw_depth_overlay(ax, image, depth_map, title='Наложение глубины', alpha=0.45, vmax=80):
    """Нарисовать RGB с полупрозрачной тепловой картой глубины."""
    ax.imshow(image)
    im = ax.imshow(depth_map, cmap='plasma', alpha=alpha, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    return im
