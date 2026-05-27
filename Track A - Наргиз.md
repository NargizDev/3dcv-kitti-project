# Задание для Наргиз — Трек A
## Реальные данные KITTI и 2D-детекция → Lift-to-3D

> Это твоё подробное задание. Мирорит структуру задания Лены, чтобы было удобно сверяться. Все артефакты — в `code/track_a/`, `results/track_a/` и `code/shared/` общей папки `3dcv-project`.

---

## 0. Контекст и твоя роль

**Кратко суть:** мы строим пайплайн «Lift-to-3D», который определяет 3D-координаты объектов через комбинацию YOLOv8 (2D-детекция) + Depth Anything v2 (моно-глубина) + back-projection через калибровочную матрицу.

**Распределение:**
- **Ты (Track A):** реальные данные KITTI + 2D-детекция (YOLOv8) + алгоритм Lift-to-3D + интеграция
- **Лена (Track B):** синтетические данные VKITTI2 + Depth Anything v2 + анализ domain gap

После Sync-day (день 7) ты подхватываешь интеграцию — про это отдельный документ `Joint_Integration_Tasks.md`.

---

## 1. Подготовка среды (день 1)

### 1.1. Структура папок
Уже создана при загрузке `3dcv-project` в Drive. Твоя территория:
- `data/kitti/` — туда уже положила 3 zip-архива
- `code/track_a/` — твои ноутбуки
- `code/shared/` — общие модули (kitti_loader, lift_to_3d, metrics)
- `results/track_a/` — твои результаты

### 1.2. Google Colab
- Все ноутбуки → Drive `code/track_a/`
- Runtime → T4 GPU (бесплатный)

---

## 2. Задача A1 — Подготовка KITTI (день 1)

### Цель
Распаковать KITTI, написать loader и парсер аннотаций.

### Что нужно сделать

**Шаг 1.** Создай `code/track_a/01_kitti_setup.ipynb`. Распаковка:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile
KITTI_DRIVE = '/content/drive/MyDrive/3dcv-project/data/kitti'
KITTI_LOCAL = '/content/kitti'
os.makedirs(KITTI_LOCAL, exist_ok=True)

for archive in ['data_object_image_2.zip', 'data_object_label_2.zip', 'data_object_calib.zip']:
    archive_path = f'{KITTI_DRIVE}/{archive}'
    if os.path.exists(archive_path):
        print(f'📦 Распаковываю {archive}...')
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(KITTI_LOCAL)
        print(f'   ✅ Готово')

# Проверка структуры
!find {KITTI_LOCAL} -maxdepth 3 -type d
```

Ожидаемая структура:
```
kitti/
├── training/
│   ├── image_2/      ← 7481 RGB-изображения (.png)
│   ├── label_2/      ← 7481 аннотаций (.txt)
│   └── calib/        ← 7481 калибровочных файлов (.txt)
└── testing/
    ├── image_2/      ← 7518 без аннотаций
    └── calib/
```

**Шаг 2.** Напиши модуль `code/shared/kitti_loader.py`:

```python
import os
import numpy as np
from PIL import Image

class KITTIObject:
    """Один объект из KITTI label-файла"""
    def __init__(self, line):
        parts = line.strip().split()
        self.type = parts[0]                              # Car, Pedestrian, Cyclist...
        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])
        self.bbox_2d = np.array([float(x) for x in parts[4:8]])   # x1, y1, x2, y2
        self.dimensions = np.array([float(x) for x in parts[8:11]])  # h, w, l (метры)
        self.location = np.array([float(x) for x in parts[11:14]])  # x, y, z в камере (метры)
        self.rotation_y = float(parts[14])
    
    @property
    def depth(self):
        """Глубина центра 3D-bbox = z-координата в камере"""
        return self.location[2]


class KITTILoader:
    """Загрузчик KITTI 3D Object Detection"""
    
    CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc']
    
    def __init__(self, root_dir, split='training'):
        self.root = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root, 'image_2')
        self.label_dir = os.path.join(self.root, 'label_2')
        self.calib_dir = os.path.join(self.root, 'calib')
        self.frame_ids = sorted([f.replace('.png', '') 
                                 for f in os.listdir(self.image_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.frame_ids)
    
    def load_image(self, frame_id):
        path = os.path.join(self.image_dir, f'{frame_id}.png')
        return np.array(Image.open(path))
    
    def load_labels(self, frame_id):
        """Возвращает список объектов KITTIObject"""
        path = os.path.join(self.label_dir, f'{frame_id}.txt')
        if not os.path.exists(path):
            return []
        objects = []
        with open(path) as f:
            for line in f:
                obj = KITTIObject(line)
                if obj.type != 'DontCare':
                    objects.append(obj)
        return objects
    
    def load_calib(self, frame_id):
        """Возвращает словарь с матрицами калибровки"""
        path = os.path.join(self.calib_dir, f'{frame_id}.txt')
        calib = {}
        with open(path) as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array([float(x) for x in value.split()])
        # Reshape стандартных матриц
        calib['P2'] = calib['P2'].reshape(3, 4)
        if 'R0_rect' in calib:
            calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in calib:
            calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
        return calib
    
    def get_intrinsics(self, frame_id):
        """Извлекает (fx, fy, cx, cy) из P2 матрицы"""
        P2 = self.load_calib(frame_id)['P2']
        K = P2[:, :3]
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
```

**Шаг 3.** Тестовая визуализация — RGB + 2D bbox + информация о 3D-локации:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append('/content/drive/MyDrive/3dcv-project/code/shared')
from kitti_loader import KITTILoader

kitti = KITTILoader('/content/kitti', split='training')
print(f'Всего кадров: {len(kitti)}')

# Визуализируем 4 случайных
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

import random
random.seed(42)
sample_ids = random.sample(kitti.frame_ids, 4)

for ax, fid in zip(axes, sample_ids):
    img = kitti.load_image(fid)
    ax.imshow(img)
    
    for obj in kitti.load_labels(fid):
        if obj.type not in ['Car', 'Pedestrian', 'Cyclist']:
            continue
        x1, y1, x2, y2 = obj.bbox_2d
        color = {'Car': 'lime', 'Pedestrian': 'cyan', 'Cyclist': 'yellow'}[obj.type]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # Подпись с глубиной
        ax.text(x1, y1-5, f'{obj.type} z={obj.depth:.1f}м', 
                color=color, fontsize=8, weight='bold')
    
    ax.set_title(f'KITTI #{fid}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/3dcv-project/results/track_a/A1_kitti_samples.png', 
            dpi=150, bbox_inches='tight')
plt.show()
```

### Артефакты к концу A1
- ✅ `code/shared/kitti_loader.py` — модуль с классами `KITTIObject`, `KITTILoader`
- ✅ `code/track_a/01_kitti_setup.ipynb` — ноутбук
- ✅ `results/track_a/A1_kitti_samples.png` — визуализация 4 примеров
- ✅ Запись в `coordination.ipynb`: «A1 готово, 7481 кадров KITTI распакованы»

### Критерий приёмки
- `KITTILoader` корректно возвращает image, labels, calib для любого frame_id
- Матрица P2 имеет форму (3, 4); fx, fy, cx, cy — разумные числа (~700, ~700, ~600, ~190)
- Класс `KITTIObject` парсит все 15 полей KITTI label-формата

---

## 3. Задача A2 — Инференс YOLOv8 на KITTI (день 2–3)

### Цель
Запустить pre-trained YOLOv8 на KITTI, получить 2D-bbox предсказания, измерить mAP.

### Что нужно сделать

**Шаг 1.** Создай `code/track_a/02_yolov8_inference.ipynb`.

**Шаг 2.** Установка и инференс:

```python
!pip install -q ultralytics

from ultralytics import YOLO
import json
from tqdm.notebook import tqdm

# Загружаем YOLOv8m — баланс между скоростью и точностью
# YOLOv8 предобучен на COCO (Car=2, Person=0, Bicycle=1)
model = YOLO('yolov8m.pt')
print('✅ YOLOv8m загружен')

# Маппинг COCO → KITTI классы
COCO_TO_KITTI = {
    2: 'Car',           # COCO car
    0: 'Pedestrian',    # COCO person
    1: 'Cyclist',       # COCO bicycle (приближение)
    7: 'Truck',         # COCO truck
}
```

**Шаг 3.** Инференс на всех изображениях KITTI training:

```python
KITTI_IMG_DIR = '/content/kitti/training/image_2'
PRED_DIR = '/content/drive/MyDrive/3dcv-project/results/track_a/yolov8_predictions'
os.makedirs(PRED_DIR, exist_ok=True)

frame_ids = sorted([f.replace('.png', '') 
                    for f in os.listdir(KITTI_IMG_DIR) if f.endswith('.png')])

for fid in tqdm(frame_ids):
    save_path = f'{PRED_DIR}/{fid}.json'
    if os.path.exists(save_path):
        continue  # уже обработано
    
    img_path = f'{KITTI_IMG_DIR}/{fid}.png'
    results = model(img_path, conf=0.25, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        coco_cls = int(box.cls[0])
        if coco_cls not in COCO_TO_KITTI:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0])
        detections.append({
            'class': COCO_TO_KITTI[coco_cls],
            'bbox_2d': [x1, y1, x2, y2],
            'confidence': conf,
        })
    
    with open(save_path, 'w') as f:
        json.dump({'frame_id': fid, 'detections': detections}, f)

print(f'\n✅ Готово: {len(frame_ids)} кадров обработано')
```

**Шаг 4.** Оценка mAP — сравнение с GT KITTI:

```python
import numpy as np

def compute_iou_2d(box1, box2):
    """IoU между двумя 2D bbox в формате [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-8)

def evaluate_2d_detection(pred_dir, kitti_loader, target_class='Car', iou_threshold=0.5):
    """Простая метрика precision/recall для одного класса"""
    tp, fp, fn = 0, 0, 0
    
    for fid in kitti_loader.frame_ids:
        gt_objs = [o for o in kitti_loader.load_labels(fid) if o.type == target_class]
        gt_bboxes = [o.bbox_2d for o in gt_objs]
        
        pred_path = f'{pred_dir}/{fid}.json'
        if not os.path.exists(pred_path):
            fn += len(gt_bboxes)
            continue
        with open(pred_path) as f:
            preds = json.load(f)['detections']
        pred_bboxes = [p['bbox_2d'] for p in preds if p['class'] == target_class]
        
        matched_gt = set()
        for pb in pred_bboxes:
            best_iou, best_gt = 0, -1
            for i, gb in enumerate(gt_bboxes):
                if i in matched_gt:
                    continue
                iou = compute_iou_2d(pb, gb)
                if iou > best_iou:
                    best_iou, best_gt = iou, i
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1
        fn += len(gt_bboxes) - len(matched_gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1, 
            'tp': tp, 'fp': fp, 'fn': fn}

# Берём первые 1000 кадров для скорости (можно потом расширить)
kitti_subset = KITTILoader('/content/kitti')
kitti_subset.frame_ids = kitti_subset.frame_ids[:1000]

for cls in ['Car', 'Pedestrian', 'Cyclist']:
    metrics = evaluate_2d_detection(PRED_DIR, kitti_subset, target_class=cls)
    print(f'{cls}: P={metrics["precision"]:.3f}, R={metrics["recall"]:.3f}, F1={metrics["f1"]:.3f}')
```

### Артефакты к концу A2
- ✅ `code/track_a/02_yolov8_inference.ipynb`
- ✅ `results/track_a/yolov8_predictions/*.json` — предсказания по каждому кадру
- ✅ `results/track_a/A2_2d_metrics.csv` — таблица P/R/F1 по классам

### Критерий приёмки
- Минимум 1000 JSON-файлов в `yolov8_predictions/`
- Recall для класса Car > 0.7 (YOLOv8m на KITTI обычно даёт ~0.85)
- Каждый JSON имеет поля `frame_id` и `detections` со списком объектов

---

## 4. Задача A3 — Алгоритм Lift-to-3D (день 4–5)

### Цель
**Самая важная техническая часть.** Реализовать back-projection из 2D в 3D с использованием калибровки камеры и оценки глубины.

### Математика

Точка `(u, v)` в пикселях с глубиной `Z` (метры) проецируется в 3D-координаты камеры:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$

где `(fx, fy, cx, cy)` — параметры камеры, извлекаемые из матрицы P2 KITTI.

### Что нужно сделать

**Шаг 1.** Создай модуль `code/shared/lift_to_3d.py`:

```python
import numpy as np

def lift_2d_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Back-project 2D-пиксель + глубину в 3D-координаты камеры.
    
    Параметры:
    ----------
    u, v: координаты пикселя
    depth: оценка глубины в этой точке (метры)
    fx, fy, cx, cy: параметры камеры
    
    Возвращает:
    ----------
    np.array([X, Y, Z]) — 3D-координаты в системе камеры (метры)
    """
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def lift_bbox_to_3d(bbox_2d, depth_map, intrinsics, depth_aggregation='median', 
                     bbox_shrink=0.2):
    """
    Превращает 2D-bbox + depth map в 3D-локацию объекта.
    
    Параметры:
    ----------
    bbox_2d: [x1, y1, x2, y2] — 2D bbox от YOLO
    depth_map: (H, W) — метрическая глубина
    intrinsics: (fx, fy, cx, cy)
    depth_aggregation: 'median' / 'mean' / 'percentile_30' — как аггрегировать глубину в bbox
    bbox_shrink: float — на сколько ужимать bbox для сэмплинга (избегаем фоновых пикселей по краям)
    
    Возвращает:
    ----------
    (X, Y, Z) или None если глубина недоступна
    """
    x1, y1, x2, y2 = bbox_2d
    fx, fy, cx, cy = intrinsics
    
    # Ужимаем bbox, чтобы избежать фоновых пикселей по краям
    w = x2 - x1
    h = y2 - y1
    sx = w * bbox_shrink / 2
    sy = h * bbox_shrink / 2
    x1_s, y1_s = int(x1 + sx), int(y1 + sy)
    x2_s, y2_s = int(x2 - sx), int(y2 - sy)
    
    # Извлекаем глубину в bbox
    H, W = depth_map.shape
    x1_s = max(0, x1_s); y1_s = max(0, y1_s)
    x2_s = min(W, x2_s); y2_s = min(H, y2_s)
    
    bbox_depth = depth_map[y1_s:y2_s, x1_s:x2_s]
    valid = bbox_depth[bbox_depth > 0.1]
    
    if len(valid) < 10:
        return None
    
    # Аггрегируем
    if depth_aggregation == 'median':
        Z = np.median(valid)
    elif depth_aggregation == 'mean':
        Z = np.mean(valid)
    elif depth_aggregation == 'percentile_30':
        Z = np.percentile(valid, 30)  # ближе к фронту объекта
    else:
        raise ValueError(f'Неизвестный метод: {depth_aggregation}')
    
    # 2D-центр bbox
    u_c = (x1 + x2) / 2
    # Для дороги — низ bbox обычно соответствует низу объекта (для машин — колёса на земле)
    v_c = (y1 + y2) / 2
    
    return lift_2d_to_3d(u_c, v_c, Z, fx, fy, cx, cy)


def calibrate_relative_depth(relative_depth, gt_depth, mask):
    """
    Подгоняет scale + shift для перевода относительной глубины в метрическую.
    
    relative_depth: (H, W) выход Depth Anything (0-1 или произвольный диапазон)
    gt_depth: (H, W) истинная глубина в метрах (где есть)
    mask: (H, W) bool — где есть валидный GT
    
    Возвращает:
    ----------
    (scale, shift) — параметры аффинной калибровки
    """
    rel = relative_depth[mask]
    gt = gt_depth[mask]
    A = np.stack([rel, np.ones_like(rel)], axis=1)
    scale, shift = np.linalg.lstsq(A, gt, rcond=None)[0]
    return scale, shift
```

**Шаг 2.** Тестовый ноутбук `code/track_a/03_lift_to_3d.ipynb`:

```python
import sys
sys.path.append('/content/drive/MyDrive/3dcv-project/code/shared')
from kitti_loader import KITTILoader
from lift_to_3d import lift_bbox_to_3d, calibrate_relative_depth

# На этом этапе depth-предсказаний от Лены может ещё не быть
# Поэтому тестируем lift на ГЛУБИНЕ ИЗ GT KITTI (sparse depth)
# Для этого используем z-координаты GT 3D bboxes как "истинную" глубину в bbox-центрах

kitti = KITTILoader('/content/kitti')

# Берём один кадр и проверяем, что lift даёт примерно те же 3D-координаты, что и GT
fid = '000010'
objects = kitti.load_labels(fid)
intrinsics = kitti.get_intrinsics(fid)

print(f'Кадр {fid}, {len(objects)} объектов')
print(f'Intrinsics: fx={intrinsics[0]:.1f}, fy={intrinsics[1]:.1f}, cx={intrinsics[2]:.1f}, cy={intrinsics[3]:.1f}\n')

for obj in objects[:3]:  # первые 3 объекта
    if obj.type not in ['Car', 'Pedestrian', 'Cyclist']:
        continue
    
    # Создаём фейковый depth map с правильной глубиной в области bbox
    H, W = kitti.load_image(fid).shape[:2]
    fake_depth = np.zeros((H, W), dtype=np.float32)
    x1, y1, x2, y2 = obj.bbox_2d
    fake_depth[int(y1):int(y2), int(x1):int(x2)] = obj.depth
    
    # Lift
    pred_3d = lift_bbox_to_3d(obj.bbox_2d, fake_depth, intrinsics)
    
    print(f'{obj.type}:')
    print(f'  GT 3D location:   X={obj.location[0]:.2f}, Y={obj.location[1]:.2f}, Z={obj.location[2]:.2f}')
    print(f'  Predicted 3D:     X={pred_3d[0]:.2f}, Y={pred_3d[1]:.2f}, Z={pred_3d[2]:.2f}')
    print(f'  Error (Euclid):   {np.linalg.norm(pred_3d - obj.location):.2f} м\n')
```

**Шаг 3.** Если разница X/Z небольшая, но Y значительно отличается от GT — это ожидаемо. KITTI хранит location как **низ bbox** (на дороге), а мы лифтуем центр bbox. Это нужно скорректировать на этапе интеграции, либо использовать низ bbox вместо центра. Зафиксируй это в `coordination.ipynb` как «открытый вопрос для интеграции».

### Артефакты к концу A3
- ✅ `code/shared/lift_to_3d.py` — функции `lift_2d_to_3d`, `lift_bbox_to_3d`, `calibrate_relative_depth`
- ✅ `code/track_a/03_lift_to_3d.ipynb` — тесты с GT-глубиной
- ✅ Решение по аггрегации (median/mean/percentile) и точке-якорю (центр / низ bbox) — в `coordination.ipynb`

### Критерий приёмки
- Если используется `obj.depth` напрямую как глубина и низ bbox как точка-якорь, ошибка по X и Z должна быть < 0.5 м (это означает корректность математики)

---

## 5. Задача A4 — Метрики 3D-локализации (день 5)

### Цель
Реализовать метрики оценки качества 3D-локализации для финальных экспериментов.

### Что нужно сделать

**Шаг 1.** Дополни модуль `code/shared/metrics.py`:

```python
import numpy as np

def euclidean_3d_error(pred_loc, gt_loc):
    """Евклидово расстояние между предсказанной и истинной 3D-локацией"""
    return np.linalg.norm(np.array(pred_loc) - np.array(gt_loc))

def depth_error(pred_loc, gt_loc):
    """Ошибка только по глубине (Z)"""
    return abs(pred_loc[2] - gt_loc[2])

def relative_depth_error(pred_loc, gt_loc):
    """Относительная ошибка по глубине"""
    return abs(pred_loc[2] - gt_loc[2]) / gt_loc[2]

def localization_accuracy(pred_locs, gt_locs, threshold_m=2.0):
    """
    Доля предсказаний, попавших в радиус threshold метров от GT.
    pred_locs, gt_locs: списки (N, 3)
    """
    if len(pred_locs) == 0:
        return 0
    errors = [euclidean_3d_error(p, g) for p, g in zip(pred_locs, gt_locs)]
    return np.mean([e < threshold_m for e in errors])


def match_predictions_to_gt(pred_dets, gt_objs, iou_threshold=0.5):
    """
    Сопоставляет предсказания с GT по 2D-IoU (Hungarian-like).
    Возвращает список пар (pred, gt) для дальнейшей оценки 3D.
    """
    from itertools import product
    
    matches = []
    used_gt = set()
    
    # Сортируем preds по confidence
    pred_dets = sorted(pred_dets, key=lambda x: -x.get('confidence', 1))
    
    for pred in pred_dets:
        if pred['class'] not in ['Car', 'Pedestrian', 'Cyclist']:
            continue
        
        best_iou, best_gt = 0, None
        for i, gt in enumerate(gt_objs):
            if i in used_gt or gt.type != pred['class']:
                continue
            from kitti_loader import KITTIObject  # для compute_iou_2d при необходимости
            # Простой IoU здесь
            box1 = pred['bbox_2d']
            box2 = gt.bbox_2d
            x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
            if x2 < x1 or y2 < y1:
                iou = 0
            else:
                inter = (x2-x1)*(y2-y1)
                a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
                a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
                iou = inter / (a1 + a2 - inter + 1e-8)
            
            if iou > best_iou:
                best_iou, best_gt = iou, i
        
        if best_iou >= iou_threshold and best_gt is not None:
            matches.append((pred, gt_objs[best_gt]))
            used_gt.add(best_gt)
    
    return matches
```

**Шаг 2.** Создай ноутбук `code/track_a/04_metrics_3d.ipynb` для проверки метрик на синтетическом примере. Финальное использование — на этапе интеграции.

### Артефакты к концу A4
- ✅ `code/shared/metrics.py` — функции метрик
- ✅ `code/track_a/04_metrics_3d.ipynb` — unit-тесты функций

### Критерий приёмки
- Все функции корректно работают на синтетических данных (например, `euclidean_3d_error([0,0,10], [0,0,10])` возвращает 0)

---

## 6. Задача A5 — Визуализации (день 6)

### Цель
Подготовить функции визуализации, которые понадобятся для статьи.

### Что нужно сделать

Создай `code/shared/visualization.py` с тремя функциями:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_2d_bboxes(ax, image, detections, gt_objects=None):
    """Рисует 2D bbox: предсказания зелёным, GT белым штриховкой"""
    ax.imshow(image)
    for det in detections:
        x1, y1, x2, y2 = det['bbox_2d']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{det["class"]} {det.get("confidence", 0):.2f}',
                color='lime', fontsize=8, weight='bold')
    if gt_objects:
        for gt in gt_objects:
            if gt.type not in ['Car', 'Pedestrian', 'Cyclist']:
                continue
            x1, y1, x2, y2 = gt.bbox_2d
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=1.5, edgecolor='white', 
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
    ax.axis('off')

def draw_birds_eye_view(ax, pred_locs, gt_locs, range_m=60):
    """BEV: вид сверху, предсказания vs GT"""
    ax.scatter([0], [0], marker='^', color='blue', s=200, label='Camera')
    if gt_locs:
        gt = np.array(gt_locs)
        ax.scatter(gt[:, 0], gt[:, 2], color='white', edgecolor='black', s=80, 
                   marker='s', label='GT')
    if pred_locs:
        pred = np.array(pred_locs)
        ax.scatter(pred[:, 0], pred[:, 2], color='lime', s=60, marker='o', label='Predicted')
        # Линии соединения GT-pred
        if gt_locs and len(pred_locs) == len(gt_locs):
            for p, g in zip(pred, gt):
                ax.plot([p[0], g[0]], [p[2], g[2]], 'r--', alpha=0.3, linewidth=0.5)
    ax.set_xlim(-range_m/2, range_m/2)
    ax.set_ylim(0, range_m)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Z — глубина (м)')
    ax.set_title("Bird's-Eye View")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

def draw_depth_overlay(ax, image, depth_map, alpha=0.5, vmax=80):
    """RGB + depth heatmap полупрозрачный"""
    ax.imshow(image)
    ax.imshow(depth_map, cmap='plasma', alpha=alpha, vmin=0, vmax=vmax)
    ax.axis('off')
```

### Артефакты к концу A5
- ✅ `code/shared/visualization.py` — три функции отрисовки
- ✅ `code/track_a/05_visualizations.ipynb` — демо использования

### Критерий приёмки
- Все функции принимают `ax` (matplotlib Axes) и работают как часть subplot-сетки
- Без хардкода размеров (адаптивно к данным)

---

## 7. День 7 — Sync-day с Леной

К этому моменту:

**Что отдаёшь Лене:**
- `code/shared/kitti_loader.py` — она использует для своего инференса на KITTI
- Доступ к `data/kitti/` (через общую папку)
- Документация формата JSON-предсказаний YOLO (см. `coordination.ipynb` → раздел Sync-day)

**Что получаешь от Лены:**
- `code/shared/vkitti_loader.py`
- `results/track_b/depth_pred_kitti/*.npy` — предсказания глубины на KITTI
- `results/track_b/depth_pred_vkitti/*.npy` — на VKITTI2
- `results/track_b/B3_depth_metrics_vkitti.csv` — метрики глубины
- `results/track_b/configs/*.json` — конфиги стратегий

**Чек-лист на sync-day** — заполни в `coordination.ipynb`:
- Все ли файлы Лены реально на месте (физическая проверка)
- Совпадают ли ключи кадров (KITTI frame_id ↔ имена .npy файлов)
- Размеры depth_map vs размеры RGB-изображений
- Согласованность с твоим `kitti_loader`

После sync-day переходишь на документ `Joint_Integration_Tasks.md` — там описана фаза интеграции и финальных экспериментов.

---

## 8. Чек-лист готовности (отмечать в `coordination.ipynb`)

```
[ ] A1.1 — KITTI распакован, структура изучена
[ ] A1.2 — KITTILoader написан и протестирован
[ ] A1.3 — Визуализация образцов сохранена

[ ] A2.1 — YOLOv8m запущен в Colab
[ ] A2.2 — Инференс на 7000+ кадрах KITTI завершён
[ ] A2.3 — JSON-файлы предсказаний сохранены
[ ] A2.4 — 2D метрики (P/R/F1) посчитаны для Car

[ ] A3.1 — lift_2d_to_3d реализован и проверен
[ ] A3.2 — lift_bbox_to_3d работает на тестовых данных
[ ] A3.3 — calibrate_relative_depth реализован
[ ] A3.4 — Решение по точке-якорю и аггрегации зафиксировано

[ ] A4.1 — Функции метрик реализованы
[ ] A4.2 — Unit-тесты проходят

[ ] A5.1 — Функции визуализации готовы
[ ] A5.2 — Демо-ноутбук показывает все три функции

[ ] Sync-day готовность: все артефакты Track A в общей папке
```

---

## 9. FAQ

**В:** YOLOv8 не находит велосипедистов (Cyclist).
**О:** Это нормально, в COCO нет класса Cyclist напрямую, мы маппим на `bicycle`. Низкий recall для Cyclist — известное ограничение, отметим в статье. Фокус на Car.

**В:** Calib-файлы выглядят странно — много матриц.
**О:** В KITTI несколько матриц: P0, P1, P2, P3 — для четырёх камер; R0_rect — ректификация; Tr_velo_to_cam — для LiDAR. Нам нужна только P2 (левая цветная камера).

**В:** Глубина в KITTI labels (`obj.depth = location[2]`) — это глубина центра 3D-bbox, не bottom.
**О:** Да. Поэтому при сравнении lift-результатов с GT location — учитывай это. Можно либо лифтовать центр bbox (тогда сравнение прямое), либо лифтовать низ bbox + добавить height/2 для оценки центра.

**В:** Что делать, если YOLO находит больше объектов, чем есть в GT?
**О:** При оценке используем функцию `match_predictions_to_gt` — она сначала сматчит предсказания с GT по IoU, остальные считаем как FP. Только сматченные пары идут в 3D-метрики.

**В:** Я застряла, не понимаю шаг.
**О:** Пиши мне (Claude) с конкретной ошибкой/вопросом. Параллельно — фиксируй блокер в `coordination.ipynb` для прозрачности перед Леной.

---

## 10. Ссылки

- **KITTI 3D Object Detection:** https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- **YOLOv8 docs:** https://docs.ultralytics.com/
- **KITTI calib explanation:** https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
- **Pinhole camera model:** https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

---

*Удачи! Я (Claude) сопровождаю на каждом этапе — пиши, если что-то непонятно или падает.*
