# Задание: Интеграция и финальная фаза (Joint)
## Совместная работа Наргиз и Лены — после Sync-day

> Этот документ — для **второй недели** работы. Когда обе закончат свои треки (A1–A5 и B1–B5), начинается фаза интеграции, которой руководит **Наргиз**.

---

## 0. Контекст

К этому моменту у нас есть:

**От Track A (Наргиз):**
- 2D-предсказания YOLOv8 на KITTI (JSON по каждому кадру)
- Алгоритм Lift-to-3D
- Loader KITTI и метрики

**От Track B (Лена):**
- Depth-предсказания Depth Anything на KITTI и VKITTI2 (.npy)
- Метрики глубины на VKITTI2
- Конфиги стратегий аугментации (real/synth/mixed)
- Loader VKITTI2

**Нужно собрать это в единый пайплайн и провести финальные эксперименты для статьи.**

---

## 1. Sync-day (день 7) — детально

### Утро: проверка артефактов

**Шаг 1.** Открой общую папку Drive и проверь физическое наличие файлов:

```python
import os

EXPECTED_FROM_LENA = {
    'Loader VKITTI2': 'code/shared/vkitti_loader.py',
    'Depth predictions KITTI': 'results/track_b/depth_pred_kitti/',
    'Depth predictions VKITTI2': 'results/track_b/depth_pred_vkitti/',
    'Metrics CSV': 'results/track_b/B3_depth_metrics_vkitti.csv',
    'Configs (real_only)': 'results/track_b/configs/real_only.json',
    'Configs (synth_only)': 'results/track_b/configs/synth_only.json',
    'Configs (mixed)': 'results/track_b/configs/mixed_50_50.json',
}

ROOT = '/content/drive/MyDrive/3dcv-project'
print('Проверка артефактов от Лены:\n')
for name, path in EXPECTED_FROM_LENA.items():
    full_path = f'{ROOT}/{path}'
    exists = os.path.exists(full_path)
    if exists and os.path.isdir(full_path):
        n_files = len(os.listdir(full_path))
        status = f'✅ ({n_files} файлов)'
    elif exists:
        status = '✅'
    else:
        status = '❌ НЕ НАЙДЕНО'
    print(f'  {status} {name}: {path}')
```

**Шаг 2.** Проверка форматов и совместимости:

```python
import numpy as np
import json

# Берём один frame_id, который есть и у тебя (YOLO), и у Лены (depth)
test_fid = '000010'

# Твой YOLO output
yolo_path = f'{ROOT}/results/track_a/yolov8_predictions/{test_fid}.json'
with open(yolo_path) as f:
    yolo_data = json.load(f)
print(f'YOLO для {test_fid}: {len(yolo_data["detections"])} объектов')
print(f'  Пример: {yolo_data["detections"][0] if yolo_data["detections"] else "нет"}')

# Depth Лены
depth_path = f'{ROOT}/results/track_b/depth_pred_kitti/{test_fid}.npy'
depth = np.load(depth_path)
print(f'\nDepth shape: {depth.shape}, dtype: {depth.dtype}')
print(f'Range: [{depth.min():.3f}, {depth.max():.3f}]')

# Размеры RGB
import sys
sys.path.append(f'{ROOT}/code/shared')
from kitti_loader import KITTILoader
kitti = KITTILoader('/content/kitti')  # уже распакован у тебя локально
img = kitti.load_image(test_fid)
print(f'\nRGB shape: {img.shape}')
print(f'Совпадают ли depth и RGB? {depth.shape[:2] == img.shape[:2]}')
```

**Шаг 3.** Запиши в `coordination.ipynb` → раздел «Sync-day заметки» статус каждого пункта чек-листа. Если что-то не так — отметь как блокер и согласуйте решение в чате.

### День: совместный созвон (или текстовое обсуждение в `coordination.ipynb`)

Вопросы для обсуждения:
1. **Точка-якорь для lift-to-3D**: центр bbox или низ bbox? Решение влияет на сравнение с GT location.
2. **Аггрегация глубины в bbox**: median / mean / percentile_30? Лена может посоветовать на основе анализа domain gap.
3. **Стратегия калибровки относительной глубины**: для каждой стратегии (real/synth/mixed) — где брать GT для подгонки scale+shift?
4. **Окончательный subset KITTI** для финальной оценки: train/val split, который мы используем как «test».

### Артефакт sync-day
- ✅ Все блокеры закрыты или внесены в план Б
- ✅ Решения по 4 вопросам выше зафиксированы в `coordination.ipynb`
- ✅ Готовы переходить к интеграции

---

## 2. J2 — Интеграция пайплайна (день 8)

### Цель
Собрать единый пайплайн: KITTI image → YOLO 2D bbox → calibrated depth → Lift-to-3D → 3D location.

### Что делать

**Шаг 1.** Создай ноутбук `code/shared/integrated_pipeline.py` — основной модуль:

```python
import numpy as np
import json
import os
import sys

class IntegratedPipeline:
    """
    Полный пайплайн моно-3D-локализации.
    Использует pre-computed YOLO predictions и Depth Anything predictions.
    """
    
    def __init__(self, kitti_loader, yolo_pred_dir, depth_pred_dir,
                 depth_calibration=None):
        """
        Параметры:
        ----------
        kitti_loader: instance of KITTILoader
        yolo_pred_dir: путь до результатов YOLO (.json по каждому кадру)
        depth_pred_dir: путь до depth-предсказаний (.npy по каждому кадру)
        depth_calibration: tuple (scale, shift) или None (без калибровки)
        """
        self.kitti = kitti_loader
        self.yolo_dir = yolo_pred_dir
        self.depth_dir = depth_pred_dir
        self.scale, self.shift = depth_calibration if depth_calibration else (1.0, 0.0)
    
    def predict_3d(self, frame_id, target_classes=('Car',), 
                   anchor='bottom', aggregation='percentile_30'):
        """
        Полный inference для одного кадра.
        
        anchor: 'bottom' (низ bbox — для машин) или 'center'
        
        Возвращает список предсказаний:
        [
            {'class': 'Car', 'bbox_2d': [...], 'location_3d': [X, Y, Z], 'confidence': float},
            ...
        ]
        """
        from lift_to_3d import lift_2d_to_3d
        
        # Загружаем YOLO предсказания
        yolo_path = os.path.join(self.yolo_dir, f'{frame_id}.json')
        if not os.path.exists(yolo_path):
            return []
        with open(yolo_path) as f:
            yolo_data = json.load(f)
        
        # Загружаем depth и калибруем
        depth_path = os.path.join(self.depth_dir, f'{frame_id}.npy')
        if not os.path.exists(depth_path):
            return []
        depth_relative = np.load(depth_path).astype(np.float32)
        depth_metric = self.scale * depth_relative + self.shift
        
        # Внутренние параметры камеры
        intrinsics = self.kitti.get_intrinsics(frame_id)
        fx, fy, cx, cy = intrinsics
        
        results = []
        for det in yolo_data['detections']:
            if det['class'] not in target_classes:
                continue
            
            x1, y1, x2, y2 = det['bbox_2d']
            
            # Аггрегация глубины в bbox
            H, W = depth_metric.shape
            bbox_h, bbox_w = y2-y1, x2-x1
            shrink = 0.2
            x1s = max(0, int(x1 + bbox_w*shrink/2))
            y1s = max(0, int(y1 + bbox_h*shrink/2))
            x2s = min(W, int(x2 - bbox_w*shrink/2))
            y2s = min(H, int(y2 - bbox_h*shrink/2))
            
            bbox_depth = depth_metric[y1s:y2s, x1s:x2s]
            valid = bbox_depth[(bbox_depth > 0.5) & (bbox_depth < 100)]
            if len(valid) < 10:
                continue
            
            if aggregation == 'median':
                Z = float(np.median(valid))
            elif aggregation == 'percentile_30':
                Z = float(np.percentile(valid, 30))
            else:
                Z = float(np.mean(valid))
            
            # Точка-якорь
            u = (x1 + x2) / 2
            if anchor == 'bottom':
                v = y2  # низ bbox
            else:
                v = (y1 + y2) / 2
            
            loc_3d = lift_2d_to_3d(u, v, Z, fx, fy, cx, cy)
            
            results.append({
                'class': det['class'],
                'bbox_2d': det['bbox_2d'],
                'location_3d': loc_3d.tolist(),
                'confidence': det.get('confidence', 1.0),
            })
        
        return results
    
    def evaluate(self, frame_ids, target_class='Car', iou_threshold=0.5):
        """
        Оценка пайплайна на наборе кадров.
        Возвращает: список (pred_3d, gt_3d) пар + сводная статистика.
        """
        from metrics import euclidean_3d_error, depth_error
        
        all_pairs = []
        all_errors_3d = []
        all_errors_depth = []
        
        for fid in frame_ids:
            preds = self.predict_3d(fid, target_classes=(target_class,))
            gts = [o for o in self.kitti.load_labels(fid) if o.type == target_class]
            
            # Простой матчинг по IoU
            from metrics import match_predictions_to_gt
            matches = match_predictions_to_gt(preds, gts, iou_threshold=iou_threshold)
            
            for pred, gt in matches:
                pred_loc = pred['location_3d']
                gt_loc = gt.location.tolist()
                
                # Если используем bottom-anchor, нужно скорректировать GT
                # (GT location.y = низ объекта, но GT хранится для центра bbox в KITTI?)
                # На sync-day согласовали — здесь оставляем как есть
                
                err_3d = euclidean_3d_error(pred_loc, gt_loc)
                err_d = depth_error(pred_loc, gt_loc)
                
                all_pairs.append((pred_loc, gt_loc))
                all_errors_3d.append(err_3d)
                all_errors_depth.append(err_d)
        
        return {
            'pairs': all_pairs,
            'mean_3d_error': float(np.mean(all_errors_3d)) if all_errors_3d else None,
            'median_3d_error': float(np.median(all_errors_3d)) if all_errors_3d else None,
            'mean_depth_error': float(np.mean(all_errors_depth)) if all_errors_depth else None,
            'localization_acc_2m': float(np.mean([e < 2.0 for e in all_errors_3d])) if all_errors_3d else None,
            'localization_acc_4m': float(np.mean([e < 4.0 for e in all_errors_3d])) if all_errors_3d else None,
            'n_matched': len(all_pairs),
        }
```

**Шаг 2.** Smoke-test: прогон пайплайна на 100 кадрах без калибровки:

```python
sys.path.append('/content/drive/MyDrive/3dcv-project/code/shared')
from kitti_loader import KITTILoader
from integrated_pipeline import IntegratedPipeline

kitti = KITTILoader('/content/kitti')
pipeline = IntegratedPipeline(
    kitti_loader=kitti,
    yolo_pred_dir=f'{ROOT}/results/track_a/yolov8_predictions',
    depth_pred_dir=f'{ROOT}/results/track_b/depth_pred_kitti',
    depth_calibration=None  # пока без калибровки
)

test_frames = kitti.frame_ids[:100]
metrics = pipeline.evaluate(test_frames, target_class='Car')
print(f'Без калибровки: mean 3D error = {metrics["mean_3d_error"]:.2f} м, '
      f'matched = {metrics["n_matched"]}')
```

⚠️ Без калибровки ошибка будет большая (~50–100 м), потому что depth относительный. Это ожидаемо.

### Артефакты к концу J2
- ✅ `code/shared/integrated_pipeline.py`
- ✅ `code/joint/01_integration_smoke_test.ipynb`
- ✅ Запись в `coordination.ipynb` об успешном smoke-test

---

## 3. J3 — Финальные эксперименты (дни 9–11)

### Дизайн экспериментов

Главная идея: **сравнить три стратегии калибровки относительной глубины** в метрическую.

| Эксперимент | Источник GT для калибровки scale+shift | Тест |
|-------------|----------------------------------------|------|
| **E1: real_only** | Sparse GT из KITTI (z из 3D-bbox в центрах GT) | KITTI val |
| **E2: synth_only** | Dense GT из VKITTI2 (полные depth maps) | KITTI val |
| **E3: mixed** | 50% real KITTI + 50% synth VKITTI2 | KITTI val |

**Логика:** в реальной жизни плотный GT-depth получить дорого (нужен LiDAR). Синтетика бесплатно даёт идеальный GT. Мы проверяем: можно ли калибровать модель глубины **только** на синтетике и при этом получить хорошее качество на реальных данных?

### Реализация (день 9)

**Шаг 1.** Скрипт калибровки `code/joint/calibration.py`:

```python
import numpy as np
import json
import os

def collect_calibration_pairs_kitti(kitti_loader, depth_pred_dir, frame_ids):
    """
    Собирает пары (relative_depth, gt_depth) из KITTI.
    Использует GT 3D-bbox: z-координата как глубина в центре 2D-bbox.
    """
    pairs = []
    for fid in frame_ids:
        depth_rel = np.load(os.path.join(depth_pred_dir, f'{fid}.npy'))
        for obj in kitti_loader.load_labels(fid):
            if obj.type != 'Car':
                continue
            x1, y1, x2, y2 = obj.bbox_2d
            u, v = int((x1+x2)/2), int((y1+y2)/2)
            if 0 <= v < depth_rel.shape[0] and 0 <= u < depth_rel.shape[1]:
                rel_val = depth_rel[v, u]
                gt_val = obj.depth  # z-координата из location
                pairs.append((rel_val, gt_val))
    return np.array(pairs)


def collect_calibration_pairs_vkitti(vkitti_loader, depth_pred_dir, n_samples=10000):
    """
    Собирает пары из VKITTI2 — рандомный сэмплинг пикселей с GT depth.
    """
    pairs = []
    for scene in vkitti_loader.SCENES:
        frames = vkitti_loader.list_frames(scene, 'clone')[:200]  # subset
        for fid in frames:
            gt = vkitti_loader.load_depth(scene, 'clone', fid)
            pred_path = f'{depth_pred_dir}/{scene}/{fid}.npy'
            if not os.path.exists(pred_path):
                continue
            pred = np.load(pred_path)
            
            if pred.shape != gt.shape:
                from PIL import Image
                pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0])))
            
            mask = (gt > 0.5) & (gt < 80)
            if mask.sum() < 100:
                continue
            
            # Рандомный сэмплинг пикселей
            ys, xs = np.where(mask)
            idx = np.random.choice(len(ys), size=min(50, len(ys)), replace=False)
            for i in idx:
                pairs.append((pred[ys[i], xs[i]], gt[ys[i], xs[i]]))
            
            if len(pairs) > n_samples:
                return np.array(pairs[:n_samples])
    return np.array(pairs)


def fit_scale_shift(pairs):
    """Least squares: gt = scale * relative + shift"""
    rel = pairs[:, 0]
    gt = pairs[:, 1]
    A = np.stack([rel, np.ones_like(rel)], axis=1)
    scale, shift = np.linalg.lstsq(A, gt, rcond=None)[0]
    return float(scale), float(shift)
```

**Шаг 2.** Прогон трёх экспериментов:

```python
# Train/val split — берём первые 80% как train (для калибровки), 20% — val (для теста)
all_frames = sorted(kitti.frame_ids)
n_train = int(0.8 * len(all_frames))
train_frames = all_frames[:n_train]
val_frames = all_frames[n_train:]

print(f'Train: {len(train_frames)} кадров, Val: {len(val_frames)} кадров\n')

# E1: калибровка по KITTI train
print('=== E1: real_only ===')
pairs_real = collect_calibration_pairs_kitti(kitti, 
    f'{ROOT}/results/track_b/depth_pred_kitti', train_frames)
scale_e1, shift_e1 = fit_scale_shift(pairs_real)
print(f'Scale={scale_e1:.3f}, Shift={shift_e1:.3f}, N pairs={len(pairs_real)}')

pipeline_e1 = IntegratedPipeline(kitti, 
    f'{ROOT}/results/track_a/yolov8_predictions',
    f'{ROOT}/results/track_b/depth_pred_kitti',
    depth_calibration=(scale_e1, shift_e1))
metrics_e1 = pipeline_e1.evaluate(val_frames)
print(f'Mean 3D error: {metrics_e1["mean_3d_error"]:.2f} м')
print(f'Localization acc @ 2m: {metrics_e1["localization_acc_2m"]:.3f}')
print(f'Localization acc @ 4m: {metrics_e1["localization_acc_4m"]:.3f}\n')

# E2: калибровка по VKITTI2
print('=== E2: synth_only ===')
from vkitti_loader import VKITTI2Loader
vkitti = VKITTI2Loader('/content/vkitti2')
pairs_synth = collect_calibration_pairs_vkitti(vkitti,
    f'{ROOT}/results/track_b/depth_pred_vkitti', n_samples=10000)
scale_e2, shift_e2 = fit_scale_shift(pairs_synth)
print(f'Scale={scale_e2:.3f}, Shift={shift_e2:.3f}, N pairs={len(pairs_synth)}')

pipeline_e2 = IntegratedPipeline(kitti,
    f'{ROOT}/results/track_a/yolov8_predictions',
    f'{ROOT}/results/track_b/depth_pred_kitti',
    depth_calibration=(scale_e2, shift_e2))
metrics_e2 = pipeline_e2.evaluate(val_frames)
print(f'Mean 3D error: {metrics_e2["mean_3d_error"]:.2f} м')
print(f'Localization acc @ 2m: {metrics_e2["localization_acc_2m"]:.3f}\n')

# E3: mixed (50/50 по числу пар)
print('=== E3: mixed_50_50 ===')
n_mix = min(len(pairs_real), len(pairs_synth)) // 2
mixed_pairs = np.concatenate([pairs_real[:n_mix], pairs_synth[:n_mix]])
scale_e3, shift_e3 = fit_scale_shift(mixed_pairs)
print(f'Scale={scale_e3:.3f}, Shift={shift_e3:.3f}, N pairs={len(mixed_pairs)}')

pipeline_e3 = IntegratedPipeline(kitti,
    f'{ROOT}/results/track_a/yolov8_predictions',
    f'{ROOT}/results/track_b/depth_pred_kitti',
    depth_calibration=(scale_e3, shift_e3))
metrics_e3 = pipeline_e3.evaluate(val_frames)
print(f'Mean 3D error: {metrics_e3["mean_3d_error"]:.2f} м')
print(f'Localization acc @ 2m: {metrics_e3["localization_acc_2m"]:.3f}\n')

# Сохраняем результаты
import pandas as pd
df = pd.DataFrame([
    {'experiment': 'E1_real_only', **metrics_e1, 'scale': scale_e1, 'shift': shift_e1},
    {'experiment': 'E2_synth_only', **metrics_e2, 'scale': scale_e2, 'shift': shift_e2},
    {'experiment': 'E3_mixed', **metrics_e3, 'scale': scale_e3, 'shift': shift_e3},
])
df = df.drop(columns=['pairs'])  # не сохраняем сырые пары
df.to_csv(f'{ROOT}/results/joint/final_metrics_table.csv', index=False)
print('✅ Финальные метрики сохранены')
print(df.to_string())
```

### Дополнительные эксперименты (если есть время — день 10)

- **E4: разные пропорции mixed** — 25/75, 75/25 — графика «доля синтетики vs точность»
- **E5: разные погодные вариации VKITTI2** — clone vs fog vs rain — устойчивость калибровки
- **E6: ablation** — отдельно вклад правильной 2D-детекции и правильной глубины

---

## 4. J4 — Сводные таблицы и графики (день 11)

### Что нужно

**Таблица 1.** Сводка по 3D-метрикам:

| Эксперимент | Mean 3D Err (м) | Median 3D Err (м) | Mean Depth Err (м) | LocAcc @ 2m | LocAcc @ 4m |
|-------------|-----------------|-------------------|--------------------|-------------|-------------|
| E1: real_only | _x.xx_ | _x.xx_ | _x.xx_ | _0.xxx_ | _0.xxx_ |
| E2: synth_only | _x.xx_ | _x.xx_ | _x.xx_ | _0.xxx_ | _0.xxx_ |
| E3: mixed | _x.xx_ | _x.xx_ | _x.xx_ | _0.xxx_ | _0.xxx_ |

**График 1.** Распределение 3D-ошибок по экспериментам (boxplot).

**График 2.** Доля синтетики vs точность (если делали E4).

**График 3.** Bird's-eye view: предсказанные vs GT положения для одного кадра, по каждому эксперименту.

**Визуализация 1.** Side-by-side: RGB + 2D bbox + depth heatmap + BEV для одного кадра, на каждый эксперимент.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot ошибок по экспериментам
fig, ax = plt.subplots(figsize=(8, 5))
data_for_plot = []
labels = []
for exp_name, pipeline in [('E1: real_only', pipeline_e1), 
                            ('E2: synth_only', pipeline_e2), 
                            ('E3: mixed', pipeline_e3)]:
    metrics = pipeline.evaluate(val_frames[:200])  # subset для скорости
    errors = [np.linalg.norm(np.array(p) - np.array(g)) 
              for p, g in metrics['pairs']]
    data_for_plot.append(errors)
    labels.append(exp_name)

ax.boxplot(data_for_plot, labels=labels, showfliers=False)
ax.set_ylabel('3D-ошибка (м)')
ax.set_title('Сравнение стратегий калибровки')
ax.grid(alpha=0.3)
plt.savefig(f'{ROOT}/results/joint/figures/fig_boxplot_errors.png', 
            dpi=150, bbox_inches='tight')
```

### Артефакты к концу J4
- ✅ `results/joint/final_metrics_table.csv`
- ✅ `results/joint/figures/fig_boxplot_errors.png`
- ✅ `results/joint/figures/fig_bev_comparison.png`
- ✅ `results/joint/figures/fig_qualitative_examples.png`
- ✅ Минимум 2 таблицы и 4 графика — готовы к статье

---

## 5. J5 — Написание статьи (дни 12–13)

### Распределение разделов (из основного плана)

| Раздел | Кто пишет | Дедлайн |
|--------|-----------|---------|
| 1. Аннотация | Совместно | день 13 |
| 2. Введение | Наргиз | день 12 |
| 3. Обзор литературы | Лена | день 12 |
| 4. Используемые датасеты | По 0.5 стр каждая | день 12 |
| 5. Метод | Совместно | день 12 |
| 6. Эксперименты | Совместно | день 12–13 |
| 7. Результаты | Совместно | день 13 |
| 8. Обсуждение | Лена | день 13 |
| 9. Заключение | Наргиз | день 13 |

**Файл:** `paper/draft.md` — каждая работает в своих секциях. Скелет уже создан.

### Workflow

1. **День 12:** каждая пишет свои секции в `paper/draft.md`. В `coordination.ipynb` ставите галочки по мере готовности.
2. **День 13 утро:** перекрёстная вычитка — каждая читает разделы другой и пишет замечания в `coordination.ipynb`.
3. **День 13 день:** правки + написание совместных разделов (метод, эксперименты, результаты).
4. **День 14:** финальная вычитка и форматирование.

### Что обязательно должно быть в статье (для оценки 5)

- ✅ **3+ архитектуры/модели:** YOLOv8 + Depth Anything v2 + Lift-to-3D алгоритм (это считается)
- ✅ **Собственный аугментированный датасет** для инференса (KITTI + VKITTI2 объединены)
- ✅ **Глубокий сравнительный анализ:** минимум 3 эксперимента (E1/E2/E3) + анализ domain gap
- ✅ **Минимум 3 таблицы и 4 графика**
- ✅ **Качественная визуализация** на конкретных примерах
- ✅ **Корректный список литературы** (RuСCS-стиль или формат журнала)

---

## 6. J6 — Финальная вычитка и сборка (день 14)

### Чек-лист

```
[ ] Все таблицы имеют номера и подписи
[ ] Все графики имеют подписи и легенды
[ ] Все формулы пронумерованы
[ ] Список литературы корректен и полон (минимум 15 ссылок)
[ ] Аннотация на русском и английском
[ ] Ключевые слова (5-7 штук)
[ ] Сведения об авторах
[ ] Указание на МИСИС, дисциплину, научного руководителя
[ ] Конвертация Markdown → Word (через pandoc или Google Docs)
[ ] Финальное оформление по требованиям выбранного журнала
[ ] Проверка на плагиат (если требуется журналом)
```

### Конвертация в Word

Самый простой способ — открыть `paper/draft.md` в **Google Docs** через расширение «Docs to Markdown» или через `pandoc`:

```bash
!apt install pandoc -y
!pandoc -f markdown -t docx paper/draft.md -o paper/article_final.docx
```

Затем открыть в Word и довести оформление вручную.

### Подача статьи

Согласовать с преподавателем:
- Какой журнал (РИНЦ-сборник МИСИС или внешний)
- Срок подачи
- Требования к оформлению (шрифт, поля, формат ссылок)

---

## 7. Критерии успеха проекта

### Технические
- ✅ Полный пайплайн работает end-to-end на ≥1000 кадрах KITTI val
- ✅ Mean 3D error для класса Car < 5 метров (на mid-range)
- ✅ Localization Acc @ 2m > 0.4

### Исследовательские
- ✅ Чёткий ответ на основной вопрос исследования: насколько синтетика помогает калибровке
- ✅ Все три эксперимента (E1, E2, E3) выполнены и сравнены
- ✅ Анализ domain gap (Лена) интегрирован в обсуждение

### Академические
- ✅ Статья оформлена по требованиям журнала
- ✅ Минимум 8 страниц, максимум 15
- ✅ Презентация результатов готова (если требует препод)
- ✅ Получен фидбек от Садекова Р.Н.

---

## 8. План Б на случай провала экспериментов

| Проблема | План Б |
|----------|--------|
| Все три эксперимента дают слишком большую ошибку (>20м) | Это тоже результат — анализируем причины (точность Depth Anything, геометрия Lift-to-3D), пишем как «честный негативный результат» с обсуждением |
| E2 (synth_only) даёт сильно хуже E1 — domain gap огромный | Это интересный результат! Демонстрирует, что simple sim-to-real не работает, нужны более сложные методы DA |
| Нет времени на E4/E5/E6 | Делаем минимум E1+E2+E3, остальное в «направления дальнейших исследований» |
| Не успеваем со статьёй | Сокращаем секции до минимума, главное — таблица результатов и анализ |

---

## 9. Roadmap последних 8 дней (после Sync-day)

| День | Задача | Кто |
|------|--------|-----|
| 7 | Sync-day, проверка артефактов | Обе |
| 8 | J2: интеграция, smoke-test | Наргиз (Лена доступна для вопросов) |
| 9 | J3: эксперименты E1, E2, E3 | Наргиз |
| 10 | Дополнительные эксперименты + начало визуализаций | Обе |
| 11 | J4: финальные таблицы и графики | Обе |
| 12 | J5: написание основных разделов | Обе |
| 13 | J5: совместные разделы + перекрёстная вычитка | Обе |
| 14 | J6: финальная сборка, конвертация в Word | Обе |

---

## 10. FAQ

**В:** Что если калибровка scale+shift даёт странные значения (отрицательный scale)?
**О:** Скорее всего проблема в направлении: Depth Anything может выдавать «inverse depth» (1/Z) вместо просто Z. Проверь корреляцию pred_depth и gt_depth в одной точке — если они анти-коррелируют, нужно инвертировать.

**В:** Точка-якорь для lift — bottom или center?
**О:** Решается на sync-day. Для машин на дороге — bottom (колёса). Тогда GT location нужно сравнивать с z=0 ground plane или корректировать на height/2.

**В:** Что если синтетических данных слишком много для калибровки и она «забивает» реальные?
**О:** Используем равные веса (50/50 пар), а не равное число изображений. Это уже учтено в коде E3.

**В:** YOLOv8 пропустил много объектов — это убьёт результаты?
**О:** Это влияет только на recall. На метрики 3D-локализации сматченных пар не влияет — они считаются только по успешным детекциям. В статье отмечаем как «ограничение текущего пайплайна, требующее улучшения 2D-детектора».

**В:** Я застряла на интеграции, что делать?
**О:** Пиши Claude с конкретной ошибкой. Параллельно — фиксируй блокер в `coordination.ipynb`, чтобы Лена тоже видела. Если блокер критический — переключаемся на план Б из раздела 8.

---

*Финальная фаза самая интенсивная — много мелких деталей, но техническая часть уже сделана. Главное — не пытаться доделать что-то идеально в ущерб срокам. Лучше работающий минимум, чем недоделанный максимум.*
