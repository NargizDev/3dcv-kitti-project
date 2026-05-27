## Синтетические данные и моно-оценка глубины

Полное описание проекта — в файле `Project_Plan_3D_CV.md` в общей папке Google Drive. **Прочитай его первым делом** — там есть архитектура пайплайна, цели исследования и общий roadmap.

**Кратко суть проекта:**
Мы строим пайплайн «Lift-to-3D», который определяет 3D-координаты объектов в дорожных сценах, комбинируя:
- **2D-детектор** (YOLOv8) — определяет, где на изображении объект
- **Моно-depth** (Depth Anything v2) — определяет расстояние до каждого пикселя
- **Lift-to-3D алгоритм** — переводит 2D-детекции в 3D через калибровку камеры

**Распределение:**
- Я (Наргиз) — реальные данные KITTI + 2D-детекция (YOLOv8) + Lift-to-3D алгоритм + интеграция
- Ты — синтетические данные Virtual KITTI 2 + Depth Anything v2 + анализ domain gap

**Финальная задача:** сравнить точность 3D-локализации при разных стратегиях смешивания реальных и синтетических данных и опубликовать статью.

---

## 1. Подготовка среды (день 1)

### 1.1. Доступ к общей папке Google Drive

Я расшарю тебе папку `3dcv-project` в Google Drive. Структура будет такая:

```
3dcv-project/
├── data/
│   ├── kitti/           ← мои данные 
│   └── vkitti2/         ← твои данные
├── code/
│   ├── track_a/         ← мои ноутбуки
│   ├── track_b/         ← ТВОИ ноутбуки
│   └── shared/          ← общие модули
├── checkpoints/         ← веса моделей
├── results/
│   ├── track_a/
│   ├── track_b/         ← ТВОИ результаты
│   └── joint/
├── paper/               ← статья
└── coordination.ipynb   ← наш общий ноутбук для статуса и вопросов
```

**Правило:** работай только в `code/track_b/`, `data/vkitti2/`, `results/track_b/`. Совместная работа — только в `coordination.ipynb` и потом в `paper/`.

### 1.2. Google Colab

Все твои ноутбуки запускай в **Google Colab** с GPU T4 (Runtime → Change runtime type → T4 GPU). Каждый ноутбук сохраняй в `code/track_b/` в Google Drive.

### 1.3. Скачать Virtual KITTI 2

Источник: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

Скачай **только эти два архива** (всё остальное не нужно):
- `vkitti_2.0.3_rgb.tar` — RGB-изображения (~17 GB)
- `vkitti_2.0.3_depth.tar` — карты глубины (~2 GB)

Загрузи их в Google Drive в папку `data/vkitti2/`.

⚠️ **Не распаковывай локально** — заливай как есть, распакуем уже в Colab.

---

## 2. Задача B1 — Подготовка Virtual KITTI 2 (дни 1–2)

### Цель
Распаковать Virtual KITTI 2, разобраться со структурой, написать модуль для загрузки изображений и depth maps.

### Что нужно сделать

**Шаг 1.** Создай ноутбук `code/track_b/01_vkitti_setup.ipynb`. Распакуй архивы:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, tarfile
VKITTI_DRIVE = '/content/drive/MyDrive/3dcv-project/data/vkitti2'
VKITTI_LOCAL = '/content/vkitti2'
os.makedirs(VKITTI_LOCAL, exist_ok=True)

# Распаковка в локальное хранилище Colab (быстрее, чем работать с Drive напрямую)
for archive in ['vkitti_2.0.3_rgb.tar', 'vkitti_2.0.3_depth.tar']:
    archive_path = f'{VKITTI_DRIVE}/{archive}'
    if os.path.exists(archive_path):
        print(f'📦 Распаковываю {archive}...')
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(VKITTI_LOCAL)
        print(f'   ✅ Готово')
    else:
        print(f'   ❌ Не найден: {archive}')
```

**Шаг 2.** Изучи структуру VKITTI2. После распаковки посмотри:

```python
!find {VKITTI_LOCAL} -maxdepth 4 -type d | head -30
```

Структура VKITTI2 такая:
```
vkitti2/
├── Scene01/  (одна из 5 сцен)
│   ├── clone/        ← базовая вариация (без эффектов)
│   ├── fog/          ← туман
│   ├── morning/      ← утро
│   ├── overcast/     ← пасмурно
│   ├── rain/         ← дождь
│   └── sunset/       ← закат
├── Scene02/
├── Scene06/
├── Scene18/
└── Scene20/
```

В каждой папке вариации:
```
clone/
├── frames/
│   ├── rgb/Camera_0/        ← RGB-изображения (.jpg)
│   ├── depth/Camera_0/      ← depth maps (.png, 16-bit)
│   └── ...
└── ...
```

**Шаг 3.** Напиши модуль `code/shared/vkitti_loader.py` с классом-загрузчиком:

```python
import os
import numpy as np
from PIL import Image

class VKITTI2Loader:
    """Загрузчик Virtual KITTI 2"""
    
    SCENES = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    VARIATIONS = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
    
    def __init__(self, root_dir):
        """
        root_dir: путь до распакованной папки vkitti2/
        """
        self.root = root_dir
    
    def list_frames(self, scene='Scene01', variation='clone'):
        """Список всех frame_id для указанной сцены и вариации"""
        rgb_dir = os.path.join(self.root, scene, variation, 'frames/rgb/Camera_0')
        if not os.path.exists(rgb_dir):
            return []
        return sorted([f.split('_')[1].split('.')[0] 
                      for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    
    def load_rgb(self, scene, variation, frame_id):
        """Загрузить RGB-изображение"""
        path = os.path.join(self.root, scene, variation, 
                           f'frames/rgb/Camera_0/rgb_{frame_id}.jpg')
        return np.array(Image.open(path))
    
    def load_depth(self, scene, variation, frame_id):
        """
        Загрузить depth map в метрах.
        VKITTI2 хранит глубину в сантиметрах в 16-bit PNG.
        """
        path = os.path.join(self.root, scene, variation,
                           f'frames/depth/Camera_0/depth_{frame_id}.png')
        depth_cm = np.array(Image.open(path))
        depth_m = depth_cm.astype(np.float32) / 100.0  # см → м
        # Глубина 65535 см = "бесконечность", обнуляем
        depth_m[depth_cm == 65535] = 0
        return depth_m
```

**Шаг 4.** Тест: загрузи одно изображение и его depth, визуализируй:

```python
import matplotlib.pyplot as plt

loader = VKITTI2Loader('/content/vkitti2')
frames = loader.list_frames('Scene01', 'clone')
print(f'Найдено {len(frames)} кадров в Scene01/clone')

# Визуализация
rgb = loader.load_rgb('Scene01', 'clone', frames[0])
depth = loader.load_depth('Scene01', 'clone', frames[0])

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].imshow(rgb)
axes[0].set_title('RGB (Virtual KITTI 2)')
axes[0].axis('off')

im = axes[1].imshow(depth, cmap='plasma', vmin=0, vmax=80)
axes[1].set_title('GT Depth (м)')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.04)
plt.savefig('/content/drive/MyDrive/3dcv-project/results/track_b/B1_vkitti_sample.png', 
            dpi=150, bbox_inches='tight')
plt.show()
```

### Артефакты к концу B1
- ✅ `code/shared/vkitti_loader.py` — модуль с классом `VKITTI2Loader`
- ✅ `code/track_b/01_vkitti_setup.ipynb` — ноутбук с распаковкой и тестами
- ✅ `results/track_b/B1_vkitti_sample.png` — визуализация одного образца
- ✅ Запись в `coordination.ipynb`: «B1 готово, найдено N кадров по 5 сценам»

### Критерий приёмки
- Модуль `VKITTI2Loader` корректно загружает RGB и depth по любой сцене/вариации
- Глубина в метрах (значения от 0 до ~80), а не в сырых единицах
- Визуализация показывает разумные цвета depth-карты (близкие объекты — один цвет, далёкие — другой)

---

## 3. Задача B2 — Инференс Depth Anything v2 (дни 2–4)

### Цель
Запустить pre-trained модель Depth Anything v2 на изображениях из KITTI и Virtual KITTI 2, сохранить предсказанные depth-карты.

### Что нужно сделать

**Шаг 1.** Создай ноутбук `code/track_b/02_depth_anything_inference.ipynb`.

**Шаг 2.** Установка и загрузка модели:

```python
!pip install -q transformers torch torchvision

from transformers import pipeline
from PIL import Image
import numpy as np
import torch

device = 0 if torch.cuda.is_available() else -1
print(f'Используем: {"GPU" if device == 0 else "CPU"}')

# Загружаем Small-версию (быстрее, помещается на T4)
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=device
)
print('✅ Depth Anything v2 загружен')
```

⚠️ **Важно:** Depth Anything v2 предсказывает **относительную** глубину (значения 0–1), не метрическую. Для метрики нам нужно её откалибровать. Подробности в задаче B3.

**Шаг 3.** Функция инференса с сохранением:

```python
import os

def predict_depth(image_path, save_path=None):
    """
    Предсказать depth для одного изображения.
    Возвращает np.array (H, W) с относительной глубиной 0–1.
    """
    image = Image.open(image_path).convert('RGB')
    result = pipe(image)
    depth = np.array(result['depth'])  # (H, W), относительная
    
    # Нормализуем в 0–1
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, depth_normalized.astype(np.float32))
    
    return depth_normalized
```

**Шаг 4.** Инференс на VKITTI2 (5 сцен × clone вариация = ~2100 кадров):

```python
from tqdm.notebook import tqdm

VKITTI_RGB = '/content/vkitti2'
DEPTH_PRED_DIR = '/content/drive/MyDrive/3dcv-project/results/track_b/depth_pred_vkitti'

loader = VKITTI2Loader(VKITTI_RGB)

for scene in loader.SCENES:
    frames = loader.list_frames(scene, 'clone')
    print(f'\n{scene}: {len(frames)} кадров')
    
    for frame_id in tqdm(frames, desc=scene):
        rgb_path = f'{VKITTI_RGB}/{scene}/clone/frames/rgb/Camera_0/rgb_{frame_id}.jpg'
        save_path = f'{DEPTH_PRED_DIR}/{scene}/{frame_id}.npy'
        if not os.path.exists(save_path):
            predict_depth(rgb_path, save_path)

print('\n✅ Инференс на VKITTI2 завершён')
```

**Шаг 5.** Инференс на KITTI. Я (Наргиз) загружу KITTI в общую папку. Путь будет `/content/drive/MyDrive/3dcv-project/data/kitti/training/image_2/`.

```python
KITTI_RGB = '/content/drive/MyDrive/3dcv-project/data/kitti/training/image_2'
DEPTH_PRED_KITTI = '/content/drive/MyDrive/3dcv-project/results/track_b/depth_pred_kitti'

# Берём первые 1000 изображений (можно больше при наличии времени)
images = sorted(os.listdir(KITTI_RGB))[:1000]

for img_name in tqdm(images):
    rgb_path = f'{KITTI_RGB}/{img_name}'
    save_path = f'{DEPTH_PRED_KITTI}/{img_name.replace(".png", ".npy")}'
    if not os.path.exists(save_path):
        predict_depth(rgb_path, save_path)

print('✅ Инференс на KITTI завершён')
```

⚠️ **Если Colab отвалится по таймауту** — это нормально, GPU-сессия ограничена ~12 часами. Перезапусти ноутбук, цикл сам пропустит уже обработанные файлы (есть проверка `os.path.exists`).

### Артефакты к концу B2
- ✅ `code/track_b/02_depth_anything_inference.ipynb` — ноутбук
- ✅ `results/track_b/depth_pred_vkitti/` — depth maps для VKITTI2 (.npy)
- ✅ `results/track_b/depth_pred_kitti/` — depth maps для KITTI (.npy)
- ✅ Запись в `coordination.ipynb` с количеством обработанных изображений

### Критерий приёмки
- Минимум 1000 изображений KITTI и 2000 изображений VKITTI2 имеют сохранённые depth-предсказания
- Все .npy файлы можно загрузить через `np.load()` и они имеют форму (H, W) с типом float32

---

## 4. Задача B3 — Метрики качества глубины (день 5)

### Цель
Откалибровать предсказанную (относительную) глубину в метрическую и посчитать стандартные метрики качества: AbsRel, RMSE, δ < 1.25.

### Что нужно сделать

**Шаг 1.** Создай `code/track_b/03_depth_metrics.ipynb`.

**Шаг 2.** Калибровка scale & shift. Поскольку Depth Anything предсказывает относительную глубину, для метрической оценки нужно подобрать масштаб и сдвиг через **least squares fit** к GT:

```python
def align_depth_lstsq(pred, gt, mask):
    """
    Подбирает scale и shift через least squares: gt ≈ scale * pred + shift
    Возвращает aligned predicted depth в метрах.
    """
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
    
    return scale * pred + shift, scale, shift
```

**Шаг 3.** Метрики:

```python
def compute_depth_metrics(pred, gt, max_depth=80.0):
    """
    Стандартные метрики оценки глубины (KITTI eval protocol)
    """
    mask = (gt > 0.1) & (gt < max_depth)
    pred = pred[mask]
    gt = gt[mask]
    
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean((pred - gt) ** 2 / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))
    
    thresh = np.maximum(pred / gt, gt / pred)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
    
    return {
        'AbsRel': abs_rel,
        'SqRel': sq_rel,
        'RMSE': rmse,
        'RMSE_log': rmse_log,
        'δ < 1.25': delta1,
        'δ < 1.25²': delta2,
        'δ < 1.25³': delta3,
    }
```

**Шаг 4.** Подсчёт метрик на VKITTI2 (там есть GT depth):

```python
import pandas as pd

results = []
for scene in loader.SCENES:
    frames = loader.list_frames(scene, 'clone')
    for frame_id in tqdm(frames, desc=scene):
        gt_depth = loader.load_depth(scene, 'clone', frame_id)
        pred_path = f'{DEPTH_PRED_DIR}/{scene}/{frame_id}.npy'
        if not os.path.exists(pred_path):
            continue
        pred_relative = np.load(pred_path)
        
        # Если предсказание и GT разного размера — ресайзим
        if pred_relative.shape != gt_depth.shape:
            from PIL import Image
            pred_relative = np.array(Image.fromarray(pred_relative).resize(
                (gt_depth.shape[1], gt_depth.shape[0])))
        
        # Калибровка
        mask = (gt_depth > 0.1) & (gt_depth < 80)
        if mask.sum() < 100:
            continue
        pred_metric, _, _ = align_depth_lstsq(pred_relative, gt_depth, mask)
        
        # Метрики
        m = compute_depth_metrics(pred_metric, gt_depth)
        m['scene'] = scene
        m['frame'] = frame_id
        results.append(m)

df = pd.DataFrame(results)
df.to_csv('/content/drive/MyDrive/3dcv-project/results/track_b/B3_depth_metrics_vkitti.csv', 
          index=False)

print('\n📊 Сводка по VKITTI2:')
print(df[['AbsRel', 'RMSE', 'δ < 1.25']].mean())
```

**Шаг 5.** Для KITTI GT depth получается из Velodyne point clouds — мы это пропустим. Вместо этого используем подход «качественной оценки»: визуальное сравнение predicted depth с RGB. Я напишу тебе позже, когда подготовлю калибровочные данные.

### Артефакты к концу B3
- ✅ `code/track_b/03_depth_metrics.ipynb`
- ✅ `results/track_b/B3_depth_metrics_vkitti.csv` — метрики по каждому кадру
- ✅ Сводная таблица средних метрик по сценам

### Критерий приёмки
- Средние метрики на VKITTI2 примерно: AbsRel ~ 0.10–0.15, δ < 1.25 ~ 0.85–0.95 (это норма для pre-trained Depth Anything v2 без обучения на VKITTI2)
- CSV корректно открывается в pandas

---

## 5. Задача B4 — Стратегии аугментации (дни 5–6)

### Цель
Подготовить три «стратегии» данных, которые мы потом используем в финальном эксперименте: real-only, synth-only, mixed (50/50).

### Что нужно сделать

**Шаг 1.** Создай `code/track_b/04_augmentation_configs.ipynb`.

**Шаг 2.** Написать конфиги:

```python
# Стратегия 1: только реальные данные (KITTI train)
config_real_only = {
    'name': 'real_only',
    'train_images': 'kitti/training/image_2/',  # ~7000 изображений
    'depth_source': 'kitti_predicted',
    'description': 'Используем только реальные изображения KITTI с предсказанной глубиной'
}

# Стратегия 2: только синтетика (VKITTI2 clone)
config_synth_only = {
    'name': 'synth_only',
    'train_images': 'vkitti2/Scene01-20/clone/',  # ~10000 изображений
    'depth_source': 'vkitti_gt',  # есть точный GT
    'description': 'Только VKITTI2 с истинной глубиной'
}

# Стратегия 3: смешанная (50/50)
config_mixed = {
    'name': 'mixed_50_50',
    'real_ratio': 0.5,
    'synth_ratio': 0.5,
    'real_source': 'kitti/training/image_2/',
    'synth_source': 'vkitti2/Scene01-20/clone/',
    'description': 'Поровну реальных и синтетических данных'
}
```

**Шаг 3.** Для каждой стратегии создай **списки путей к файлам**, которые мы используем как trainset:

```python
import json

def build_filelist(config, output_path):
    files = []
    if config['name'] == 'real_only':
        kitti_dir = '/content/drive/MyDrive/3dcv-project/data/kitti/training/image_2'
        files = [f'{kitti_dir}/{f}' for f in sorted(os.listdir(kitti_dir))]
    
    elif config['name'] == 'synth_only':
        for scene in loader.SCENES:
            for fid in loader.list_frames(scene, 'clone'):
                files.append(f'/content/vkitti2/{scene}/clone/frames/rgb/Camera_0/rgb_{fid}.jpg')
    
    elif config['name'] == 'mixed_50_50':
        # 50% реальных + 50% синтетических
        kitti_dir = '/content/drive/MyDrive/3dcv-project/data/kitti/training/image_2'
        real_files = sorted(os.listdir(kitti_dir))[:3500]
        synth_files = []
        for scene in loader.SCENES:
            for fid in loader.list_frames(scene, 'clone')[:700]:
                synth_files.append(f'/content/vkitti2/{scene}/clone/frames/rgb/Camera_0/rgb_{fid}.jpg')
        files = [f'{kitti_dir}/{f}' for f in real_files] + synth_files
    
    with open(output_path, 'w') as f:
        json.dump({'config': config, 'files': files, 'count': len(files)}, f, indent=2)
    print(f'✅ {config["name"]}: {len(files)} файлов → {output_path}')

# Сохраняем конфиги
CONFIG_DIR = '/content/drive/MyDrive/3dcv-project/results/track_b/configs'
os.makedirs(CONFIG_DIR, exist_ok=True)

for cfg in [config_real_only, config_synth_only, config_mixed]:
    build_filelist(cfg, f'{CONFIG_DIR}/{cfg["name"]}.json')
```

### Артефакты к концу B4
- ✅ `results/track_b/configs/real_only.json`
- ✅ `results/track_b/configs/synth_only.json`
- ✅ `results/track_b/configs/mixed_50_50.json`

### Критерий приёмки
- Каждый JSON содержит поля `config`, `files`, `count`
- Все пути в `files` существуют (проверь рандомным сэмплингом 10 файлов)

---

## 6. Задача B5 — Анализ domain gap (день 6)

### Цель
Показать численно и визуально, насколько Depth Anything v2 деградирует на синтетике по сравнению с реальными данными (или наоборот).

### Что нужно сделать

**Шаг 1.** Создай `code/track_b/05_domain_gap_analysis.ipynb`.

**Шаг 2.** Сравни средние метрики глубины на KITTI vs VKITTI2:

```python
import pandas as pd
import seaborn as sns

df_vkitti = pd.read_csv('/content/drive/MyDrive/3dcv-project/results/track_b/B3_depth_metrics_vkitti.csv')

# Группировка по сценам
agg = df_vkitti.groupby('scene')[['AbsRel', 'RMSE', 'δ < 1.25']].mean()
print(agg)

# Boxplot AbsRel по сценам
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_vkitti, x='scene', y='AbsRel')
plt.title('AbsRel по сценам Virtual KITTI 2')
plt.ylabel('Absolute Relative Error')
plt.savefig('/content/drive/MyDrive/3dcv-project/results/track_b/B5_absrel_by_scene.png',
            dpi=150, bbox_inches='tight')
plt.show()
```

**Шаг 3.** Визуальный failure case analysis. Найди 5 кадров с наибольшей AbsRel и покажи:
- RGB
- Predicted depth
- GT depth
- Карта ошибки (|pred - gt|)

```python
worst_frames = df_vkitti.nlargest(5, 'AbsRel')[['scene', 'frame', 'AbsRel']]
print('5 худших кадров:')
print(worst_frames)

fig, axes = plt.subplots(5, 4, figsize=(20, 20))
for i, row in enumerate(worst_frames.itertuples()):
    rgb = loader.load_rgb(row.scene, 'clone', row.frame)
    gt = loader.load_depth(row.scene, 'clone', row.frame)
    pred_rel = np.load(f'{DEPTH_PRED_DIR}/{row.scene}/{row.frame}.npy')
    
    if pred_rel.shape != gt.shape:
        from PIL import Image
        pred_rel = np.array(Image.fromarray(pred_rel).resize((gt.shape[1], gt.shape[0])))
    
    mask = (gt > 0.1) & (gt < 80)
    pred_metric, _, _ = align_depth_lstsq(pred_rel, gt, mask)
    error = np.abs(pred_metric - gt) * mask
    
    axes[i, 0].imshow(rgb); axes[i, 0].set_title(f'RGB ({row.scene}/{row.frame})')
    axes[i, 1].imshow(gt, cmap='plasma', vmin=0, vmax=80); axes[i, 1].set_title('GT depth')
    axes[i, 2].imshow(pred_metric, cmap='plasma', vmin=0, vmax=80); axes[i, 2].set_title('Pred depth')
    axes[i, 3].imshow(error, cmap='hot', vmin=0, vmax=20); axes[i, 3].set_title(f'Error (AbsRel={row.AbsRel:.3f})')
    for j in range(4):
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/3dcv-project/results/track_b/B5_failure_cases.png',
            dpi=150, bbox_inches='tight')
plt.show()
```

**Шаг 4.** Краткие выводы (1–2 абзаца) запиши в `coordination.ipynb`:
- На каких сценах метрики хуже всего и почему (предположения)
- Какие визуальные паттерны характерны для ошибок (например, далёкие объекты, тени, прозрачные поверхности)
- Гипотеза о domain gap: модель обучена на mix реальных датасетов, поэтому на VKITTI2 могут быть систематические смещения

### Артефакты к концу B5
- ✅ `code/track_b/05_domain_gap_analysis.ipynb`
- ✅ `results/track_b/B5_absrel_by_scene.png`
- ✅ `results/track_b/B5_failure_cases.png`
- ✅ Краткий текстовый анализ в `coordination.ipynb`

---

## 7. День 7 — Sync-day 

К этому моменту у тебя должны быть готовы все артефакты B1–B5. Я буду интегрировать твой компонент с моим.

**Что мне нужно от тебя на sync-day:**
1. Все .npy файлы depth-предсказаний в `results/track_b/depth_pred_*/`
2. Загрузчик `vkitti_loader.py` в `code/shared/`
3. Метрики в CSV
4. Конфиги стратегий в JSON
5. **15-минутный созвон (если надо) или подробное сообщение в `coordination.ipynb`**, где ты пройдёшься по всем артефактам

После sync-day мы вместе будем работать в `code/joint/` над финальными экспериментами и статьёй.

---

## 8. Чек-лист готовности (отмечай галочками в `coordination.ipynb`)

```
[ ] B1.1 — VKITTI2 распакован, структура изучена
[ ] B1.2 — Класс VKITTI2Loader написан и протестирован
[ ] B1.3 — Визуализация образца сохранена

[ ] B2.1 — Depth Anything v2 запущен в Colab
[ ] B2.2 — Инференс на VKITTI2 (1000+ кадров)
[ ] B2.3 — Инференс на KITTI (1000+ кадров)

[ ] B3.1 — Функции align_depth и compute_metrics реализованы
[ ] B3.2 — CSV с метриками по VKITTI2 сохранён
[ ] B3.3 — Средние метрики разумны (AbsRel < 0.2, δ < 1.25 > 0.8)

[ ] B4.1 — Три JSON-конфига созданы (real/synth/mixed)
[ ] B4.2 — Все пути в конфигах валидны

[ ] B5.1 — Графики по сценам построены
[ ] B5.2 — Failure case visualization готов
[ ] B5.3 — Текстовые выводы записаны
```

---

## 10. Ссылки и материалы

- **Depth Anything v2** на HuggingFace: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- **Virtual KITTI 2 paper:** Cabon et al., 2020 — https://arxiv.org/abs/2001.10773
- **KITTI depth eval protocol:** https://www.cvlibs.net/datasets/kitti/eval_depth.php
- **Стандартные depth-метрики:** Eigen et al., 2014 — https://arxiv.org/abs/1406.2283

---

*Удачи! Если что-то непонятно — пиши :)
