# code/shared/

Общие Python-модули, используемые обеими треками:

- `kitti_loader.py` — загрузчик KITTI (пишет Наргиз)
- `vkitti/_loader.py` — основная реализация загрузчика Virtual KITTI 2
- `vkitti_loader.py` — совместимый re-export `VKITTI2Loader` для старых ноутбуков
- `metrics.py` — 2D IoU matching, 3D localization/depth metrics
- `lift_to_3d.py` — алгоритм проекции 2D→3D (пишет Наргиз)
- `visualization.py` — общие функции визуализации

**Правило:** изменения в shared-модулях согласуем через `coordination.ipynb`, чтобы не сломать друг другу пайплайны.
