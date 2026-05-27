# code/track_a/ — Наргиз

Локально воспроизводимые ноутбуки Track A:

- `01_kitti_setup.ipynb` — загрузка и парсинг KITTI
- `02_yolov8_inference.ipynb` — 2D-детекция YOLOv8
- `03_lift_to_3d.ipynb` — алгоритм Lift-to-3D
- `04_metrics_3d.ipynb` — метрики 3D-локализации
- `05_visualizations.ipynb` — визуализации для статьи

Вспомогательные файлы:

- `track_a_pipeline.py` — общая логика A1-A5 и сохранение артефактов
- `make_track_a_notebooks.py` — пересборка коротких локальных ноутбуков
- `execute_track_a_notebooks.py` — последовательный запуск A1-A5 через `nbclient`

По умолчанию A2 использует `yolov8n.pt`, потому что локальный запуск идет на CPU.
Для Colab/GPU можно задать `TRACK_A_YOLO_MODEL=yolov8m.pt`.
