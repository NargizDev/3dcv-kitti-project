# code/

Структура:

- **`track_a/`** — ноутбуки и скрипты Наргиз (KITTI + YOLOv8 + Lift-to-3D)
- **`track_b/`** — ноутбуки и скрипты Лены (VKITTI2 + Depth Anything v2)
- **`joint/`** — интеграция Track A + Track B, J1-J4 эксперименты и article-ready outputs
- **`shared/`** — общие модули, используемые обеими (loaders, metrics, utilities)

**Правило:** работаем только в своих папках. Совместный код — только в `shared/` и через явное согласование в `coordination.ipynb`.

## Соглашение по именованию ноутбуков

`NN_короткое_описание.ipynb`, например:
- `01_kitti_setup.ipynb`
- `02_yolov8_inference.ipynb`
- `03_lift_to_3d.ipynb`

## Текущий статус

- Track A локально воспроизводится через `code/track_a/execute_track_a_notebooks.py`.
- Joint J1-J2 запускается через `code/joint/01_j1_j2_integration.ipynb`.
- Joint J3 запускается через `code/joint/02_j3_final_experiments.ipynb`.
- J4 article assets собираются скриптом `code/joint/j4_article_assets.py`.
