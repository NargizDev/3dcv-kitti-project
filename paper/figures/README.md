# paper/figures/ — система управления изображениями статьи

Эта папка хранит **все** изображения для статьи «Анализ влияния синтетической
аугментации для моно-3D-локализации в дорожных сценах», вместе с привязкой к
исходным кадрам датасетов и подписями для подачи в журнал.

## Иерархия

```
paper/figures/
├── source/                          # Оригиналы — НЕ редактируем
│   ├── kitti/                       # Кадры KITTI image_2 (имя = frame_id)
│   ├── vkitti2/                     # Кадры Virtual KITTI 2
│   └── generated/                   # PNG, сгенерированные пайплайном/ноутбуками
├── final/                           # Конечные PNG, вставленные в статью
├── journal_submission/              # Пакет для подачи в журнал (TIFF + captions)
├── editor/                          # Рабочие PowerPoint исходники (Лена)
├── figures_manifest.csv             # Истина: что, откуда, где использовано
└── README.md                        # Этот файл
```

## Правила

1. `source/` — write-once. Никогда не пересохраняем поверх. Если правишь —
   копия в `editor/<имя>.pptx`, итог в `final/Fig_NN_*.png`.
2. Имена в `final/` — **латиница**: `Fig_01_kitti_examples.png`. Кириллица в
   именах файлов ломается в части редакционных систем при заливке.
3. При замене картинки нужно обновить:
   а) `final/Fig_NN_*.png` — сам файл;
   б) `journal_submission/Figure_N.tif` — пересобрать через `export_journal_pack.py`;
   в) строку в `figures_manifest.csv` (поле `last_modified`).
4. Сборщик статьи (`paper/build_final_article.py`) тянет картинки **только** из
   `paper/figures/final/`. Не из `results/...` напрямую.

## Что хранится в манифесте

Колонки `figures_manifest.csv` (разделитель `;`, кодировка UTF-8):

- `figure_id` — `Рис. 1`, `Рис. 2`, …
- `final_filename` — относительный путь к финальной картинке.
- `source_files` — исходники, перечисленные через `|`.
- `dataset_origin` — `KITTI` / `VKITTI2` / `generated` / `mixed`.
- `dataset_frame_ids` — конкретные ID кадров датасета (через `, `). Для
  generated — пусто.
- `original_dataset_paths` — абсолютные пути в исходной структуре датасета.
- `produced_by` — скрипт/ноутбук, который создал картинку (для воспроизводимости).
- `random_seed` — если применимо.
- `caption_ru`, `caption_en` — подписи (без префикса «Рис. N.»).
- `used_in_section` — раздел статьи, где картинка фигурирует.
- `dpi`, `width_cm_in_article` — параметры вставки.
- `last_modified` — ISO-дата последнего ручного редактирования.

## Воркфлоу при правке («увеличьте Рис. 3 и пришлите оригинал»)

1. Открыть `figures_manifest.csv` → найти `Рис. 3`.
2. В колонке `source_files` — список оригиналов.
3. Колонка `original_dataset_paths` — где в исходной структуре датасета
   лежат кадры, попавшие на картинку (для запросов вида «оригинальный
   кадр без аннотаций»).
4. Колонка `produced_by` — какой ноутбук/скрипт сгенерировал. Открыть,
   подправить параметры, перезапустить → перезаписывает `source/generated/...`
   → копировать в `final/Fig_03_*.png` → пересобрать.

## Скрипты в `paper/` для работы с figures

- `paper/build_final_article.py` — собирает `Финальная_статья.docx`, тянет из `final/`.
- `paper/make_captions_doc.py` — читает manifest, генерирует `journal_submission/Captions_RU.docx` и `Captions_EN.docx`.
- `paper/export_journal_pack.py` — конвертирует `final/*.png` → `journal_submission/Figure_N.tif` (300 dpi LZW).
- `paper/check_figures_quality.py` — проверяет DPI, размер, формат всех `final/*.png`.
- `paper/make_vkitti_composite.py` — собирает Рис. 2 из 4 VKITTI вариаций.

Полная пересборка пакета подачи:

```
python paper/check_figures_quality.py
python paper/export_journal_pack.py
python paper/make_captions_doc.py
python paper/build_final_article.py
```

## Связь с датасетом

- KITTI: имя файла в `source/kitti/` = `frame_id` (например `000917.png`).
  Полный путь в датасете — `data/kitti/training/training/image_2/000917.png`.
  Сопоставление train/val для эксперимента J3 — `results/joint/J3_experiment_summary.json`.
- VKITTI2: имя файла в `source/vkitti2/` имеет формат
  `{Scene}_{variation}_{frame_id}.png`, исходный путь —
  `data/vkitti2/{Scene}/{variation}/frames/rgb/Camera_0/rgb_{frame_id}.jpg`
  (локальная копия может быть неполной — см. fallback в `J3_experiment_summary.json`).

## Бэкап

Всё содержимое `paper/figures/` должно синхронизироваться в Google Drive
вместе с проектом. PowerPoint исходники (`editor/*.pptx`) — обязательно,
без них правки картинок не воспроизводятся.
