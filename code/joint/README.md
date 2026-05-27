# code/joint/

Совместная интеграция Track A и Track B.

- `joint_pipeline.py` — J1 compatibility check и J2 pipeline
- `01_j1_j2_integration.ipynb` — notebook-обёртка для запуска J1-J2
- `j3_final_experiments.py` — J3 сравнение real_only / synth_only / mixed_50_50
- `02_j3_final_experiments.ipynb` — notebook-обёртка для запуска J3
- `j4_article_assets.py` — J4 лёгкие таблицы и графики для статьи из готовых результатов
- `make_j3_notebook.py` / `execute_j3_notebook.py` — генерация и запуск J3 notebook

J2 использует Track A YOLO JSON и Track B KITTI depth `.npy`, калибрует
относительную глубину через KITTI GT object depths и сохраняет результаты в
`results/joint/`.

J3 сравнивает три стратегии калибровки depth (`real_only`, `synth_only`,
`mixed_50_50`) на одном KITTI val split. J4 не запускает модели заново, а
собирает article-ready CSV/PNG из Track A, Track B и J3 outputs.
