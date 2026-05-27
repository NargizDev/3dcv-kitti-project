# План написания статьи

## Общий статус

- Техническая часть Track A, Track B handoff, J1-J2, J3 и J4 доведена до состояния, пригодного для статьи.
- Основной результат J3: на текущем KITTI val split лучшая стратегия по Car median 3D error — `synth_only`.
- Статья пишется после синхронизации локального проекта с Google Drive и передачи Лене `results/joint/J3_lena_brief.md`.

## Наргиз

- **Введение:** проблема дорогих 3D-аннотаций, цель проекта, исследовательские вопросы, вклад работы.
- **Метод:** YOLOv8 + Depth Anything v2 + affine depth calibration + Lift-to-3D через калибровку KITTI.
- **Track A:** KITTI loader, 2D detection, oracle-depth sanity-check, выбор `anchor='bottom'`, `aggregation='median'`.
- **Joint J1-J3:** compatibility check, integrated pipeline, real/synth/mixed calibration experiments.
- **Результаты:** финальные J3/J4 таблицы, интерпретация Car metrics, ограничения маленького split.
- **Заключение:** что показал эксперимент и какие улучшения нужны дальше.

## Лена

- **Обзор литературы:** synthetic data, sim-to-real/domain gap, monocular depth estimation.
- **Данные:** Virtual KITTI2, вариации сцен, формат RGB/depth, отличие от KITTI.
- **Track B:** Depth Anything v2 setup, VKITTI2 depth metrics, B3/B5 результаты.
- **Domain gap:** объяснить failure cases, различия синтетики и реальных KITTI кадров.
- **Обсуждение ограничений:** неполный локальный VKITTI2 clone GT depth, fallback на доступные VKITTI2 variation files, переносимость результата.

## Совместно

- Аннотация RU/EN и ключевые слова.
- Дизайн экспериментов `real_only`, `synth_only`, `mixed_50_50`.
- Финальная интерпретация: синтетическая калибровка улучшила median Car 3D error на малом val split, но вывод нужно формулировать осторожно.
- Вычитка, подписи таблиц/рисунков, список литературы.

## Артефакты для цитирования

- Track A: `results/track_a/A2_2d_metrics.csv`, `results/track_a/A3_lift_to_3d_summary.json`, `results/track_a/metrics_3d.csv`.
- Track B: `results/track_b/B3_depth_metrics_vkitti.csv`, `results/track_b/B5_absrel_by_scene.png`, `results/track_b/B5_failure_cases.png`.
- Joint: `results/joint/J3_final_metrics_table.csv`, `results/joint/J3_experiment_summary.json`, `results/joint/J4_table_joint_car_metrics.csv`.
- Figures: `results/joint/figures/J4_joint_car_mean_median_errors.png`, `results/joint/figures/J4_j3_car_error_boxplot.png`, `results/joint/figures/J4_qualitative_overview.png`.

## Ключевые числа

- J3 KITTI val split: 11 кадров, 22 matched Car.
- Car median 3D error: `real_only` — 6.12 м, `synth_only` — 4.23 м, `mixed_50_50` — 5.40 м.
- Car Acc@4m: `real_only` — 0.273, `synth_only` — 0.500, `mixed_50_50` — 0.318.
- Depth Anything predictions ведут себя как inverse/relative depth: scale отрицательный во всех стратегиях.
