# Сравнительный анализ влияния синтетических дорожных данных на качество моно-3D-локализации объектов

**Авторы:** Исмаилова Н., Донецкая Е.  
**Научный руководитель:** Садеков Р.Н.  
**МИСИС, магистратура «ИИ и машинное обучение», 1 курс**

## Аннотация

**Авторы:** совместно.  
**Цель раздела:** кратко описать задачу, пайплайн YOLOv8 + Depth Anything v2 + Lift-to-3D, сравнение стратегий калибровки `real_only`, `synth_only`, `mixed_50_50` и главный результат.  
**Числа для включения:** J3 val split — 11 KITTI кадров, 22 matched Car; лучший Car median 3D error — `synth_only`, 4.23 м.  
**Ключевые слова:** моно-3D-локализация, синтетические данные, аугментация, оценка глубины, дорожные сцены, Depth Anything v2, YOLOv8, KITTI, Virtual KITTI 2.

## 1. Введение

**Автор:** Наргиз.  
**Что написать:** почему 3D-аннотации дороги, зачем нужны синтетические данные, какая гипотеза проверяется и почему выбран простой воспроизводимый pipeline без обучения с нуля.  
**Обязательные акценты:** проект исследует не SOTA 3D detector, а практическую связку 2D detection + monocular depth + camera back-projection.  
**Артефакты:** `README.md`, `Track A - Наргиз.md`, `results/joint/J3_experiment_summary.json`.

## 2. Обзор литературы

**Автор:** Лена.  
**Что написать:** synthetic data для autonomous driving, sim-to-real/domain gap, monocular depth estimation, Depth Anything v2, KITTI и Virtual KITTI 2.  
**Артефакты:** `paper/references.bib`, `results/track_b/B5_absrel_by_scene.png`, `results/track_b/B5_failure_cases.png`.  
**Минимум ссылок:** Geiger et al. 2012, Cabon et al. 2020, Yang et al. 2024, Jocher et al. 2023, Tobin et al. 2017, Eigen et al. 2014.

## 3. Используемые данные

**Авторы:** Наргиз — KITTI, Лена — Virtual KITTI 2.  
**KITTI:** реальные дорожные сцены, RGB `image_2`, 2D/3D labels, calibration `P2`; локально в J1 доступно 2579 изображений, 2865 labels, 2853 calib files.  
**Virtual KITTI 2:** синтетические сцены с RGB/depth; Track B depth predictions доступны для VKITTI2 и KITTI, J3 synthetic calibration использует доступные GT depth files.  
**Ограничение:** локальный `clone` GT depth неполный, поэтому J3 фиксирует fallback на доступные VKITTI2 variation files.

## 4. Метод

**Авторы:** совместно, Наргиз ведёт математическую часть.  
**4.1. Общая архитектура:** KITTI RGB -> YOLOv8 detections -> Depth Anything relative depth -> affine calibration `depth_m = scale * depth_rel + shift` -> Lift-to-3D.  
**4.2. 2D-детекция:** YOLOv8, COCO-to-KITTI class mapping, оценка по 2D IoU.  
**4.3. Моно-глубина:** Depth Anything v2, относительная глубина, необходимость scale/shift calibration.  
**4.4. Lift-to-3D:** back-projection через `fx, fy, cx, cy`; выбран `anchor='bottom'`, `aggregation='median'`.  
**Артефакты:** `code/shared/lift_to_3d.py`, `code/shared/metrics.py`, `results/track_a/A3_lift_to_3d_summary.json`.

## 5. Эксперименты

**Авторы:** совместно.  
**Track A sanity-check:** oracle GT object depth проверяет корректность геометрии Lift-to-3D отдельно от ошибки Depth Anything.  
**J1-J2:** проверка совместимости артефактов и первый полный pipeline на 55 common KITTI frames.  
**J3:** сравнение трёх стратегий калибровки: `real_only`, `synth_only`, `mixed_50_50`; оценка проводится на одном KITTI val split из 11 кадров.  
**Метрики:** mean/median 3D error, mean depth error, localization accuracy @2m/@4m, matched count.

## 6. Результаты

**Авторы:** совместно, Наргиз собирает таблицы.  
**Таблица Track A:** `results/joint/J4_table_track_a_summary.csv`.  
**Таблица Track B:** `results/joint/J4_table_track_b_depth_by_scene.csv`.  
**Таблица Joint:** `results/joint/J4_table_joint_car_metrics.csv`.  
**Главные числа Car:** `real_only` median 3D error — 6.12 м; `synth_only` — 4.23 м; `mixed_50_50` — 5.40 м.  
**Графики:** `results/joint/figures/J4_joint_car_mean_median_errors.png`, `results/joint/figures/J4_j3_car_error_boxplot.png`, `results/joint/figures/J4_qualitative_overview.png`.

## 7. Обсуждение

**Автор:** Лена, финальная интерпретация совместно.  
**Что написать:** почему `synth_only` оказался лучшим по median Car error на малом split; чем это может объясняться; почему результат нельзя переобобщать без расширенного val set.  
**Обязательные ограничения:** 11 val frames, 22 matched Car; неполный VKITTI2 clone GT depth; Depth Anything behaves like inverse/relative depth, поэтому negative scale во всех стратегиях ожидаем.

## 8. Заключение

**Автор:** Наргиз.  
**Что написать:** pipeline работает end-to-end, synthetic calibration в текущем J3 дала лучший median результат для Car, но дальнейшая работа должна расширить KITTI split, добавить больше VKITTI2 clone depth и проверить пропорции synthetic/real beyond 50/50.  
**Финальный вывод:** синтетика полезна как источник depth calibration signal, но текущий результат является пилотным и требует более широкого подтверждения.

## Список литературы

**Авторы:** совместно.  
**Источник:** `paper/references.bib`.  
**Проверить перед сдачей:** единый стиль ссылок, достаточное число источников, наличие ссылок на KITTI, VKITTI2, Depth Anything v2 и YOLOv8.
