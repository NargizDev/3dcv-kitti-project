
import os

import numpy as np
from PIL import Image


class KITTIObject:
    """
    Один объект из KITTI label файла
    """

    def __init__(self, line):
        parts = line.strip().split()
        if len(parts) < 15:
            raise ValueError(f'Строка KITTI label должна содержать 15 полей, получено {len(parts)}: {line!r}')

        self.type = parts[0]

        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])

        # x1, y1, x2, y2
        self.bbox_2d = np.array(
            [float(x) for x in parts[4:8]],
            dtype=np.float32
        )

        # h, w, l
        self.dimensions = np.array(
            [float(x) for x in parts[8:11]],
            dtype=np.float32
        )

        # x, y, z
        self.location = np.array(
            [float(x) for x in parts[11:14]],
            dtype=np.float32
        )

        self.rotation_y = float(parts[14])

    @property
    def depth(self):
        """
        Z-координата объекта в системе камеры
        """
        return self.location[2]


class KITTILoader:

    CLASSES = [
        'Car',
        'Pedestrian',
        'Cyclist',
        'Van',
        'Truck',
        'Person_sitting',
        'Tram',
        'Misc'
    ]

    def __init__(self, root_dir, split='training'):
        self.root = os.path.join(root_dir, split)

        self.image_dir = os.path.join(self.root, 'image_2')
        self.label_dir = os.path.join(self.root, 'label_2')
        self.calib_dir = os.path.join(self.root, 'calib')

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f'Папка KITTI image_2 не найдена: {self.image_dir}')
        if not os.path.isdir(self.calib_dir):
            raise FileNotFoundError(f'Папка KITTI calib не найдена: {self.calib_dir}')

        self.frame_ids = sorted([
            f.replace('.png', '')
            for f in os.listdir(self.image_dir)
            if f.endswith('.png')
        ])

    def __len__(self):
        return len(self.frame_ids)

    def load_image(self, frame_id):
        path = os.path.join(
            self.image_dir,
            f'{frame_id}.png'
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f'Изображение KITTI не найдено: {path}')

        return np.array(Image.open(path))

    def load_labels(self, frame_id):
        path = os.path.join(
            self.label_dir,
            f'{frame_id}.txt'
        )

        if not os.path.exists(path):
            return []

        objects = []

        with open(path, 'r', encoding='utf-8') as f:

            for line in f:
                if not line.strip():
                    continue

                obj = KITTIObject(line)

                if obj.type != 'DontCare':
                    objects.append(obj)

        return objects

    def load_calib(self, frame_id):
        path = os.path.join(
            self.calib_dir,
            f'{frame_id}.txt'
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f'Файл калибровки KITTI не найден: {path}')

        calib = {}

        with open(path, 'r', encoding='utf-8') as f:

            for line in f:

                if ':' not in line:
                    continue

                key, value = line.split(':', 1)

                calib[key.strip()] = np.array(
                    [float(x) for x in value.split()],
                    dtype=np.float32
                )

        if 'P2' not in calib:
            raise KeyError(f'Файл калибровки не содержит матрицу P2: {path}')

        calib['P2'] = calib['P2'].reshape(3, 4)

        if 'R0_rect' in calib:
            calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)

        if 'Tr_velo_to_cam' in calib:
            calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)

        return calib

    def get_intrinsics(self, frame_id):

        P2 = self.load_calib(frame_id)['P2']

        K = P2[:, :3]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        return fx, fy, cx, cy
