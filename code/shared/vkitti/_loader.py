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