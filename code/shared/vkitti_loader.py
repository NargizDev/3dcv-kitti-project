"""Compatibility wrapper for the Virtual KITTI 2 loader.

Older notebooks import ``VKITTI2Loader`` from ``code/shared/vkitti_loader.py``.
The implementation now lives in ``code/shared/vkitti/_loader.py``.
"""

from vkitti import VKITTI2Loader

__all__ = ['VKITTI2Loader']
