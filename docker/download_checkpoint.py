"""Download HMR 2.0 checkpoint at Docker build time.
Uses a pyrender mock since pyrender is not installed in the container.
"""
import sys
import os
import types
import importlib.abc
import importlib.machinery


class _PyrenderMockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'pyrender' or fullname.startswith('pyrender.'):
            return importlib.machinery.ModuleSpec(fullname, _PyrenderMockLoader())
        return None


class _PyrenderMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        class _Mod:
            OffscreenRenderer = type('OffscreenRenderer', (), {
                '__init__': lambda s, *a, **k: None,
                'render': lambda s, *a, **k: (None, None),
                'delete': lambda s: None,
            })
            class Mesh:
                @staticmethod
                def from_trimesh(*a, **k): return None
            class Scene:
                def __init__(s, *a, **k): pass
                def add(s, *a, **k): return None
            class PerspectiveCamera:
                def __init__(s, *a, **k): pass
            class DirectionalLight:
                def __init__(s, *a, **k): pass
            class Node:
                def __init__(s, *a, **k): pass
            RenderFlags = type('RenderFlags', (), {'RGBA': 1, 'DEPTH_ONLY': 2, 'FLAT': 4})()
            def __getattr__(self, name):
                return lambda *a, **k: None
        return _Mod()

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _PyrenderMockFinder())

os.environ['HOME'] = '/root'
sys.path.insert(0, '/app/4D-Humans')

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, DEFAULT_CHECKPOINT
from pathlib import Path

ckpt = Path(DEFAULT_CHECKPOINT)
if not ckpt.exists():
    print("Downloading HMR 2.0 checkpoint (~700MB)...")
    download_models(CACHE_DIR_4DHUMANS)

print(f"Checkpoint ready: {ckpt.exists()}")
