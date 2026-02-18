"""
Minimal chumpy stub for unpickling SMPL .pkl files.

The real chumpy package has build issues on modern Python/pip.
This stub provides just enough for smplx to load SMPL body model files,
where chumpy arrays are deserialized via pickle and converted to numpy.

Key insight: chumpy's Ch is a plain Python object (NOT an ndarray subclass).
Pickle reconstructs Ch() with no args, then calls __setstate__ with a dict
containing the actual data. smplx then calls np.array(ch_obj) which invokes
__array__ to get the underlying numpy data.
"""
import numpy as np


class Ch:
    """Stub for chumpy.ch.Ch — plain object that converts to numpy on access."""

    def __init__(self, *args, **kwargs):
        if args:
            self.x = np.asarray(args[0])
        else:
            self.x = None

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.x = state

    def __getstate__(self):
        return self.__dict__

    def __array__(self, dtype=None):
        """Called by np.array(ch_obj) — returns the underlying data."""
        for attr in ('x', '_data', 'data', 'a'):
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    if hasattr(val, '__array__'):
                        return np.array(val, dtype=dtype)
                    return np.asarray(val, dtype=dtype)
        return np.array([], dtype=dtype)

    @property
    def r(self):
        """The 'r' property returns the raw value as ndarray."""
        return np.asarray(self)

    @property
    def shape(self):
        return np.asarray(self).shape

    @property
    def ndim(self):
        return np.asarray(self).ndim

    @property
    def size(self):
        return np.asarray(self).size

    @property
    def dtype(self):
        return np.asarray(self).dtype

    @property
    def T(self):
        return np.asarray(self).T

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, idx):
        return np.asarray(self)[idx]

    def __float__(self):
        return float(np.asarray(self))

    def __int__(self):
        return int(np.asarray(self))

    def __repr__(self):
        return f"Ch({np.asarray(self)!r})"
