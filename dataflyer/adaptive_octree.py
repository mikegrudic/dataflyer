"""Minimal particle container used by the GPU subsample upload.

In its previous form this module implemented a full Morton-sorted
adaptive octree with multi-level moments and CPU frustum culling. The
GPU subsample path doesn't need any of that — it only needs the raw
particle arrays + a way to swap mass/qty after a field change. This
file keeps just that interface, under the same class name so existing
call sites (`grid._raw_pos`, `grid.update_weights(...)`, etc.) keep
working.
"""

import numpy as np


class AdaptiveOctree:
    """Holds the raw particle arrays for the GPU upload pipeline."""

    def __init__(self, positions, masses, hsml, quantity):
        def _f32(a):
            return a if a.dtype == np.float32 else a.astype(np.float32)
        self._raw_pos = _f32(positions)
        self._raw_mass = _f32(masses)
        self._raw_hsml = _f32(hsml)
        self._raw_qty = _f32(quantity)

        # The GPU upload path only ever reads `_raw_*`. The `sorted_*`
        # attributes exist for legacy callers that conditionally check
        # them; they're always None now.
        self.sort_order = None
        self.sorted_pos = None
        self.sorted_hsml = None
        self.sorted_mass = None
        self.sorted_qty = None

    def update_weights(self, masses, quantity=None):
        """Replace the mass/qty arrays after a field swap."""
        self._raw_mass = np.asarray(masses, dtype=np.float32)
        if quantity is None:
            self._raw_qty = self._raw_mass
        else:
            self._raw_qty = np.asarray(quantity, dtype=np.float32)
