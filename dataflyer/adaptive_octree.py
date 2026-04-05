"""Adaptive octree for frustum culling and LOD rendering.

Morton-sorted octree with configurable leaf size and arbitrary depth.
Dense 8-children-per-parent layout for simple GPU indexing.
Drop-in replacement for SpatialGrid with the same public interface.
"""

import numpy as np
from numba import njit, prange
from .spatial_grid import CellMoments
from .spatial_grid import _gather_subsampled_direct, _gather_importance_sampled


# ---- Morton code utilities ----

@njit(cache=True)
def _split_by_3(x):
    """Spread 21-bit integer into every 3rd bit position for Morton encoding."""
    x = np.uint64(x) & np.uint64(0x1fffff)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    x = (x | (x << np.uint64(8)))  & np.uint64(0x100f00f00f00f00f)
    x = (x | (x << np.uint64(4)))  & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return x


@njit(cache=True)
def _morton_encode(ix, iy, iz):
    return _split_by_3(ix) | (_split_by_3(iy) << np.uint64(1)) | (_split_by_3(iz) << np.uint64(2))


@njit(parallel=True, cache=True)
def _compute_morton_codes(positions, pmin, inv_box, max_coord):
    """Compute 63-bit Morton codes for all particles."""
    n = len(positions)
    codes = np.empty(n, dtype=np.uint64)
    mc = np.int64(max_coord)
    for i in prange(n):
        fx = (positions[i, 0] - pmin[0]) * inv_box[0]
        fy = (positions[i, 1] - pmin[1]) * inv_box[1]
        fz = (positions[i, 2] - pmin[2]) * inv_box[2]
        ix = min(max(int(fx * mc), 0), mc - 1)
        iy = min(max(int(fy * mc), 0), mc - 1)
        iz = min(max(int(fz * mc), 0), mc - 1)
        codes[i] = _morton_encode(ix, iy, iz)
    return codes


@njit(cache=True)
def _compact_by_3(x):
    """Inverse of _split_by_3: extract every 3rd bit into contiguous integer."""
    x = x & np.uint64(0x1249249249249249)
    x = (x | (x >> np.uint64(2)))  & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x >> np.uint64(4)))  & np.uint64(0x100f00f00f00f00f)
    x = (x | (x >> np.uint64(8)))  & np.uint64(0x1f0000ff0000ff)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x1f00000000ffff)
    x = (x | (x >> np.uint64(32))) & np.uint64(0x1fffff)
    return x


@njit(cache=True)
def _morton_decode(code):
    """Decode Morton code to (ix, iy, iz)."""
    ix = np.int64(_compact_by_3(code))
    iy = np.int64(_compact_by_3(code >> np.uint64(1)))
    iz = np.int64(_compact_by_3(code >> np.uint64(2)))
    return ix, iy, iz


# ---- Fast tree build from sorted Morton codes ----

def _build_adaptive_leaves(sorted_codes, n, leaf_size, max_depth):
    """Build variable-depth leaves: subdivide only cells with >leaf_size particles.

    Returns: (leaf_codes, leaf_depths, cell_start) where each leaf has its own depth.
    leaf_codes are Morton codes at max_depth resolution.
    """
    # Start at depth 1: all particles in cells at depth 1
    # Iteratively subdivide cells exceeding leaf_size
    # Particle ranges are contiguous in sorted_codes, so cell boundaries
    # can be found by shifts at each depth.

    # Track which particles are in "done" (leaf) cells vs "pending" (need subdivision)
    # Use a list of (start, end, depth, code_at_depth) per cell
    # Initialize at depth 1
    shift = np.uint64(3 * (max_depth - 1))
    codes_d = sorted_codes >> shift
    changes = np.empty(n, dtype=np.bool_)
    changes[0] = True
    changes[1:] = codes_d[1:] != codes_d[:-1]
    starts = np.where(changes)[0]
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = n

    # Each cell: (start, end, depth)
    pending = []
    leaves = []  # (start, end, depth, morton_code_at_max_depth)
    for i in range(len(starts)):
        count = ends[i] - starts[i]
        if count <= leaf_size:
            leaves.append((int(starts[i]), int(ends[i]), 1))
        else:
            pending.append((int(starts[i]), int(ends[i]), 1))

    for d in range(2, max_depth + 1):
        if not pending:
            break
        shift = np.uint64(3 * (max_depth - d))
        new_pending = []
        for (s, e, _dep) in pending:
            # Subdivide this cell's particles at depth d
            sub_codes = sorted_codes[s:e] >> shift
            sub_changes = np.empty(e - s, dtype=np.bool_)
            sub_changes[0] = True
            sub_changes[1:] = sub_codes[1:] != sub_codes[:-1]
            sub_starts = np.where(sub_changes)[0] + s
            sub_ends = np.empty(len(sub_starts), dtype=np.int64)
            sub_ends[:-1] = sub_starts[1:]
            sub_ends[-1] = e
            for j in range(len(sub_starts)):
                count = sub_ends[j] - sub_starts[j]
                if count <= leaf_size or d == max_depth:
                    leaves.append((int(sub_starts[j]), int(sub_ends[j]), d))
                else:
                    new_pending.append((int(sub_starts[j]), int(sub_ends[j]), d))
        pending = new_pending

    # Any remaining pending cells become leaves at max_depth
    for (s, e, _dep) in pending:
        leaves.append((s, e, max_depth))

    # Sort leaves by their start index (maintains Morton order)
    leaves.sort(key=lambda x: x[0])

    # Build cell_start CSR and leaf codes
    n_leaves = len(leaves)
    cell_start = np.empty(n_leaves + 1, dtype=np.int64)
    leaf_depths = np.empty(n_leaves, dtype=np.int32)
    leaf_codes = np.empty(n_leaves, dtype=np.uint64)
    for i, (s, e, d) in enumerate(leaves):
        cell_start[i] = s
        leaf_depths[i] = d
        # Store the Morton code at the leaf's own depth
        leaf_codes[i] = sorted_codes[s] >> np.uint64(3 * (max_depth - d))
    cell_start[n_leaves] = n

    return leaf_codes, leaf_depths, cell_start


def _build_tree(sorted_codes, n, leaf_size, max_depth, pmin, box):
    """Build adaptive octree from sorted Morton codes.

    Variable-depth leaves: dense regions subdivide deeper, sparse regions stop early.
    Returns: (levels, cell_start) where levels[0] is finest (all leaves).
    """
    leaf_codes, leaf_depths, cell_start = _build_adaptive_leaves(
        sorted_codes, n, leaf_size, max_depth)
    n_leaves = len(leaf_codes)
    actual_max_depth = int(leaf_depths.max()) if n_leaves > 0 else 1

    # Compute leaf centers (each leaf has its own depth → its own cell size)
    leaf_centers = np.empty((n_leaves, 3), dtype=np.float32)
    leaf_half_diag = np.empty(n_leaves, dtype=np.float32)
    for i in range(n_leaves):
        d = leaf_depths[i]
        cs = box / (2 ** d)
        ix, iy, iz = _morton_decode(leaf_codes[i])
        leaf_centers[i, 0] = pmin[0] + (ix + 0.5) * cs[0]
        leaf_centers[i, 1] = pmin[1] + (iy + 0.5) * cs[1]
        leaf_centers[i, 2] = pmin[2] + (iz + 0.5) * cs[2]
        leaf_half_diag[i] = float(np.linalg.norm(cs) * 0.5)

    # Use median cell size for the level "cs" (used for summary overlap padding)
    median_depth = int(np.median(leaf_depths))
    median_cs = (box / (2 ** median_depth)).astype(np.float32)

    # Build levels bottom-up. Level 0 = all leaves (variable depth).
    # Coarser levels group leaves by their parent Morton codes.
    levels = []
    levels.append({
        "nc": n_leaves,
        "depth": actual_max_depth,  # nominal depth for the level
        "cs": median_cs,
        "half_diag": float(np.median(leaf_half_diag)),
        "centers": leaf_centers,
        "cell_codes": leaf_codes,
        "_per_cell_half_diag": leaf_half_diag,  # variable per cell
    })

    # Build coarser levels: group by parent Morton code at each depth
    cur_codes = leaf_codes
    cur_depths = leaf_depths

    for d in range(actual_max_depth - 1, -1, -1):
        # Parent code: for leaves at depth > d, shift to depth d
        # For leaves at depth <= d, they ARE the parent (or ancestor)
        parent_codes = np.empty(len(cur_codes), dtype=np.uint64)
        for i in range(len(cur_codes)):
            if cur_depths[i] > d:
                # Shift from leaf depth to depth d
                shift = np.uint64(3 * (cur_depths[i] - d))
                parent_codes[i] = cur_codes[i] >> shift
            else:
                # This leaf is at or above depth d — use its own code shifted
                shift = np.uint64(3 * (cur_depths[i] - d)) if cur_depths[i] > d else np.uint64(0)
                parent_codes[i] = cur_codes[i] >> shift if cur_depths[i] > d else cur_codes[i]

        unique_parents, inverse = np.unique(parent_codes, return_inverse=True)
        n_parents = len(unique_parents)

        child_offset = np.empty(n_parents, dtype=np.uint32)
        parent_idx = np.empty(len(cur_codes), dtype=np.uint32)
        ci = 0
        for pi in range(n_parents):
            child_offset[pi] = ci
            while ci < len(cur_codes) and inverse[ci] == pi:
                parent_idx[ci] = pi
                ci += 1

        levels[-1]["parent_idx"] = parent_idx

        cell_size = box / (2 ** d) if d > 0 else box.copy()
        half_diag = float(np.linalg.norm(cell_size) * 0.5)
        centers = _compute_centers_from_codes(unique_parents, d, pmin, cell_size)

        levels.append({
            "nc": n_parents,
            "depth": d,
            "cs": cell_size.astype(np.float32),
            "half_diag": half_diag,
            "centers": centers,
            "cell_codes": unique_parents,
            "child_offset": child_offset,
        })

        cur_codes = unique_parents
        # All parents are at depth d
        cur_depths = np.full(n_parents, d, dtype=np.int32)

    # Root has no parent
    if "parent_idx" not in levels[-1]:
        levels[-1]["parent_idx"] = np.zeros(levels[-1]["nc"], dtype=np.uint32)

    return levels, cell_start, actual_max_depth


@njit(parallel=True, cache=True)
def _compute_centers_vec(codes, depth, pmin, cell_size):
    """Compute cell center coordinates from Morton codes."""
    n = len(codes)
    centers = np.empty((n, 3), dtype=np.float32)
    for i in prange(n):
        code = codes[i]
        # Decode at the correct depth
        if depth > 0:
            ix, iy, iz = _morton_decode(code)
        else:
            ix, iy, iz = np.int64(0), np.int64(0), np.int64(0)
        centers[i, 0] = pmin[0] + (ix + np.float32(0.5)) * cell_size[0]
        centers[i, 1] = pmin[1] + (iy + np.float32(0.5)) * cell_size[1]
        centers[i, 2] = pmin[2] + (iz + np.float32(0.5)) * cell_size[2]
    return centers


def _compute_centers_from_codes(codes, depth, pmin, cell_size):
    """Compute centers, handling depth 0 (root) specially."""
    if depth == 0:
        return np.array([[pmin[0] + cell_size[0] * 0.5,
                          pmin[1] + cell_size[1] * 0.5,
                          pmin[2] + cell_size[2] * 0.5]], dtype=np.float32)
    return _compute_centers_vec(codes, depth, pmin.astype(np.float32),
                                cell_size.astype(np.float32))


# ---- Moment coarsening ----

@njit(parallel=True, cache=True)
def _coarsen_from_children(child_offset, n_parents,
                           child_mass, child_mp, child_mq, child_mh2, child_mxx,
                           n_children_total):
    """Sum children's extensive moments into parent nodes."""
    parent_mass = np.empty(n_parents, dtype=np.float32)
    parent_mp = np.empty((n_parents, 3), dtype=np.float32)
    parent_mq = np.empty(n_parents, dtype=np.float32)
    parent_mh2 = np.empty(n_parents, dtype=np.float32)
    parent_mxx = np.empty((n_parents, 6), dtype=np.float32)

    for p in prange(n_parents):
        cs = child_offset[p]
        ce = child_offset[p + 1] if p < n_parents - 1 else n_children_total

        m = np.float64(0.0)
        mpx = np.float64(0.0)
        mpy = np.float64(0.0)
        mpz = np.float64(0.0)
        mq = np.float64(0.0)
        mh2 = np.float64(0.0)
        v0 = np.float64(0.0)
        v1 = np.float64(0.0)
        v2 = np.float64(0.0)
        v3 = np.float64(0.0)
        v4 = np.float64(0.0)
        v5 = np.float64(0.0)

        for c in range(cs, ce):
            m += np.float64(child_mass[c])
            mpx += np.float64(child_mp[c, 0])
            mpy += np.float64(child_mp[c, 1])
            mpz += np.float64(child_mp[c, 2])
            mq += np.float64(child_mq[c])
            mh2 += np.float64(child_mh2[c])
            v0 += np.float64(child_mxx[c, 0])
            v1 += np.float64(child_mxx[c, 1])
            v2 += np.float64(child_mxx[c, 2])
            v3 += np.float64(child_mxx[c, 3])
            v4 += np.float64(child_mxx[c, 4])
            v5 += np.float64(child_mxx[c, 5])

        parent_mass[p] = np.float32(m)
        parent_mp[p, 0] = np.float32(mpx)
        parent_mp[p, 1] = np.float32(mpy)
        parent_mp[p, 2] = np.float32(mpz)
        parent_mq[p] = np.float32(mq)
        parent_mh2[p] = np.float32(mh2)
        parent_mxx[p, 0] = np.float32(v0)
        parent_mxx[p, 1] = np.float32(v1)
        parent_mxx[p, 2] = np.float32(v2)
        parent_mxx[p, 3] = np.float32(v3)
        parent_mxx[p, 4] = np.float32(v4)
        parent_mxx[p, 5] = np.float32(v5)

    return parent_mass, parent_mp, parent_mq, parent_mh2, parent_mxx


@njit(cache=True)
def _shuffle_within_cells(sort_order, cell_start):
    """Fisher-Yates shuffle of sort_order within each cell.

    Deterministic seed per cell (based on cell index) for reproducibility.
    Modifies sort_order in-place.
    """
    n_cells = len(cell_start) - 1
    for c in range(n_cells):
        start = cell_start[c]
        end = cell_start[c + 1]
        n = end - start
        if n <= 1:
            continue
        seed = np.uint64(c * 2654435761 + 1)
        for i in range(n - 1, 0, -1):
            seed = seed * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            j = int(seed >> np.uint64(33)) % (i + 1)
            tmp = sort_order[start + i]
            sort_order[start + i] = sort_order[start + j]
            sort_order[start + j] = tmp


# ---- Main class ----

class AdaptiveOctree:
    """Adaptive octree for frustum culling and LOD rendering.

    Drop-in replacement for SpatialGrid with the same public interface.
    """

    def __init__(self, positions, masses, hsml, quantity, leaf_size=32, max_depth=10):
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.pmin = positions.min(axis=0).astype(np.float32)
        self.pmax = positions.max(axis=0).astype(np.float32)
        box = (self.pmax - self.pmin).astype(np.float64)
        box[box == 0] = 1.0
        self._box = box

        # Morton sort
        inv_box = 1.0 / box
        max_coord = 2 ** max_depth
        codes = _compute_morton_codes(
            positions.astype(np.float32), self.pmin.astype(np.float64),
            inv_box, max_coord)
        self.sort_order = np.argsort(codes)
        sorted_codes = codes[self.sort_order]

        # Store sorted particle data
        so = self.sort_order
        self.sorted_pos = positions[so].astype(np.float32)
        self.sorted_hsml = hsml[so].astype(np.float32)
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)

        # Build tree structure
        self.levels, self.cell_start, self._leaf_depth = _build_tree(
            sorted_codes, len(positions), leaf_size, max_depth, self.pmin, box)
        self.n_cells = self.levels[0]["nc"]

        # Shuffle particles within each cell so uniform stride ≈ random subsampling
        _shuffle_within_cells(self.sort_order, self.cell_start)
        so = self.sort_order
        self.sorted_pos = positions[so].astype(np.float32)
        self.sorted_hsml = hsml[so].astype(np.float32)
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)

        # Compute moments and NN-based smoothing lengths
        self._compute_all_moments()
        self._compute_nn_hsml()

    def _compute_all_moments(self):
        """Compute extensive and intensive moments for all levels."""
        # Leaf level: accumulate from particles
        finest = self.levels[0]
        n_leaves = finest["nc"]
        moments = CellMoments(n_leaves)
        moments.accumulate_from_particles(
            self.cell_start, self.sorted_pos, self.sorted_mass,
            self.sorted_hsml, self.sorted_qty)
        com, qty, cov, cell_hsml = moments.derive()
        finest.update({
            "moments": moments, "mass": moments.mass, "com": com,
            "hsml": cell_hsml, "qty": qty, "cov": cov,
            "mxx": moments.mxx, "mh2": moments.mh2,
        })

        # Coarser levels: sum children's moments
        for li in range(1, len(self.levels)):
            child_lv = self.levels[li - 1]
            lv = self.levels[li]
            n_nodes = lv["nc"]
            child_moments = child_lv["moments"]

            p_mass, p_mp, p_mq, p_mh2, p_mxx = _coarsen_from_children(
                lv["child_offset"], n_nodes,
                child_moments.mass, child_moments.mp, child_moments.mq,
                child_moments.mh2, child_moments.mxx,
                child_lv["nc"])

            moments = CellMoments(n_nodes)
            moments.mass = p_mass
            moments.mp = p_mp
            moments.mq = p_mq
            moments.mh2 = p_mh2
            moments.mxx = p_mxx
            com, qty, cov, cell_hsml = moments.derive()

            lv.update({
                "moments": moments, "mass": moments.mass, "com": com,
                "hsml": cell_hsml, "qty": qty, "cov": cov,
                "mxx": moments.mxx, "mh2": moments.mh2,
            })

    def _compute_nn_hsml(self, n_ngb=32):
        """Precompute neighbor-based smoothing lengths for all levels."""
        from scipy.spatial import cKDTree

        for lv in self.levels:
            n = lv["nc"]
            if n <= 1:
                lv["hsml_nn"] = np.full(n, lv["half_diag"] * 2, dtype=np.float32)
                continue
            mask = lv["mass"] > 0
            n_active = mask.sum()
            if n_active <= n_ngb:
                lv["hsml_nn"] = np.full(n, lv["half_diag"] * 2, dtype=np.float32)
                continue
            centers = lv["com"][mask]
            k = min(n_ngb + 1, n_active)
            tree = cKDTree(centers)
            dist, _ = tree.query(centers, k=k)
            nn_dist = dist[:, -1].astype(np.float32)
            hsml_nn = np.full(n, lv["half_diag"] * 2, dtype=np.float32)
            hsml_nn[mask] = nn_dist
            lv["hsml_nn"] = hsml_nn

    def set_nn_hsml(self, enabled):
        """Swap between covariance-based and NN-based smoothing lengths."""
        for lv in self.levels:
            if enabled and "hsml_nn" in lv:
                if "hsml_cov" not in lv:
                    lv["hsml_cov"] = lv["hsml"].copy()
                lv["hsml"] = lv["hsml_nn"].copy()
            elif "hsml_cov" in lv:
                lv["hsml"] = lv["hsml_cov"].copy()

    def update_weights(self, masses, quantity=None):
        """Re-weight the tree without rebuilding structure."""
        if quantity is None:
            quantity = masses
        so = self.sort_order
        self.sorted_mass = masses[so].astype(np.float32)
        self.sorted_qty = quantity[so].astype(np.float32)
        self._compute_all_moments()

    # ---- CPU query (same interface as SpatialGrid) ----

    def query_frustum_lod(self, camera, max_particles, lod_pixels=4,
                          importance_sampling=False, viewport_width=2048,
                          summary_overlap=0.0):
        """Top-down multi-level LOD query. Returns same format as SpatialGrid."""
        if lod_pixels <= 2 or len(self.levels) <= 1:
            return self._frustum_cull_finest(camera, max_particles, importance_sampling)

        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        pix_per_rad = viewport_width / (2.0 * np.tan(fov_rad / 2))
        half_tan = np.tan(fov_rad / 2)
        summary_parts = []
        real_pos = real_hsml = real_mass = real_qty = None
        n_visible_real = 0

        # Start from coarsest level
        lv = self.levels[-1]
        centers = lv["centers"]
        hd = lv["half_diag"]

        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        cell_extent = hd + np.minimum(lv["hsml"], 2 * hd)
        front_depth = np.maximum(depths + cell_extent, 0)
        lim_h = front_depth * half_tan * camera.aspect + cell_extent
        lim_v = front_depth * half_tan + cell_extent
        in_front = depths > -cell_extent
        has_mass = lv["mass"] > 0
        visible = has_mass & in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        dist = np.sqrt(depths**2 + rights**2 + ups**2)
        safe_dist = np.maximum(dist, 0.01)
        h_pix = lv["hsml"] / safe_dist * pix_per_rad

        # Always refine small coarsest levels
        if lv["nc"] <= 8:
            refine_mask = visible & has_mass
        else:
            summary_mask = visible & has_mass & (h_pix <= lod_pixels)
            refine_mask = visible & has_mass & (h_pix > lod_pixels)
            s_idx = np.where(summary_mask)[0]
            if len(s_idx) > 0:
                summary_parts.append((
                    lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx],
                    lv["qty"][s_idx], lv["cov"][s_idx],
                    lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"],
                ))

        refine_cells = np.where(refine_mask)[0]

        # Traverse finer levels
        for li in range(len(self.levels) - 2, -1, -1):
            if len(refine_cells) == 0:
                break

            lv = self.levels[li]
            parent_lv = self.levels[li + 1]
            hd = lv["half_diag"]

            # Expand parents to children using child_offset
            child_indices = []
            for pi in refine_cells:
                cs = parent_lv["child_offset"][pi]
                ce = parent_lv["child_offset"][pi + 1] if pi < parent_lv["nc"] - 1 else lv["nc"]
                for ci in range(cs, ce):
                    child_indices.append(ci)

            if not child_indices:
                break
            child_flat = np.array(child_indices, dtype=np.int64)

            valid = lv["mass"][child_flat] > 0
            child_flat = child_flat[valid]
            if len(child_flat) == 0:
                break

            centers = lv["centers"][child_flat]
            diff = centers - cam_pos
            dist = np.sqrt((diff * diff).sum(axis=1))
            safe_dist = np.maximum(dist, 0.01)
            h_pix = lv["hsml"][child_flat] / safe_dist * pix_per_rad

            small = h_pix <= lod_pixels
            large = ~small

            s_idx = child_flat[small]
            if len(s_idx) > 0:
                summary_parts.append((
                    lv["com"][s_idx], lv["hsml"][s_idx], lv["mass"][s_idx],
                    lv["qty"][s_idx], lv["cov"][s_idx],
                    lv["mh2"][s_idx] / np.maximum(lv["mass"][s_idx], 1e-30), lv["cs"],
                ))

            if li == 0:
                vis_cells = child_flat[large]
                if len(vis_cells) > 0:
                    n_summaries = sum(p[0].shape[0] for p in summary_parts) if summary_parts else 0
                    budget = max(max_particles - n_summaries, max_particles // 2)
                    if importance_sampling:
                        vis_depths = safe_dist[large].astype(np.float64)
                        vis_cell_h = lv["hsml"][child_flat[large]].astype(np.float64)
                        real_pos, real_hsml, real_mass, real_qty, n_visible_real = _gather_importance_sampled(
                            self.cell_start, vis_cells.astype(np.int64), budget,
                            self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                            vis_depths, vis_cell_h)
                    else:
                        real_pos, real_hsml, real_mass, real_qty, n_visible_real = _gather_subsampled_direct(
                            self.cell_start, vis_cells.astype(np.int64), budget,
                            self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty)
            else:
                refine_cells = child_flat[large]

        # Assemble output
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros(0, dtype=np.float32)
        z6 = np.zeros((0, 6), dtype=np.float32)

        n_visible_real = int(n_visible_real)
        r_pos = real_pos if real_pos is not None else z3
        r_hsml = real_hsml if real_hsml is not None else z1
        r_mass = real_mass if real_mass is not None else z1
        r_qty = real_qty if real_qty is not None else z1

        n_sampled = len(r_pos)
        if not importance_sampling and n_sampled > 0 and n_visible_real > n_sampled:
            ratio = n_visible_real / n_sampled
            r_mass = r_mass * ratio
            r_hsml = r_hsml * (ratio ** (1.0 / 3.0))

        if summary_parts:
            s_pos = np.concatenate([p[0] for p in summary_parts]).astype(np.float32)
            s_hsml = np.concatenate([p[1] for p in summary_parts]).astype(np.float32)
            s_mass = np.concatenate([p[2] for p in summary_parts]).astype(np.float32)
            s_qty = np.concatenate([p[3] for p in summary_parts]).astype(np.float32)
            s_cov = np.concatenate([p[4] for p in summary_parts]).astype(np.float32)
            s_mean_h2 = np.concatenate([p[5] for p in summary_parts]).astype(np.float32)
            s_cs2 = np.concatenate([
                np.broadcast_to((p[6]**2)[None, :], (len(p[0]), 3))
                for p in summary_parts
            ]).astype(np.float32)
            s_cov[:, 0] += 0.225 * s_mean_h2
            s_cov[:, 3] += 0.225 * s_mean_h2
            s_cov[:, 5] += 0.225 * s_mean_h2
            alpha = summary_overlap
            s_cov[:, 0] += alpha * s_cs2[:, 0]
            s_cov[:, 3] += alpha * s_cs2[:, 1]
            s_cov[:, 5] += alpha * s_cs2[:, 2]
        else:
            s_pos, s_hsml, s_mass, s_qty, s_cov = z3, z1, z1, z1, z6

        return (
            r_pos.astype(np.float32), r_hsml.astype(np.float32),
            r_mass.astype(np.float32), r_qty.astype(np.float32),
            s_pos, s_hsml, s_mass, s_qty, s_cov,
        )

    def _frustum_cull_finest(self, camera, max_particles, importance_sampling=False):
        """Fast path: frustum cull directly on leaf cells."""
        finest = self.levels[0]
        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up
        fov_rad = np.radians(camera.fov)
        half_tan = np.tan(fov_rad / 2)
        hd = finest["half_diag"]

        centers = finest["centers"]
        depths = centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = centers @ cam_up - np.dot(cam_pos, cam_up)

        cell_extent = hd + np.minimum(finest["hsml"], 2 * hd)
        front_depth = np.maximum(depths + cell_extent, 0)
        lim_h = front_depth * half_tan * camera.aspect + cell_extent
        lim_v = front_depth * half_tan + cell_extent
        in_front = depths > -cell_extent
        has_mass = finest["mass"] > 0
        visible = has_mass & in_front & (np.abs(rights) < lim_h) & (np.abs(ups) < lim_v)

        vis_cells = np.where(visible)[0]
        if len(vis_cells) == 0:
            z3 = np.zeros((0, 3), dtype=np.float32)
            z1 = np.zeros(0, dtype=np.float32)
            return z3, z1, z1, z1

        if importance_sampling:
            dist = np.sqrt(depths[visible]**2 + rights[visible]**2 + ups[visible]**2)
            safe_dist = np.maximum(dist, 0.01).astype(np.float64)
            vis_cell_h = finest["hsml"][vis_cells].astype(np.float64)
            pos, hsml, mass, qty, n_vis = _gather_importance_sampled(
                self.cell_start, vis_cells.astype(np.int64), max_particles,
                self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty,
                safe_dist, vis_cell_h)
        else:
            pos, hsml, mass, qty, n_vis = _gather_subsampled_direct(
                self.cell_start, vis_cells.astype(np.int64), max_particles,
                self.sorted_pos, self.sorted_hsml, self.sorted_mass, self.sorted_qty)

        n_vis = int(n_vis)
        n_sampled = len(pos)
        if not importance_sampling and n_sampled > 0 and n_vis > n_sampled:
            ratio = n_vis / n_sampled
            mass = mass * ratio
            hsml = hsml * (ratio ** (1.0 / 3.0))

        return (
            pos.astype(np.float32), hsml.astype(np.float32),
            mass.astype(np.float32), qty.astype(np.float32),
        )
