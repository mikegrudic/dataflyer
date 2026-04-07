"""Tests for adaptive octree moment computation.

Walks the tree top-down and verifies that every node's stored moments
match the direct computation from its children (internal nodes) or
from its particles (leaf nodes).
"""

from pathlib import Path

import numpy as np
import pytest
from dataflyer.adaptive_octree import AdaptiveOctree

SNAPSHOT_PATH = Path(__file__).parent.parent / "snapshot_600.hdf5"


def _make_test_data(n=50_000, seed=42, clustered=True):
    """Generate test particle data with a mix of dense and sparse regions."""
    rng = np.random.default_rng(seed)
    if clustered:
        n_cluster = int(n * 0.6)
        n_spread = n - n_cluster
        pos_cluster = rng.normal(0.5, 0.02, (n_cluster, 3)).astype(np.float32)
        pos_spread = rng.uniform(0, 1, (n_spread, 3)).astype(np.float32)
        pos = np.vstack([pos_cluster, pos_spread])
    else:
        pos = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, n).astype(np.float32)
    hsml = rng.uniform(0.001, 0.02, n).astype(np.float32)
    qty = rng.uniform(-1, 5, n).astype(np.float32)
    return pos, mass, hsml, qty


def _direct_moments_from_particles(cell_start, leaf_idx, sorted_pos, sorted_mass,
                                    sorted_hsml, sorted_qty):
    """Compute moments directly from sorted particle arrays for a single leaf."""
    s = cell_start[leaf_idx]
    e = cell_start[leaf_idx + 1]
    m = sorted_mass[s:e].astype(np.float64)
    p = sorted_pos[s:e].astype(np.float64)
    h = sorted_hsml[s:e].astype(np.float64)
    q = sorted_qty[s:e].astype(np.float64)

    M = m.sum()
    mp = (m[:, None] * p).sum(axis=0)
    mq = (m * q).sum()
    mh2 = (m * h * h).sum()
    mxx = np.zeros(6, dtype=np.float64)
    mxx[0] = (m * p[:, 0] * p[:, 0]).sum()
    mxx[1] = (m * p[:, 0] * p[:, 1]).sum()
    mxx[2] = (m * p[:, 0] * p[:, 2]).sum()
    mxx[3] = (m * p[:, 1] * p[:, 1]).sum()
    mxx[4] = (m * p[:, 1] * p[:, 2]).sum()
    mxx[5] = (m * p[:, 2] * p[:, 2]).sum()

    return M, mp, mq, mh2, mxx


def _sum_children_moments(parent_lv, child_lv, parent_idx):
    """Sum child moments for a single parent node."""
    co = parent_lv["child_offset"]
    nc_child = child_lv["nc"]
    cs = co[parent_idx]
    ce = co[parent_idx + 1] if parent_idx < parent_lv["nc"] - 1 else nc_child
    cm = child_lv["moments"]

    M = cm.mass[cs:ce].astype(np.float64).sum()
    mp = cm.mp[cs:ce].astype(np.float64).sum(axis=0)
    mq = cm.mq[cs:ce].astype(np.float64).sum()
    mh2 = cm.mh2[cs:ce].astype(np.float64).sum()
    mxx = cm.mxx[cs:ce].astype(np.float64).sum(axis=0)

    return M, mp, mq, mh2, mxx


def _derive_cov(M, mp, mxx):
    """Compute covariance from extensive moments."""
    safe = max(M, 1e-30)
    com = mp / safe
    cov = np.zeros(6, dtype=np.float64)
    cov[0] = mxx[0] / safe - com[0] * com[0]
    cov[1] = mxx[1] / safe - com[0] * com[1]
    cov[2] = mxx[2] / safe - com[0] * com[2]
    cov[3] = mxx[3] / safe - com[1] * com[1]
    cov[4] = mxx[4] / safe - com[1] * com[2]
    cov[5] = mxx[5] / safe - com[2] * com[2]
    return com, cov


class TestAdaptiveOctreeMoments:

    @pytest.fixture(params=["clustered", "uniform"])
    def tree(self, request):
        clustered = request.param == "clustered"
        pos, mass, hsml, qty = _make_test_data(n=50_000, clustered=clustered)
        return AdaptiveOctree(pos, mass, hsml, qty, leaf_size=256)

    def test_leaf_moments_match_particles(self, tree):
        """Verify every leaf node's stored moments match direct particle sums."""
        rtol = 1e-4
        for lv in tree.levels:
            leaf_mask = np.where(lv["is_leaf"])[0]
            for node_idx in leaf_mask:
                li = lv["leaf_indices"][node_idx]
                M_direct, mp_direct, mq_direct, mh2_direct, mxx_direct = \
                    _direct_moments_from_particles(
                        tree.cell_start, li,
                        tree.sorted_pos, tree.sorted_mass,
                        tree.sorted_hsml, tree.sorted_qty)

                mom = lv["moments"]
                M_stored = float(mom.mass[node_idx])
                mp_stored = mom.mp[node_idx].astype(np.float64)
                mq_stored = float(mom.mq[node_idx])
                mh2_stored = float(mom.mh2[node_idx])
                mxx_stored = mom.mxx[node_idx].astype(np.float64)

                assert M_stored == pytest.approx(M_direct, rel=rtol), \
                    f"mass mismatch at depth {lv['depth']} node {node_idx}: {M_stored} vs {M_direct}"
                np.testing.assert_allclose(mp_stored, mp_direct, rtol=rtol,
                    err_msg=f"mp mismatch at depth {lv['depth']} node {node_idx}")
                assert mq_stored == pytest.approx(mq_direct, rel=rtol), \
                    f"mq mismatch at depth {lv['depth']} node {node_idx}"
                assert mh2_stored == pytest.approx(mh2_direct, rel=rtol), \
                    f"mh2 mismatch at depth {lv['depth']} node {node_idx}"
                np.testing.assert_allclose(mxx_stored, mxx_direct, rtol=rtol,
                    err_msg=f"mxx mismatch at depth {lv['depth']} node {node_idx}")

    def test_internal_moments_match_children(self, tree):
        """Verify every internal node's moments equal the sum of its children."""
        rtol = 1e-4
        for li in range(1, len(tree.levels)):
            lv = tree.levels[li]
            child_lv = tree.levels[li - 1]
            internal_mask = np.where(~lv["is_leaf"])[0]
            for node_idx in internal_mask:
                M_sum, mp_sum, mq_sum, mh2_sum, mxx_sum = \
                    _sum_children_moments(lv, child_lv, node_idx)

                mom = lv["moments"]
                M_stored = float(mom.mass[node_idx])
                mp_stored = mom.mp[node_idx].astype(np.float64)
                mq_stored = float(mom.mq[node_idx])
                mh2_stored = float(mom.mh2[node_idx])
                mxx_stored = mom.mxx[node_idx].astype(np.float64)

                assert M_stored == pytest.approx(M_sum, rel=rtol), \
                    f"mass mismatch at depth {lv['depth']} internal {node_idx}: {M_stored} vs {M_sum}"
                np.testing.assert_allclose(mp_stored, mp_sum, rtol=rtol,
                    err_msg=f"mp mismatch at depth {lv['depth']} internal {node_idx}")
                assert mq_stored == pytest.approx(mq_sum, rel=rtol), \
                    f"mq mismatch at depth {lv['depth']} internal {node_idx}"
                assert mh2_stored == pytest.approx(mh2_sum, rel=rtol), \
                    f"mh2 mismatch at depth {lv['depth']} internal {node_idx}"
                np.testing.assert_allclose(mxx_stored, mxx_sum, rtol=rtol,
                    err_msg=f"mxx mismatch at depth {lv['depth']} internal {node_idx}")

    def test_covariance_matches_moments(self, tree):
        """Verify stored cov = <xx> - <x><x>^T derived from extensive moments."""
        for lv in tree.levels:
            mom = lv["moments"]
            for node_idx in range(lv["nc"]):
                if mom.mass[node_idx] <= 0:
                    continue
                com_derived, cov_derived = _derive_cov(
                    float(mom.mass[node_idx]),
                    mom.mp[node_idx].astype(np.float64),
                    mom.mxx[node_idx].astype(np.float64))

                com_stored = lv["com"][node_idx].astype(np.float64)
                cov_stored = lv["cov"][node_idx].astype(np.float64)

                np.testing.assert_allclose(com_stored, com_derived, rtol=1e-4,
                    err_msg=f"com mismatch at depth {lv['depth']} node {node_idx}")
                # float32 derive: <xx>/M - <x/M>^2 loses precision on off-diagonals.
                # Absolute tolerance from float32 noise on the <xx> terms.
                mxx_scale = max(abs(mom.mxx[node_idx]).max() / max(mom.mass[node_idx], 1e-30), 1e-30)
                atol = 1e-6 * mxx_scale  # ~float32 eps relative to <xx>/M
                np.testing.assert_allclose(cov_stored, cov_derived,
                    atol=atol, rtol=0,
                    err_msg=f"cov mismatch at depth {lv['depth']} node {node_idx}")

    def test_covariance_positive_semidefinite(self, tree):
        """Verify all stored covariance matrices are positive semi-definite."""
        for lv in tree.levels:
            for node_idx in range(lv["nc"]):
                if lv["moments"].mass[node_idx] <= 0:
                    continue
                c = lv["cov"][node_idx].astype(np.float64)
                mat = np.array([[c[0], c[1], c[2]],
                                [c[1], c[3], c[4]],
                                [c[2], c[4], c[5]]])
                evals = np.linalg.eigvalsh(mat)
                trace = mat.trace()
                if trace < 1e-20:
                    continue  # near-empty node, skip
                # float32 cancellation in <xx>-<x><x>^T can produce small negatives
                assert evals.min() >= -1e-3 * trace, \
                    f"negative eigenvalue {evals.min():.3e} (trace={trace:.3e}) at depth {lv['depth']} node {node_idx}"

    def test_root_mass_equals_total(self, tree):
        """Root node mass should equal the sum of all particle masses."""
        root = tree.levels[-1]
        assert root["nc"] == 1
        total_mass = tree.sorted_mass.astype(np.float64).sum()
        root_mass = float(root["moments"].mass[0])
        assert root_mass == pytest.approx(total_mass, rel=1e-3)

    def test_npart_consistency(self, tree):
        """Verify npart propagation: root npart == total particles,
        internal npart == sum of children npart."""
        n_total = len(tree.sorted_pos)
        root = tree.levels[-1]
        assert int(root["npart"][0]) == n_total

        for li in range(1, len(tree.levels)):
            lv = tree.levels[li]
            child_lv = tree.levels[li - 1]
            co = lv["child_offset"]
            internal_mask = np.where(~lv["is_leaf"])[0]
            for pi in internal_mask:
                cs = co[pi]
                ce = co[pi + 1] if pi < lv["nc"] - 1 else child_lv["nc"]
                child_sum = child_lv["npart"][cs:ce].sum()
                assert int(lv["npart"][pi]) == int(child_sum), \
                    f"npart mismatch at depth {lv['depth']} node {pi}: {lv['npart'][pi]} vs {child_sum}"

    def test_all_particles_accounted(self, tree):
        """Every particle should belong to exactly one leaf cell."""
        n = len(tree.sorted_pos)
        covered = np.zeros(n, dtype=bool)
        for i in range(len(tree.cell_start) - 1):
            s, e = tree.cell_start[i], tree.cell_start[i + 1]
            assert not covered[s:e].any(), f"overlap in leaf {i}"
            covered[s:e] = True
        assert covered.all(), f"{(~covered).sum()} particles not in any leaf"

    def test_leaf_size_respected(self, tree):
        """No leaf should exceed leaf_size (except at max_depth)."""
        for i in range(len(tree.cell_start) - 1):
            count = tree.cell_start[i + 1] - tree.cell_start[i]
            if count > tree.leaf_size:
                # Find this leaf's depth — it must be at max_depth
                found = False
                for lv in tree.levels:
                    mask = lv["is_leaf"] & (lv["leaf_indices"] == i)
                    if mask.any():
                        assert lv["depth"] == tree.max_depth, \
                            f"leaf {i} has {count} particles at depth {lv['depth']} (not max_depth={tree.max_depth})"
                        found = True
                        break
                assert found, f"leaf {i} with {count} particles not found in any level"


@pytest.fixture(scope="module")
def snapshot_tree():
    """Build adaptive octree from the real snapshot (skipped if file missing)."""
    pytest.importorskip("h5py")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"snapshot not found: {SNAPSHOT_PATH}")
    import h5py
    with h5py.File(SNAPSHOT_PATH, "r") as f:
        pos = f["PartType0/Coordinates"][:].astype(np.float32)
        mass = f["PartType0/Masses"][:].astype(np.float32)
        hsml = f["PartType0/SmoothingLength"][:].astype(np.float32)
    return AdaptiveOctree(pos, mass, hsml, mass.copy(), leaf_size=1024)


class TestSnapshotOctree:
    """Moment-verification tests on the real snapshot."""

    def test_root_mass(self, snapshot_tree):
        root = snapshot_tree.levels[-1]
        total = snapshot_tree.sorted_mass.astype(np.float64).sum()
        assert float(root["moments"].mass[0]) == pytest.approx(total, rel=1e-3)

    def test_all_particles_accounted(self, snapshot_tree):
        n = len(snapshot_tree.sorted_pos)
        assert snapshot_tree.cell_start[-1] == n
        covered = np.zeros(n, dtype=bool)
        for i in range(len(snapshot_tree.cell_start) - 1):
            s, e = snapshot_tree.cell_start[i], snapshot_tree.cell_start[i + 1]
            assert not covered[s:e].any(), f"overlap in leaf {i}"
            covered[s:e] = True
        assert covered.all()

    def test_npart_consistency(self, snapshot_tree):
        tree = snapshot_tree
        n_total = len(tree.sorted_pos)
        root = tree.levels[-1]
        assert int(root["npart"][0]) == n_total
        for li in range(1, len(tree.levels)):
            lv = tree.levels[li]
            child_lv = tree.levels[li - 1]
            co = lv["child_offset"]
            for pi in np.where(~lv["is_leaf"])[0]:
                cs = co[pi]
                ce = co[pi + 1] if pi < lv["nc"] - 1 else child_lv["nc"]
                assert int(lv["npart"][pi]) == int(child_lv["npart"][cs:ce].sum())

    def test_internal_moments_match_children(self, snapshot_tree):
        """Spot-check internal moments on 100 random nodes per level."""
        tree = snapshot_tree
        rng = np.random.default_rng(0)
        rtol = 1e-4
        for li in range(1, len(tree.levels)):
            lv = tree.levels[li]
            child_lv = tree.levels[li - 1]
            internal = np.where(~lv["is_leaf"])[0]
            if len(internal) == 0:
                continue
            sample = rng.choice(internal, size=min(100, len(internal)), replace=False)
            for node_idx in sample:
                M_sum, mp_sum, mq_sum, mh2_sum, mxx_sum = \
                    _sum_children_moments(lv, child_lv, node_idx)
                mom = lv["moments"]
                assert float(mom.mass[node_idx]) == pytest.approx(M_sum, rel=rtol)
                np.testing.assert_allclose(
                    mom.mp[node_idx].astype(np.float64), mp_sum, rtol=rtol)
                np.testing.assert_allclose(
                    mom.mxx[node_idx].astype(np.float64), mxx_sum, rtol=rtol)

    def test_leaf_moments_spot_check(self, snapshot_tree):
        """Spot-check 200 random leaf nodes against direct particle sums."""
        tree = snapshot_tree
        rng = np.random.default_rng(1)
        rtol = 1e-4
        all_leaves = []
        for lv in tree.levels:
            for ni in np.where(lv["is_leaf"])[0]:
                all_leaves.append((lv, ni))
        sample_idx = rng.choice(len(all_leaves), size=min(200, len(all_leaves)), replace=False)
        for idx in sample_idx:
            lv, node_idx = all_leaves[idx]
            li = lv["leaf_indices"][node_idx]
            M_d, mp_d, mq_d, mh2_d, mxx_d = _direct_moments_from_particles(
                tree.cell_start, li,
                tree.sorted_pos, tree.sorted_mass, tree.sorted_hsml, tree.sorted_qty)
            mom = lv["moments"]
            assert float(mom.mass[node_idx]) == pytest.approx(M_d, rel=rtol)
            np.testing.assert_allclose(
                mom.mp[node_idx].astype(np.float64), mp_d, rtol=rtol)
            np.testing.assert_allclose(
                mom.mxx[node_idx].astype(np.float64), mxx_d, rtol=rtol)
