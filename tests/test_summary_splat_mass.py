"""Test that anisotropic summary splats conserve mass vs individual particles.

Generates particles from randomly-oriented triaxial Gaussians, renders
surface density at full resolution (all particles) using the wgpu renderer,
then renders using the tree's summary splat for the parent node. Compares
total mass and per-pixel shape.
"""

import os
import numpy as np
import pytest
import wgpu


def _random_rotation(rng):
    """Random 3x3 rotation matrix via QR decomposition."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_renderer_and_camera(center, camera_distance, res=256, fov=90):
    """Create a headless WGPURenderer + Camera pointing at center."""
    from dataflyer.wgpu_renderer import WGPURenderer
    from dataflyer.colormaps import colormap_to_texture_data
    from dataflyer.camera import Camera

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    features = []
    if "float32-blendable" in adapter.features:
        features.append("float32-blendable")
    device = adapter.request_device_sync(required_features=features)

    renderer = WGPURenderer(device, canvas_context=None)
    renderer.set_colormap(colormap_to_texture_data("magma"))
    renderer.resolve_mode = 0
    renderer.log_scale = 0
    renderer.use_tree = False
    renderer.bypass_cull = True

    camera = Camera(fov=fov, aspect=1.0)
    camera.position = np.array(
        [center[0], center[1], center[2] + camera_distance], dtype=np.float32)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)
    camera.near = camera_distance * 0.01
    camera.far = camera_distance * 10

    return renderer, camera, device


def _render_particles(renderer, camera, pos, masses, hsml, res):
    """Upload particles and render, return denominator (surface density) map."""
    renderer._upload_arrays(
        pos.astype(np.float32), hsml.astype(np.float32),
        masses.astype(np.float32), masses.astype(np.float32), camera)
    renderer._upload_aniso_summaries(
        np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
        np.zeros(0, np.float32), np.zeros((0, 6), np.float32))
    renderer._ensure_fbo(res, res, which=1)
    renderer._render_accum(camera, res, res, renderer._accum_textures)
    den = renderer._read_accum_texture_r(renderer._accum_textures["textures"][1])
    return den.reshape(res, res)


def _render_summary(renderer, camera, com, mass_total, qty_avg, cov, res):
    """Render a single anisotropic summary splat, return denominator map."""
    renderer.n_particles = 0
    renderer._particle_bufs = {}
    renderer._upload_aniso_summaries(
        com.reshape(1, 3).astype(np.float32),
        np.array([mass_total], dtype=np.float32),
        np.array([qty_avg], dtype=np.float32),
        cov.reshape(1, 6).astype(np.float32))
    renderer._ensure_fbo(res, res, which=1)
    renderer._render_accum(camera, res, res, renderer._accum_textures)
    den = renderer._read_accum_texture_r(renderer._accum_textures["textures"][1])
    return den.reshape(res, res)


def _generate_triaxial(rng, center=None):
    """Generate one random triaxial Gaussian particle set."""
    from meshoid import Meshoid

    N = int(rng.integers(100, 10_000))
    log_sigmas = rng.uniform(-2.5, -0.5, 3)
    log_sigmas.sort()
    sigmas = 10.0 ** log_sigmas
    R = _random_rotation(rng)
    cov_3d = R @ np.diag(sigmas**2) @ R.T

    if center is None:
        center = np.array([0.5, 0.5, 0.5])
    pos = rng.multivariate_normal(center, cov_3d, N).astype(np.float64)
    masses = rng.uniform(0.5, 2.0, N).astype(np.float64)
    masses /= masses.sum()
    hsml = Meshoid(pos).SmoothingLength()

    return pos, masses, hsml, center, sigmas


def _compare_one(renderer, camera, res, pos, masses, hsml, center):
    """Render particles vs summary, return mass ratio and maps."""
    from dataflyer.adaptive_octree import AdaptiveOctree

    sigma_particles = _render_particles(renderer, camera, pos, masses, hsml, res)

    tree = AdaptiveOctree(
        pos.astype(np.float32), masses.astype(np.float32),
        hsml.astype(np.float32), masses.astype(np.float32),
        leaf_size=len(pos) + 1)

    root = tree.levels[-1]
    com = root["com"][0]
    mass_total = float(root["mass"][0])
    qty_avg = float(root["qty"][0])
    cov = root["cov"][0].copy().astype(np.float64)
    mean_h2 = float(root["mh2"][0]) / max(mass_total, 1e-30)
    cov[0] += 0.225 * mean_h2
    cov[3] += 0.225 * mean_h2
    cov[5] += 0.225 * mean_h2

    sigma_summary = _render_summary(
        renderer, camera, com, mass_total, qty_avg,
        cov.astype(np.float32), res)

    total_p = float(sigma_particles.sum())
    total_s = float(sigma_summary.sum())
    ratio = total_s / total_p if total_p > 0 else 0.0
    return ratio, total_p, total_s, sigma_particles, sigma_summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def triaxial_data():
    """Single triaxial Gaussian for the basic test."""
    pytest.importorskip("meshoid")
    rng = np.random.default_rng(123)
    pos, masses, hsml, center, _ = _generate_triaxial(rng)
    return pos, masses, hsml, center


def test_summary_conserves_mass(triaxial_data):
    """Total mass of summary should match particles for one triaxial."""
    pos, masses, hsml, center = triaxial_data
    res = 256

    renderer, camera, device = _make_renderer_and_camera(center, 1.0, res)
    ratio, total_p, total_s, sigma_p, sigma_s = _compare_one(
        renderer, camera, res, pos, masses, hsml, center)
    renderer.release()

    print(f"\nParticle total: {total_p:.6g}")
    print(f"Summary total:  {total_s:.6g}")
    print(f"Ratio: {ratio:.4f}")

    assert total_p > 0, "particle rendering produced empty map"
    assert total_s > 0, "summary rendering produced empty map"
    assert 0.8 < ratio < 1.2, (
        f"Mass conservation failed: ratio={ratio:.3f}")

    _save_comparison(sigma_p, sigma_s, ratio, os.path.dirname(__file__))


def test_summary_mass_100_triaxials():
    """Mass conservation for 100 overlapping triaxial Gaussians at various LODs.

    Builds all blobs into a single adaptive octree, then for each LOD level
    queries the tree (mix of emitted particles + summaries) and renders.
    Compares total mass and per-pixel correlation against full particle render.
    """
    pytest.importorskip("meshoid")
    from meshoid import Meshoid
    from dataflyer.adaptive_octree import AdaptiveOctree
    from dataflyer.camera import Camera

    rng = np.random.default_rng(42)
    scene_center = np.array([0.5, 0.5, 0.5])
    res = 256

    # Generate 100 triaxial Gaussians
    all_pos = []
    all_mass = []
    all_hsml = []

    for i in range(100):
        N = int(rng.integers(50, 2000))
        log_sigmas = rng.uniform(-2.5, -1.0, 3)
        log_sigmas.sort()
        sigmas = 10.0 ** log_sigmas
        R = _random_rotation(rng)
        cov_3d = R @ np.diag(sigmas**2) @ R.T

        blob_center = scene_center + rng.uniform(-0.3, 0.3, 3)
        pos = rng.multivariate_normal(blob_center, cov_3d, N).astype(np.float64)
        masses = rng.uniform(0.5, 2.0, N).astype(np.float64)
        masses /= masses.sum()
        hsml = Meshoid(pos).SmoothingLength()

        all_pos.append(pos)
        all_mass.append(masses)
        all_hsml.append(hsml)

    all_pos = np.concatenate(all_pos)
    all_mass = np.concatenate(all_mass)
    all_hsml = np.concatenate(all_hsml)
    n_total = len(all_pos)
    print(f"\nScene: 100 blobs, {n_total} total particles")

    # Build one tree
    tree = AdaptiveOctree(
        all_pos.astype(np.float32), all_mass.astype(np.float32),
        all_hsml.astype(np.float32), all_mass.astype(np.float32),
        leaf_size=1024)
    print(f"Tree: {len(tree.levels)} levels, {len(tree.cell_start)-1} leaves")

    # Camera
    camera = Camera(fov=90, aspect=1.0)
    camera.position = np.array(
        [scene_center[0], scene_center[1], scene_center[2] + 1.5],
        dtype=np.float32)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)
    camera.near = 0.015
    camera.far = 15.0

    # Renderer
    renderer, camera_r, device = _make_renderer_and_camera(scene_center, 1.5, res)

    # Reference: full particle render
    sigma_ref = _render_particles(renderer, camera_r, all_pos, all_mass, all_hsml, res)
    total_ref = float(sigma_ref.sum())
    assert total_ref > 0, "reference render produced empty map"

    # Sweep LODs: from fully summarized to nearly full resolution
    lod_values = [99999, 256, 64, 16, 4, 1]
    results = []

    print(f"\n{'LOD':>8} {'emitted':>8} {'summaries':>10} {'ratio':>8} {'corr':>8}")
    print("-" * 52)

    for lod in lod_values:
        result = tree.query_frustum_lod(
            camera, max_particles=n_total,
            lod_pixels=lod, viewport_width=res, summary_overlap=0.0)

        if len(result) == 9:
            r_pos, r_hsml, r_mass, r_qty, s_pos, s_hsml, s_mass, s_qty, s_cov = result
        else:
            r_pos, r_hsml, r_mass, r_qty = result
            s_pos = np.zeros((0, 3), np.float32)
            s_mass = np.zeros(0, np.float32)
            s_qty = np.zeros(0, np.float32)
            s_cov = np.zeros((0, 6), np.float32)

        # Render emitted particles + summary splats together
        if len(r_pos) > 0:
            renderer._upload_arrays(
                r_pos, r_hsml, r_mass, r_qty, camera_r)
        else:
            renderer.n_particles = 0
            renderer._particle_bufs = {}

        if len(s_pos) > 0:
            renderer._upload_aniso_summaries(s_pos, s_mass, s_qty, s_cov)
        else:
            renderer._upload_aniso_summaries(
                np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), np.zeros((0, 6), np.float32))

        renderer._ensure_fbo(res, res, which=1)
        renderer._render_accum(camera_r, res, res, renderer._accum_textures)
        sigma_lod = renderer._read_accum_texture_r(
            renderer._accum_textures["textures"][1]).reshape(res, res)

        total_lod = float(sigma_lod.sum())
        mass_ratio = total_lod / total_ref if total_ref > 0 else 0.0

        mask = (sigma_ref > 0) & (sigma_lod > 0)
        if mask.sum() > 100:
            corr = float(np.corrcoef(
                np.log10(sigma_ref[mask]),
                np.log10(sigma_lod[mask]))[0, 1])
        else:
            corr = float('nan')

        print(f"{lod:>8} {len(r_pos):>8} {len(s_pos):>10} "
              f"{mass_ratio:>8.4f} {corr:>8.4f}")
        results.append((lod, len(r_pos), len(s_pos), mass_ratio, corr, sigma_lod))

    renderer.release()

    # Assert mass conservation at all LODs
    for lod, n_emit, n_summ, ratio, corr, _ in results:
        assert 0.7 < ratio < 1.3, (
            f"Mass conservation failed at lod={lod}: ratio={ratio:.3f}")

    # Save multi-panel comparison
    _save_lod_sweep(sigma_ref, results, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_comparison(sigma_p, sigma_s, mass_ratio, out_dir,
                     filename="summary_splat_comparison.png", title_prefix=""):
    """Save 3-panel comparison: particles | summary | log ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    pos_p = sigma_p[sigma_p > 0]
    pos_s = sigma_s[sigma_s > 0]
    if len(pos_p) == 0 or len(pos_s) == 0:
        return
    vmin = min(pos_p.min(), pos_s.min())
    vmax = max(pos_p.max(), pos_s.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(sigma_p, norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap="magma", origin="lower", interpolation="nearest")
    axes[0].set_title("Particles (full resolution)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sigma_s, norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap="magma", origin="lower", interpolation="nearest")
    axes[1].set_title("Summary splat (1 aniso Gaussian)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (sigma_p > 0) & (sigma_s > 0),
            np.log10(sigma_s / sigma_p), 0.0)
    vlim = max(abs(ratio.min()), abs(ratio.max()), 0.05)
    im2 = axes[2].imshow(ratio, vmin=-vlim, vmax=vlim,
                         cmap="coolwarm", origin="lower", interpolation="nearest")
    axes[2].set_title(r"$\log_{10}$(summary / particles)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    prefix = f"{title_prefix} " if title_prefix else ""
    fig.suptitle(f"{prefix}Summary mass conservation: ratio={mass_ratio:.4f}", fontsize=11)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out = os.path.join(out_dir, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Comparison image saved to {out}")


def _save_lod_sweep(sigma_ref, results, out_dir):
    """Save LOD sweep: reference + one panel per LOD level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    n_lods = len(results)
    fig, axes = plt.subplots(2, n_lods + 1, figsize=(3 * (n_lods + 1), 6))

    pos_ref = sigma_ref[sigma_ref > 0]
    if len(pos_ref) == 0:
        return
    vmin = pos_ref.min()
    vmax = pos_ref.max()

    # Top row: surface density maps
    axes[0, 0].imshow(sigma_ref, norm=LogNorm(vmin=vmin, vmax=vmax),
                      cmap="magma", origin="lower", interpolation="nearest")
    axes[0, 0].set_title("Reference\n(all particles)", fontsize=9)

    for j, (lod, n_emit, n_summ, ratio, corr, sigma_lod) in enumerate(results):
        axes[0, j + 1].imshow(sigma_lod, norm=LogNorm(vmin=vmin, vmax=vmax),
                              cmap="magma", origin="lower", interpolation="nearest")
        axes[0, j + 1].set_title(f"LOD={lod}\n{n_emit}p+{n_summ}s", fontsize=9)

    # Bottom row: log ratio vs reference
    axes[1, 0].set_visible(False)

    for j, (lod, n_emit, n_summ, ratio, corr, sigma_lod) in enumerate(results):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(
                (sigma_ref > 0) & (sigma_lod > 0),
                np.log10(sigma_lod / sigma_ref), 0.0)
        vlim = 1.0
        axes[1, j + 1].imshow(log_ratio, vmin=-vlim, vmax=vlim,
                              cmap="coolwarm", origin="lower", interpolation="nearest")
        axes[1, j + 1].set_title(f"mass={ratio:.3f} corr={corr:.3f}", fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("LOD sweep: 100 triaxial Gaussians in single tree", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "summary_lod_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"LOD sweep image saved to {out}")
