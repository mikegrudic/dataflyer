"""Performance regression benchmark across git commits.

Loads SN_512 (134M particles), flies camera to opposite side of box and
turns around, measuring cull + render times at each step.
"""

import time
import sys
import types
import numpy as np


SNAPSHOT = "/Users/mgrudic/code/bubblebuddies_gizmo/popeye/bubblebuddies_gizmo/SN_128/snapshot_065.hdf5"
RES = 512
FOV = 90


def load_snapshot(path):
    import h5py
    with h5py.File(path, "r") as f:
        pos = f["PartType0/Coordinates"][:].astype(np.float32)
        masses = f["PartType0/Masses"][:].astype(np.float32)
        boxsize = float(f["Header"].attrs["BoxSize"])
        # Try KernelMaxRadius, fall back to SmoothingLength
        for field in ("KernelMaxRadius", "SmoothingLength"):
            if field in f["PartType0"]:
                hsml = f["PartType0"][field][:].astype(np.float32)
                break
        else:
            raise KeyError("No smoothing length field found")
    return pos, masses, hsml, boxsize


def make_camera(boxsize, fov):
    from dataflyer.camera import Camera
    camera = Camera(fov=fov, aspect=1.0)
    center = boxsize / 2
    camera.position = np.array([center, center, boxsize], dtype=np.float32)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)
    camera.near = boxsize * 1e-6
    camera.far = boxsize * 10
    camera.speed = boxsize / 10
    return camera


def setup_renderer(ctx):
    from dataflyer.renderer import SplatRenderer
    from dataflyer.colormaps import create_colormap_texture_safe
    renderer = SplatRenderer(ctx)
    renderer.colormap_tex = create_colormap_texture_safe(ctx, "magma")
    renderer.resolve_mode = 0
    renderer.log_scale = 1
    if hasattr(renderer, 'use_tree'):
        renderer.use_tree = True
    return renderer


def patch_render_headless(ctx, renderer):
    """Patch render() to work without a real screen framebuffer."""
    original_render = renderer.render.__func__ if hasattr(renderer.render, '__func__') else renderer.render
    original_screen = type(ctx).screen

    def _headless_render(self, camera, width, height):
        try:
            type(ctx).screen = property(lambda s: (_ for _ in ()).throw(StopIteration))
            original_render(self, camera, width, height)
        except (StopIteration, AttributeError):
            pass
        finally:
            type(ctx).screen = original_screen

    renderer.render = types.MethodType(_headless_render, renderer)


def run_benchmark():
    import moderngl

    print(f"Loading {SNAPSHOT}...")
    pos, masses, hsml, boxsize = load_snapshot(SNAPSHOT)
    print(f"  {len(pos):,} particles, boxsize={boxsize}")

    ctx = moderngl.create_standalone_context()
    renderer = setup_renderer(ctx)
    camera = make_camera(boxsize, FOV)
    renderer._viewport_width = RES

    # set_particles (includes grid build)
    t0 = time.perf_counter()
    renderer.set_particles(pos, hsml, masses)
    t_build = time.perf_counter() - t0
    print(f"  set_particles: {t_build:.2f}s")

    patch_render_headless(ctx, renderer)

    # Define camera waypoints: start at top looking -z, fly to bottom, turn around
    center = boxsize / 2
    waypoints = [
        # (position, forward) — fly through box then look back
        (np.array([center, center, boxsize], dtype=np.float32),
         np.array([0, 0, -1], dtype=np.float32)),
        (np.array([center, center, center], dtype=np.float32),
         np.array([0, 0, -1], dtype=np.float32)),
        (np.array([center, center, 0], dtype=np.float32),
         np.array([0, 0, -1], dtype=np.float32)),
        # Turn around — now at bottom looking +z back through the box
        (np.array([center, center, 0], dtype=np.float32),
         np.array([0, 0, 1], dtype=np.float32)),
        (np.array([center, center, center], dtype=np.float32),
         np.array([0, 0, 1], dtype=np.float32)),
    ]

    print(f"\n  {'Step':<35s} {'cull_ms':>8s} {'render_ms':>10s} {'n_vis':>10s}")
    print("  " + "-" * 67)

    cull_times = []
    render_times = []

    # Warmup: 3 cull+render cycles to prime CPU caches and JIT
    for wp in waypoints[:3]:
        camera.position = wp[0].copy()
        camera._forward = wp[1].copy()
        camera._up = np.array([0, 1, 0], dtype=np.float32)
        if hasattr(renderer, 'update_visible'):
            renderer.update_visible(camera)
        renderer.render(camera, RES, RES)
        ctx.finish()

    for _, (position, forward) in enumerate(waypoints):
        camera.position = position.copy()
        camera._forward = forward.copy()
        camera._up = np.array([0, 1, 0], dtype=np.float32)

        t0 = time.perf_counter()
        if hasattr(renderer, 'update_visible'):
            renderer.update_visible(camera)
        t_cull = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        renderer.render(camera, RES, RES)
        ctx.finish()
        t_render = (time.perf_counter() - t0) * 1000

        cull_times.append(t_cull)
        render_times.append(t_render)

        n_vis = renderer.n_particles + getattr(renderer, 'n_big', 0)
        label = f"pos=({position[0]:.0f},{position[1]:.0f},{position[2]:.0f}) fwd=({forward[2]:+.0f}z)"
        print(f"  {label:<35s} {t_cull:>7.1f}  {t_render:>9.1f}  {n_vis:>10,}")

    med_cull = float(np.median(cull_times))
    med_render = float(np.median(render_times))
    print(f"\n  Median: cull={med_cull:.0f}ms  render={med_render:.0f}ms  build={t_build:.2f}s")

    renderer.release()
    ctx.release()

    return {
        "build_s": t_build,
        "median_cull_ms": med_cull,
        "median_render_ms": med_render,
    }


if __name__ == "__main__":
    run_benchmark()
