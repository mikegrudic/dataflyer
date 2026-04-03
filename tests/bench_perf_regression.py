"""Performance regression benchmark.

Loads a snapshot and simulates realistic camera movement: look up, down,
left, right, roll left, roll right, then fly to the opposite side of the
box, turn around, and fly back. Measures cull + render times at each step.
"""

import time
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
    camera._dirty = True
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
    renderer.use_tree = True
    return renderer


def patch_render_headless(ctx, renderer):
    original_render = renderer.render.__func__
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


def rodrigues(v, axis, angle):
    """Rotate vector v around unit axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def build_waypoints(boxsize):
    """Build realistic camera waypoints: look around, fly across, turn, fly back."""
    center = boxsize / 2
    start_pos = np.array([center, center, boxsize], dtype=np.float32)
    fwd_z = np.array([0, 0, -1], dtype=np.float32)
    up_y = np.array([0, 1, 0], dtype=np.float32)
    right_x = np.array([1, 0, 0], dtype=np.float32)

    waypoints = []

    def add(pos, fwd, up, label):
        waypoints.append((pos.astype(np.float32), fwd.astype(np.float32),
                          up.astype(np.float32), label))

    # Start: looking down -z
    add(start_pos, fwd_z, up_y, "start (look -z)")

    # Look up (pitch up 30 deg)
    fwd_up = rodrigues(fwd_z, right_x, np.radians(30))
    add(start_pos, fwd_up, up_y, "look up 30deg")

    # Look down (pitch down 30 deg from original)
    fwd_down = rodrigues(fwd_z, right_x, np.radians(-30))
    add(start_pos, fwd_down, up_y, "look down 30deg")

    # Look left (yaw left 45 deg)
    fwd_left = rodrigues(fwd_z, up_y, np.radians(45))
    add(start_pos, fwd_left, up_y, "look left 45deg")

    # Look right (yaw right 45 deg)
    fwd_right = rodrigues(fwd_z, up_y, np.radians(-45))
    add(start_pos, fwd_right, up_y, "look right 45deg")

    # Roll left 30 deg (forward unchanged, up rotates)
    up_roll_l = rodrigues(up_y, fwd_z, np.radians(30))
    add(start_pos, fwd_z, up_roll_l, "roll left 30deg")

    # Roll right 30 deg
    up_roll_r = rodrigues(up_y, fwd_z, np.radians(-30))
    add(start_pos, fwd_z, up_roll_r, "roll right 30deg")

    # Back to center, looking -z
    add(start_pos, fwd_z, up_y, "reset orientation")

    # Fly forward through center to opposite side
    mid_pos = np.array([center, center, center], dtype=np.float32)
    add(mid_pos, fwd_z, up_y, "fly to center")

    far_pos = np.array([center, center, 0], dtype=np.float32)
    add(far_pos, fwd_z, up_y, "fly to far side")

    # Turn around 180 deg (yaw 180)
    fwd_back = np.array([0, 0, 1], dtype=np.float32)
    add(far_pos, fwd_back, up_y, "turn around")

    # Fly back through center
    add(mid_pos, fwd_back, up_y, "fly back to center")

    # Fly back to start
    add(start_pos, fwd_back, up_y, "fly back to start")

    return waypoints


def run_benchmark():
    import moderngl

    print(f"Loading {SNAPSHOT}...")
    pos, masses, hsml, boxsize = load_snapshot(SNAPSHOT)
    print(f"  {len(pos):,} particles, boxsize={boxsize}")

    ctx = moderngl.create_standalone_context()
    renderer = setup_renderer(ctx)
    camera = make_camera(boxsize, FOV)
    renderer._viewport_width = RES

    t0 = time.perf_counter()
    renderer.set_particles(pos, hsml, masses)
    t_build = time.perf_counter() - t0
    print(f"  set_particles: {t_build:.2f}s")

    patch_render_headless(ctx, renderer)

    waypoints = build_waypoints(boxsize)

    # Warmup
    for wp_pos, wp_fwd, wp_up, _ in waypoints[:3]:
        camera.position = wp_pos.copy()
        camera._forward = wp_fwd.copy()
        camera._up = wp_up.copy()
        camera._dirty = True
        renderer.update_visible(camera)
        renderer.render(camera, RES, RES)
        ctx.finish()

    print(f"\n  {'Step':<25s} {'cull_ms':>8s} {'render_ms':>10s} {'n_vis':>10s}")
    print("  " + "-" * 57)

    cull_times = []
    render_times = []

    for wp_pos, wp_fwd, wp_up, label in waypoints:
        camera.position = wp_pos.copy()
        camera._forward = wp_fwd.copy()
        camera._up = wp_up.copy()
        camera._dirty = True

        t0 = time.perf_counter()
        renderer.update_visible(camera)
        t_cull = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        renderer.render(camera, RES, RES)
        ctx.finish()
        t_render = (time.perf_counter() - t0) * 1000

        cull_times.append(t_cull)
        render_times.append(t_render)

        n_vis = renderer.n_particles + getattr(renderer, 'n_big', 0)
        print(f"  {label:<25s} {t_cull:>7.1f}  {t_render:>9.1f}  {n_vis:>10,}")

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
