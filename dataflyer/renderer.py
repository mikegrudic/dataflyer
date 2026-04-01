"""Core splat renderer: additive accumulation + resolve pass."""

import numpy as np
import moderngl
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"

# Maximum particles to render per frame for interactive performance
MAX_RENDER_PARTICLES = 4_000_000


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


class SpatialGrid:
    """Uniform grid spatial index for fast frustum queries."""

    def __init__(self, positions, n_cells=64):
        self.n_cells = n_cells
        self.pmin = positions.min(axis=0)
        self.pmax = positions.max(axis=0)
        self.cell_size = (self.pmax - self.pmin) / n_cells
        self.cell_size[self.cell_size == 0] = 1.0  # avoid division by zero

        # Assign each particle to a cell
        cell_idx = np.clip(
            ((positions - self.pmin) / self.cell_size).astype(np.int32),
            0, n_cells - 1
        )
        # Flatten 3D cell index to 1D
        self.cell_id = cell_idx[:, 0] * n_cells * n_cells + cell_idx[:, 1] * n_cells + cell_idx[:, 2]

        # Sort particles by cell for cache-friendly access
        self.sort_order = np.argsort(self.cell_id)
        self.sorted_cell_id = self.cell_id[self.sort_order]

        # Build cell start/end lookup
        n_total_cells = n_cells ** 3
        self.cell_start = np.zeros(n_total_cells + 1, dtype=np.int64)
        # Count particles per cell
        unique_cells, counts = np.unique(self.sorted_cell_id, return_counts=True)
        self.cell_start[unique_cells + 1] = counts
        np.cumsum(self.cell_start, out=self.cell_start)

    def query_frustum(self, camera, max_particles):
        """Return indices of particles visible in the camera frustum."""
        nc = self.n_cells
        cs = self.cell_size

        # Cell center positions
        cx = np.arange(nc) * cs[0] + self.pmin[0] + cs[0] * 0.5
        cy = np.arange(nc) * cs[1] + self.pmin[1] + cs[1] * 0.5
        cz = np.arange(nc) * cs[2] + self.pmin[2] + cs[2] * 0.5
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing='ij')
        cell_centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

        # Camera-space dot products on cell centers (n_cells^3 is small, e.g. 262K)
        cam_pos = camera.position
        cam_fwd = camera.forward
        cam_right = camera.right
        cam_up = camera.up

        depths = cell_centers @ cam_fwd - np.dot(cam_pos, cam_fwd)
        rights = cell_centers @ cam_right - np.dot(cam_pos, cam_right)
        ups = cell_centers @ cam_up - np.dot(cam_pos, cam_up)

        # Cell half-diagonal for conservative test
        half_diag = np.linalg.norm(cs) * 0.5

        # Frustum test on cells (wide: 2x FOV)
        half_tan = np.tan(np.radians(min(camera.fov, 120)) / 2) * 2.0
        in_front = depths > -half_diag
        limit_h = (depths + half_diag) * half_tan * camera.aspect + half_diag
        limit_v = (depths + half_diag) * half_tan + half_diag

        visible_cells = np.where(
            in_front & (np.abs(rights) < limit_h) & (np.abs(ups) < limit_v)
        )[0]

        # Gather particle indices from visible cells
        idx_parts = []
        total = 0
        for c in visible_cells:
            start = self.cell_start[c]
            end = self.cell_start[c + 1]
            n = end - start
            if n == 0:
                continue
            idx_parts.append(self.sort_order[start:end])
            total += n

        if len(idx_parts) == 0:
            return np.array([], dtype=np.int64)

        visible_idx = np.concatenate(idx_parts)

        # LOD subsample if over budget
        if len(visible_idx) > max_particles:
            step = max(1, len(visible_idx) // max_particles)
            visible_idx = visible_idx[::step]

        return visible_idx


class SplatRenderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.n_particles = 0
        self.n_total = 0  # total particles in dataset

        # Compile shader programs
        self.prog_additive = ctx.program(
            vertex_shader=_load_shader("splat.vert"),
            fragment_shader=_load_shader("splat_additive.frag"),
        )
        self.prog_resolve = ctx.program(
            vertex_shader=_load_shader("resolve.vert"),
            fragment_shader=_load_shader("resolve.frag"),
        )
        self.prog_star = ctx.program(
            vertex_shader=_load_shader("star.vert"),
            fragment_shader=_load_shader("star.frag"),
        )

        # Quad vertices for billboard (two triangles)
        quad = np.array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())

        quad_indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.int32)
        self.quad_ibo = ctx.buffer(quad_indices.tobytes())

        # Fullscreen quad for resolve pass
        fs_quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.fs_quad_vbo = ctx.buffer(fs_quad.tobytes())

        # Particle buffers (set by upload_visible)
        self.pos_vbo = None
        self.hsml_vbo = None
        self.mass_vbo = None
        self.qty_vbo = None
        self.vao_additive = None

        # Full dataset stored on CPU for culling
        self._all_pos = None
        self._all_hsml = None
        self._all_mass = None
        self._all_qty = None

        # FBO for accumulation (2 float textures: numerator + denominator)
        self._accum_fbo = None
        self._accum_tex_num = None
        self._accum_tex_den = None
        self._fbo_size = (0, 0)

        # Resolve VAO
        self.vao_resolve = ctx.vertex_array(
            self.prog_resolve,
            [(self.fs_quad_vbo, "2f", "in_position")],
        )

        # Star particle buffers
        self.star_pos_vbo = None
        self.star_mass_vbo = None
        self.vao_star = None
        self.n_stars = 0
        self.star_point_size = 50.0

        # Colormap texture (set externally)
        self.colormap_tex = None

        # Render state
        self.alpha_scale = 1.0
        self.qty_min = -1.0
        self.qty_max = 3.0
        self.mode = 0       # 0: surface density, 1: weighted quantity
        self.log_scale = 1  # 1: log10, 0: linear
        self.max_render_particles = MAX_RENDER_PARTICLES

    def set_particles(self, positions, hsml, masses, quantity=None):
        """Store particle data on CPU. Call update_visible() to upload a subset."""
        self._all_pos = positions.astype(np.float32)
        self._all_hsml = hsml.astype(np.float32)
        self._all_mass = masses.astype(np.float32)
        if quantity is None:
            quantity = masses
        self._all_qty = quantity.astype(np.float32)
        self.n_total = len(masses)

        # Build spatial grid for fast frustum queries on large datasets
        if self.n_total > self.max_render_particles:
            import time
            t0 = time.perf_counter()
            self._grid = SpatialGrid(self._all_pos)
            print(f"  Spatial grid built in {time.perf_counter()-t0:.1f}s")
        else:
            self._grid = None

    def update_visible(self, camera):
        """Cull and upload only visible particles for this frame."""
        if self._all_pos is None:
            return

        if self._grid is not None:
            idx = self._grid.query_frustum(camera, self.max_render_particles)
        else:
            idx = np.arange(self.n_total)

        self._upload_subset(idx)

    def _upload_subset(self, idx):
        """Upload a subset of particles to GPU."""
        self.n_particles = len(idx)

        if self.n_particles == 0:
            return

        # Release old buffers
        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.pos_vbo = self.ctx.buffer(self._all_pos[idx].tobytes())
        self.hsml_vbo = self.ctx.buffer(self._all_hsml[idx].tobytes())
        self.mass_vbo = self.ctx.buffer(self._all_mass[idx].tobytes())
        self.qty_vbo = self.ctx.buffer(self._all_qty[idx].tobytes())

        self._build_vao()

    def upload_particles(self, positions, hsml, masses, quantity=None):
        """Upload all particles directly (for small datasets)."""
        self.set_particles(positions, hsml, masses, quantity)
        self.n_particles = self.n_total

        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.pos_vbo = self.ctx.buffer(self._all_pos.tobytes())
        self.hsml_vbo = self.ctx.buffer(self._all_hsml.tobytes())
        self.mass_vbo = self.ctx.buffer(self._all_mass.tobytes())
        self.qty_vbo = self.ctx.buffer(self._all_qty.tobytes())

        self._build_vao()

    def update_quantity(self, quantity):
        """Update just the quantity data."""
        self._all_qty = quantity.astype(np.float32)
        # If we have a subset uploaded, re-upload is deferred to next update_visible()
        # For now, re-upload all
        if self.qty_vbo is not None:
            self.qty_vbo.release()
        self.qty_vbo = self.ctx.buffer(self._all_qty.tobytes())
        self._build_vao()

    def upload_stars(self, positions, masses):
        """Upload star particle data for point sprite rendering."""
        self.n_stars = len(masses)
        if self.n_stars == 0:
            return

        for attr in ("star_pos_vbo", "star_mass_vbo"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self.star_pos_vbo = self.ctx.buffer(positions.astype(np.float32).tobytes())
        self.star_mass_vbo = self.ctx.buffer(masses.astype(np.float32).tobytes())

        if self.vao_star is not None:
            self.vao_star.release()
        self.vao_star = self.ctx.vertex_array(
            self.prog_star,
            [
                (self.star_pos_vbo, "3f", "in_position"),
                (self.star_mass_vbo, "f", "in_mass"),
            ],
        )

    def _build_vao(self):
        """(Re)build vertex array object after buffer changes."""
        if self.vao_additive is not None:
            self.vao_additive.release()

        if self.pos_vbo is None:
            return

        content = [
            (self.quad_vbo, "2f", "in_corner"),
            (self.pos_vbo, "3f/i", "in_position"),
            (self.hsml_vbo, "f/i", "in_hsml"),
            (self.mass_vbo, "f/i", "in_mass"),
            (self.qty_vbo, "f/i", "in_quantity"),
        ]

        self.vao_additive = self.ctx.vertex_array(
            self.prog_additive, content, index_buffer=self.quad_ibo,
        )

    def _ensure_accum_fbo(self, width, height):
        """Create or resize the accumulation FBO."""
        if self._fbo_size == (width, height) and self._accum_fbo is not None:
            return

        for attr in ("_accum_fbo", "_accum_tex_num", "_accum_tex_den"):
            old = getattr(self, attr, None)
            if old is not None:
                old.release()

        self._accum_tex_num = self.ctx.texture((width, height), 1, dtype="f4")
        self._accum_tex_den = self.ctx.texture((width, height), 1, dtype="f4")
        self._accum_fbo = self.ctx.framebuffer(
            color_attachments=[self._accum_tex_num, self._accum_tex_den],
        )
        self._fbo_size = (width, height)

    def render(self, camera, width, height):
        """Render particle splats via additive accumulation + resolve."""
        if self.n_particles == 0 or self.colormap_tex is None:
            return

        # GLSL mat4 is column-major; numpy is row-major
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(camera.projection_matrix().T)

        self._ensure_accum_fbo(width, height)

        # Pass 1: additive accumulation into float FBO
        self._accum_fbo.use()
        self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.prog_additive["u_view"].write(view.tobytes())
        self.prog_additive["u_proj"].write(proj.tobytes())
        self.prog_additive["u_viewport_size"].value = (float(width), float(height))

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # pure additive
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.vao_additive.render(
            moderngl.TRIANGLES,
            instances=self.n_particles,
        )

        # Pass 2: resolve to screen
        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)

        self._accum_tex_num.use(location=0)
        self._accum_tex_den.use(location=1)
        self.colormap_tex.use(location=2)

        self.prog_resolve["u_numerator"].value = 0
        self.prog_resolve["u_denominator"].value = 1
        self.prog_resolve["u_colormap"].value = 2
        self.prog_resolve["u_qty_min"].value = self.qty_min
        self.prog_resolve["u_qty_max"].value = self.qty_max
        self.prog_resolve["u_mode"].value = self.mode
        self.prog_resolve["u_log_scale"].value = self.log_scale

        self.vao_resolve.render(moderngl.TRIANGLE_STRIP, vertices=4)

        # Pass 3: render star particles on top
        if self.n_stars > 0 and self.vao_star is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

            self.prog_star["u_view"].write(view.tobytes())
            self.prog_star["u_proj"].write(proj.tobytes())
            self.prog_star["u_point_size"].value = self.star_point_size

            self.vao_star.render(moderngl.POINTS)

            self.ctx.disable(moderngl.BLEND)

    def read_accum_range(self):
        """Read back the accumulation textures and compute percentile range.
        Returns (lo, hi) in log10 or linear depending on self.log_scale.
        Must be called after render()."""
        if self._accum_tex_den is None:
            return self.qty_min, self.qty_max

        w, h = self._fbo_size
        # Read denominator (surface density) for mode 0, or compute ratio for mode 1
        den_data = np.frombuffer(self._accum_tex_den.read(), dtype=np.float32)
        if self.mode == 1:
            num_data = np.frombuffer(self._accum_tex_num.read(), dtype=np.float32)
            mask = den_data > 1e-30
            vals = np.where(mask, num_data / den_data, 0)
            vals = vals[mask]
        else:
            vals = den_data[den_data > 1e-30]

        if len(vals) == 0:
            return self.qty_min, self.qty_max

        # Mass-weighted CDF limits (matches SinkVis):
        # find values that enclose 1% and 99% of total integrated signal
        sorted_vals = np.sort(vals)
        cdf = sorted_vals.cumsum() / sorted_vals.sum()
        lim_lo = float(np.interp(0.01, cdf, sorted_vals))
        lim_hi = float(np.interp(0.99, cdf, sorted_vals))

        if self.log_scale:
            if lim_lo <= 0:
                lim_lo = float(sorted_vals[sorted_vals > 0].min()) if (sorted_vals > 0).any() else 1e-10
            lo = float(np.log10(max(lim_lo, 1e-30)))
            hi = float(np.log10(max(lim_hi, 1e-30)))
            if hi - lo < 0.1:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1
        else:
            lo, hi = lim_lo, lim_hi
            if hi - lo < 1e-30:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1

        return lo, hi

    def release(self):
        """Clean up GPU resources."""
        for attr in ("pos_vbo", "hsml_vbo", "mass_vbo", "qty_vbo",
                     "quad_vbo", "quad_ibo", "fs_quad_vbo",
                     "star_pos_vbo", "star_mass_vbo",
                     "vao_additive", "vao_resolve", "vao_star",
                     "prog_additive", "prog_resolve", "prog_star",
                     "_accum_fbo", "_accum_tex_num", "_accum_tex_den"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
