// Brute-force subsample + frustum cull on all particles.
// Each thread tests one source particle. Particles whose hashed index
// modulo `stride` is zero AND that pass a frustum test are written
// to the output buffers via an atomic counter.
//
// The hash decorrelates spatial locality so striding gives a uniform
// random sample even though we step through Morton-sorted positions.

struct Params {
    cam_pos: vec3<f32>,
    _p0: f32,
    cam_fwd: vec3<f32>,
    _p1: f32,
    cam_right: vec3<f32>,
    _p2: f32,
    cam_up: vec3<f32>,
    _p3: f32,
    fov_rad: f32,
    aspect: f32,
    stride: u32,
    n_particles: u32,
    h_scale: f32,         // hsml scale = stride^(1/3)
    mass_scale: f32,      // mass scale = stride
    max_output: u32,
    chunk_offset: u32,    // global index of first particle in this chunk
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> src_hsml: array<f32>;
@group(0) @binding(3) var<storage, read> src_mass: array<f32>;
@group(0) @binding(4) var<storage, read> src_qty: array<f32>;
@group(0) @binding(5) var<storage, read_write> counter: array<atomic<u32>>;
// counter[0] = output write index

@group(1) @binding(0) var<storage, read_write> out_pos: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out_hsml: array<f32>;
@group(1) @binding(2) var<storage, read_write> out_mass: array<f32>;
@group(1) @binding(3) var<storage, read_write> out_qty: array<f32>;

// 64-bit avalanche hash (uint64 splitmix). WGSL has u32 only, so we
// fold the 64-bit version down to 32 bits.
fn hash_idx(i: u32) -> u32 {
    var x: u32 = i;
    x = x ^ (x >> 17u);
    x = x * 0xed5ad4bbu;
    x = x ^ (x >> 11u);
    x = x * 0xac4c1b51u;
    x = x ^ (x >> 15u);
    x = x * 0x31848babu;
    x = x ^ (x >> 14u);
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    // 2D workgroup grid wraps a 1D thread index (workgroup limit is
    // 65535 per dim, so large particle counts use wgy > 1).
    let wgx_count = num_wg.x;
    let idx = gid.y * wgx_count * 256u + gid.x;
    if (idx >= params.n_particles) { return; }

    // Stride sample using the *global* index so hashing is independent
    // of chunk boundaries.
    let global_idx = params.chunk_offset + idx;
    let stride = max(params.stride, 1u);
    let h = hash_idx(global_idx);
    if ((h % stride) != 0u) { return; }

    let pos = src_pos[idx].xyz;
    let hsml = src_hsml[idx] * params.h_scale;

    // Frustum test (use scaled hsml so enlarged kernels at boundary aren't dropped)
    let depth = dot(pos - params.cam_pos, params.cam_fwd);
    let right_d = dot(pos - params.cam_pos, params.cam_right);
    let up_d = dot(pos - params.cam_pos, params.cam_up);

    let cell_extent = hsml;
    let half_tan = tan(params.fov_rad * 0.5);
    let front_depth = max(depth + cell_extent, 0.0);
    let lim_h = front_depth * half_tan * params.aspect + cell_extent;
    let lim_v = front_depth * half_tan + cell_extent;
    let in_front = depth > -cell_extent;
    if (!in_front || abs(right_d) >= lim_h || abs(up_d) >= lim_v) {
        return;
    }

    // Atomically claim an output slot
    let out_idx = atomicAdd(&counter[0], 1u);
    if (out_idx >= params.max_output) { return; }

    out_pos[out_idx] = vec4<f32>(pos, 0.0);
    out_hsml[out_idx] = hsml;
    out_mass[out_idx] = src_mass[idx] * params.mass_scale;
    out_qty[out_idx] = src_qty[idx];
}
