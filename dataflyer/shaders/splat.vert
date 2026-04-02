#version 330

// Per-vertex: one vertex per particle (point sprite)
in vec3 in_position;
in float in_hsml;
in float in_mass;
in float in_quantity;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport_size;  // (width, height) in pixels

out float v_mass;
out float v_hsml;
out float v_quantity;
out float v_coord_scale;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    vec4 clip_pos = u_proj * view_pos;

    // Compute pixel radius by projecting center and center+h through the
    // same projection matrix the quad path uses (no manual NDC math)
    vec4 edge_clip = u_proj * (view_pos + vec4(in_hsml, 0.0, 0.0, 0.0));
    vec2 center_ndc = clip_pos.xy / clip_pos.w;
    vec2 edge_ndc = edge_clip.xy / edge_clip.w;
    float radius_px = length((edge_ndc - center_ndc) * 0.5 * u_viewport_size);
    float desired_pixels = radius_px * 2.0;  // full diameter

    // Hardware clamps to [1, 64] on macOS Metal
    float actual_pixels = clamp(desired_pixels, 2.0, 64.0);
    gl_PointSize = actual_pixels;
    gl_Position = clip_pos;

    // If clamped, rescale gl_PointCoord so the kernel maps over the full desired size
    v_coord_scale = desired_pixels / actual_pixels;

    v_mass = in_mass;
    v_hsml = in_hsml;
    v_quantity = in_quantity;
}
