#version 330

// Instanced quad for anisotropic Gaussian summary splats.
// Each instance has a 3D covariance (upper triangle, 6 floats).
// We project to 2D screen-space, eigendecompose, and orient the quad.

in vec2 in_corner;  // (-1,-1), (1,-1), (-1,1), (1,1)

in vec3 in_position;
in float in_mass;
in float in_quantity;
in vec3 in_cov_a;   // (cov_xx, cov_xy, cov_xz)
in vec3 in_cov_b;   // (cov_yy, cov_yz, cov_zz)

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_cov_scale;  // scaling factor for covariance (preserves mass)

out vec2 v_offset;    // position in sigma units for Gaussian evaluation
out float v_mass;
out float v_quantity;
out float v_gauss_norm;  // 1/(2π √det(Σ_2D))

void main() {
    // Reconstruct 3D covariance
    float c_xx = in_cov_a.x, c_xy = in_cov_a.y, c_xz = in_cov_a.z;
    float c_yy = in_cov_b.x, c_yz = in_cov_b.y, c_zz = in_cov_b.z;

    vec4 view_center = u_view * vec4(in_position, 1.0);

    // View rotation (upper-left 3x3)
    mat3 R = mat3(u_view);

    // Rotate covariance to view space: Σ_view = R Σ_3D R^T
    mat3 S = mat3(
        c_xx, c_xy, c_xz,
        c_xy, c_yy, c_yz,
        c_xz, c_yz, c_zz
    );
    mat3 cov_view = R * S * transpose(R);

    // Take upper-left 2x2 (marginalize over z) for 2D projected covariance
    float s_xx = cov_view[0][0];
    float s_xy = cov_view[0][1];
    float s_yy = cov_view[1][1];

    // Regularize
    s_xx = max(s_xx, 1e-8);
    s_yy = max(s_yy, 1e-8);

    // Eigendecomposition of 2x2 symmetric [[s_xx, s_xy], [s_xy, s_yy]]
    float trace = s_xx + s_yy;
    float diff = s_xx - s_yy;
    float disc = sqrt(max(diff * diff + 4.0 * s_xy * s_xy, 0.0));
    float lambda1 = max(0.5 * (trace + disc), 1e-8) * u_cov_scale;
    float lambda2 = max(0.5 * (trace - disc), 1e-8) * u_cov_scale;

    // Eigenvector for lambda1
    vec2 ev1;
    if (abs(s_xy) > 1e-10) {
        ev1 = normalize(vec2(lambda1 - s_yy, s_xy));
    } else {
        ev1 = (s_xx >= s_yy) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }
    vec2 ev2 = vec2(-ev1.y, ev1.x);

    // 3-sigma ellipse radii in view space
    float r1 = 3.0 * sqrt(lambda1);
    float r2 = 3.0 * sqrt(lambda2);

    // Offset in clip space to avoid perspective distortion of off-axis splats
    vec4 clip_center = u_proj * view_center;
    vec2 view_offset = in_corner.x * ev1 * r1 + in_corner.y * ev2 * r2;
    gl_Position = clip_center;
    gl_Position.xy += view_offset * vec2(u_proj[0][0], u_proj[1][1]);

    // Fragment gets position in sigma units ([-3, 3] at quad edges)
    v_offset = in_corner * 3.0;

    v_mass = in_mass;
    v_quantity = in_quantity;
    v_gauss_norm = 1.0 / (6.2831853 * sqrt(lambda1 * lambda2));
}
