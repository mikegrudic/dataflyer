#version 330

// Fragment shader for anisotropic Gaussian summary splats.
// v_offset is in sigma units along principal axes of the projected covariance.

in vec2 v_offset;
in float v_mass;
in float v_quantity;
in float v_gauss_norm;  // 1/(2π √det(Σ_2D)), already includes cov_scale

layout(location = 0) out float out_numerator;
layout(location = 1) out float out_denominator;

void main() {
    float r2 = dot(v_offset, v_offset);
    if (r2 > 9.0) discard;

    // σ = m / (2π √det) * exp(-½ xᵀΣ⁻¹x)
    // v_gauss_norm includes the scaled det, so mass is conserved
    float sigma = v_mass * v_gauss_norm * exp(-0.5 * r2);

    out_numerator = sigma * v_quantity;
    out_denominator = sigma;
}
