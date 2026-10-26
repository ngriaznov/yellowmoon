// The colour of the light of one pixel (docs/21 section 4, *As built (the light's colour)*), shared by the
// kernels that read the picture as light: lens_flare.wgsl (the restored light) and lens_light.wgsl (the input of
// L1 and L2). The CPU reference is `ym_lens::l3::Restore::light_colour`.
//
// The pixel with the clip pattern's cast taken off (every channel at its own clip level, which highlight
// recovery removes: a neutral of the same total), then each saturated channel (from LL_SAT_FROM to LL_SAT_TO of
// its own clip level, a smoothstep) raised to the mean of the channels that are not (weighted by how unsaturated
// they are), by the share it is saturated and the share of the others that are measuring, never lowered.
//
// Also the white light weight of the pixel (0 to 1: how much it is a clipped light of a white colour that holds
// more light than it records), `ym_lens::l3::Restore::light`: a saturated channel counts by how saturated it is,
// by the share of the others that measure (0.5 of their weight, at most 1) and by how far the others' mean lies
// above it, from 0 at its own value to 1 at LL_WHITE_RAMP above it; the largest over the channels.
//
// Generated before this text (kernels/lens_light.rs): LL_CLIP_PATTERN, LL_SAT_FROM, LL_SAT_TO, LL_WHITE_RAMP.

fn ll_smooth(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

// `clip` is g0 of each channel (0: no clip level), `ramp` the clip test's ramp (from, to). The colour of the
// light in xyz and the white light weight in w.
fn ll_light(v: vec3<f32>, clip: vec3<f32>, ramp: vec2<f32>) -> vec4<f32> {
    var m = 1.0e30;
    var hi = -1.0e30;
    var lo = 1.0e30;
    for (var c = 0u; c < 3u; c = c + 1u) {
        if (clip[c] > 0.0) {
            let n = v[c] / clip[c];
            m = min(m, n);
            hi = max(hi, n);
            lo = min(lo, n);
        }
    }
    var y = v;
    if (m < 1.0e29 && hi > 0.0) {
        let f = clamp((m - ramp.x) / (ramp.y - ramp.x), 0.0, 1.0);
        let t = clamp((hi - lo) / hi / LL_CLIP_PATTERN, 0.0, 1.0);
        let w = f * (1.0 - t * t * (3.0 - 2.0 * t));
        let mean = (v.x + v.y + v.z) / 3.0;
        y = v + w * (vec3<f32>(mean) - v);
    }
    var s = vec3<f32>(0.0);
    var any_sat = false;
    for (var c = 0u; c < 3u; c = c + 1u) {
        if (clip[c] > 0.0) {
            s[c] = ll_smooth((y[c] / clip[c] - LL_SAT_FROM) / (LL_SAT_TO - LL_SAT_FROM));
            any_sat = any_sat || s[c] > 0.0;
        }
    }
    if (!any_sat) {
        return vec4<f32>(y, 0.0);
    }
    var out = y;
    var white = 0.0;
    for (var c = 0u; c < 3u; c = c + 1u) {
        if (s[c] <= 0.0) {
            continue;
        }
        var weight = 0.0;
        var sum = 0.0;
        for (var o = 0u; o < 3u; o = o + 1u) {
            if (o != c) {
                let wo = 1.0 - s[o];
                weight = weight + wo;
                sum = sum + wo * y[o];
            }
        }
        if (weight <= 0.0) {
            continue;
        }
        let estimate = sum / weight;
        let lift = max(estimate - y[c], 0.0);
        out[c] = y[c] + s[c] * 0.5 * weight * lift;
        if (y[c] > 0.0) {
            let above = clamp(lift / (LL_WHITE_RAMP * y[c]), 0.0, 1.0);
            white = max(white, s[c] * min(0.5 * weight, 1.0) * above);
        }
    }
    return vec4<f32>(out, white);
}

// The colour of the light of one pixel (the first three lanes of ll_light).
fn ll_light_colour(v: vec3<f32>, clip: vec3<f32>, ramp: vec2<f32>) -> vec3<f32> {
    return ll_light(v, clip, ramp).xyz;
}
