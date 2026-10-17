// The k x k binning of the fit view and the sigma that binning leaves: the GPU port of
// `ym-core`'s `bin_mosaic` (with the vignetting gain of `gain_plane` multiplied in) and
// `binned_sigma` (`docs/01`, the reduced path).
//
// bin_main: one thread per output pixel (ox, oy). Green is the mean of the block's k^2 / 2 green
// photosites (each row's greens added left to right, the rows added top to bottom, divided by the
// count). Red and blue are the block's mean of the colour's lattice interpolated linearly: the
// weights `lattice_weights(k, q)` come from the host (f64 integrals rounded to f32), the lattice rows
// are visited in order, each row's taps are weighted and added left to right (`line`), then
// `plane += wy line`. Indices beyond the frame mirror with the parity kept (`same_parity`). The
// mosaic is a slab of the frame (full width, some rows); the rows a block needs are inside it
// (two rows of context on each side, clamped at the frame's edges). Each product and sum is hidden
// from fast-math (`opaque`), so the values have the CPU's rounding. With a vignetting profile the
// binned value is multiplied by the gain at the binned pixel's centre.
//
// bin_sigma_main: one thread per output pixel: `sqrt(mean of s^2 over the step x step cells) * f * g`
// with the cell index clamped to the plane, and `g` the vignetting gain at the binned pixel's centre.
//
// Bindings: 0 uniform, 1 the slab mosaic (bin) or the half-resolution sigma plane (bin_sigma),
// 2 the output planes (bin) or the sigma plane (bin_sigma).

struct BinU {
    zero: vec4<u32>,   // always +0.0 bits
    dims: vec4<u32>,   // slab width (the frame's), slab height, slab row stride, k
    frame: vec4<u32>,  // frame width, frame height, the slab's first frame row, output width
    out: vec4<u32>,    // first output row, output rows, output row stride, output plane length
    cfa: vec4<u32>,    // plane index at (0,0), (1,0), (0,1), (1,1) of the frame
    lat: vec4<u32>,    // red lattice phase (x, y), blue lattice phase (x, y)
    nw: vec4<u32>,     // weights: red x, red y, blue x, blue y
    flags: vec4<u32>,  // vignetting on, sigma plane width, sigma plane height, step = max(k / 2, 1)
    f: vec4<f32>,      // green count k^2 / 2, the noise factor of the binning, 0, 0
    band: vec4<u32>,   // the destination image's first output row (0 when it holds the whole output), 0, 0, 0
    wts: array<vec4<f32>, 12>,   // red x, red y, blue x, blue y: up to 12 weights each, 3 vec4s apiece
    vig: VigU,
};

@group(0) @binding(0) var<uniform> pu: BinU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn weight(which: u32, i: u32) -> f32 {
    return pu.wts[which * 3u + (i >> 2u)][i & 3u];
}

// The index moved by whole periods of two into 0..n: the mirror that keeps the colour of a photosite.
fn same_parity(i: i32, n: i32) -> i32 {
    if (i < 0) {
        return min(i + 2, n - 1);
    }
    if (i >= n) {
        return max(i - 2, 0);
    }
    return i;
}

fn mosaic_at(x: i32, y: i32) -> f32 {
    let fy = u32(same_parity(y, i32(pu.frame.y)));
    let fx = u32(same_parity(x, i32(pu.frame.x)));
    return src[(fy - pu.frame.z) * pu.dims.z + fx];
}

fn bin_main(gid: vec3<u32>) {
    if (gid.x >= pu.frame.w || gid.y >= pu.out.y) {
        return;
    }
    let k = i32(pu.dims.w);
    let ox = i32(gid.x);
    let oy = i32(pu.out.x + gid.y);
    let w = i32(pu.frame.x);
    let h = i32(pu.frame.y);
    // Green.
    var gsum = 0.0;
    for (var y = oy * k; y < (oy + 1) * k; y = y + 1) {
        let first = select(0, 1, pu.cfa[u32(y & 1) * 2u] != 1u);
        var rs = 0.0;
        for (var x = ox * k + first; x < (ox + 1) * k; x = x + 2) {
            rs = opaque(rs + mosaic_at(x, y));
        }
        gsum = opaque(gsum + rs);
    }
    var g = 1.0;
    if (pu.flags.x != 0u) {
        g = vig_gain((2 * ox + 1) * k - w, (2 * oy + 1) * k - h);
    }
    let o = (u32(oy) - pu.band.x) * pu.out.z + gid.x;
    var gv = div_exact(gsum, pu.f.x);
    if (pu.flags.x != 0u) {
        gv = opaque(gv * g);
    }
    dst[pu.out.w + o] = gv;
    // Red (plane 0) and blue (plane 2).
    for (var c = 0u; c < 2u; c = c + 1u) {
        let qx = i32(pu.lat[2u * c]);
        let qy = i32(pu.lat[2u * c + 1u]);
        let nx = pu.nw[2u * c];
        let ny = pu.nw[2u * c + 1u];
        var plane = 0.0;
        for (var ii = 0u; ii < ny; ii = ii + 1u) {
            let y = same_parity(oy * k + 2 * (i32(ii) - 1) + qy, h);
            var line = 0.0;
            for (var jj = 0u; jj < nx; jj = jj + 1u) {
                let x = ox * k + qx + 2 * (i32(jj) - 1);
                line = opaque(line + opaque(weight(2u * c, jj) * mosaic_at(x, y)));
            }
            plane = opaque(plane + opaque(weight(2u * c + 1u, ii) * line));
        }
        if (pu.flags.x != 0u) {
            plane = opaque(plane * g);
        }
        dst[select(0u, 2u * pu.out.w, c == 1u) + o] = plane;
    }
}

fn bin_sigma_main(gid: vec3<u32>) {
    if (gid.x >= pu.frame.w || gid.y >= pu.out.y) {
        return;
    }
    let step = i32(pu.flags.w);
    let ox = i32(gid.x);
    let oy = i32(pu.out.x + gid.y);
    let sw = i32(pu.flags.y);
    let sh = i32(pu.flags.z);
    var s2 = 0.0;
    var n = 0.0;
    for (var cy = oy * step; cy < (oy + 1) * step; cy = cy + 1) {
        let yy = min(cy, sh - 1);
        for (var cx = ox * step; cx < (ox + 1) * step; cx = cx + 1) {
            let s = src[u32(yy) * pu.dims.z + u32(min(cx, sw - 1))];
            s2 = opaque(s2 + opaque(s * s));
            n = n + 1.0;
        }
    }
    var g = 1.0;
    if (pu.flags.x != 0u) {
        g = vig_gain((2 * ox + 1) * i32(pu.dims.w) - i32(pu.frame.x), (2 * oy + 1) * i32(pu.dims.w) - i32(pu.frame.y));
    }
    let root = sqrt(div_exact(s2, n));
    dst[(u32(oy) - pu.band.x) * pu.out.z + gid.x] = opaque(opaque(root * pu.f.y) * g);
}
