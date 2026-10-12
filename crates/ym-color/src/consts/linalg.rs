//! Small f64 linear algebra for 3x3 colour matrices.
//!
//! Matrices are row-major (`m[row][col]`) and act on column vectors: `out = M * v`.

/// A colour triple (three channels, f64).
pub type Vec3 = [f64; 3];

/// A row-major 3x3 matrix acting on column vectors.
pub type Mat3 = [[f64; 3]; 3];

/// The 3x3 identity matrix.
pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Matrix product `a * b`.
#[must_use]
pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Matrix-vector product `m * v`.
#[must_use]
pub fn mul_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose of `m`.
#[must_use]
pub fn transpose(m: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = m[j][i];
        }
    }
    out
}

/// Diagonal matrix with `d` on the diagonal.
#[must_use]
pub fn diag(d: Vec3) -> Mat3 {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}

/// Determinant of `m`.
#[must_use]
pub fn det(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of `m` by the adjugate, or `None` when `m` is singular (|det| below 1e-300) or not finite.
#[must_use]
pub fn inv(m: &Mat3) -> Option<Mat3> {
    let d = det(m);
    if !d.is_finite() || d.abs() < 1e-300 {
        return None;
    }
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let adj = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = adj[i][j] / d;
        }
    }
    Some(out)
}

/// Sum of each row of `m`.
#[must_use]
pub fn row_sums(m: &Mat3) -> Vec3 {
    [
        m[0][0] + m[0][1] + m[0][2],
        m[1][0] + m[1][1] + m[1][2],
        m[2][0] + m[2][1] + m[2][2],
    ]
}

/// Largest absolute element-wise difference between `a` and `b`.
#[must_use]
pub fn max_abs_diff(a: &Mat3, b: &Mat3) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}

/// Element-wise linear blend `(1 - t) * a + t * b`. At `t == 0` the result is exactly `a`, at `t == 1` exactly `b`.
#[must_use]
pub fn lerp(a: &Mat3, b: &Mat3, t: f64) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (1.0 - t) * a[i][j] + t * b[i][j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_round_trips() {
        let m = [[0.5, 0.2, 0.1], [0.1, 0.9, -0.3], [0.0, 0.2, 1.1]];
        let i = inv(&m).expect("invertible");
        let p = mul(&m, &i);
        assert!(max_abs_diff(&p, &IDENTITY) < 1e-15);
    }

    #[test]
    fn singular_has_no_inverse() {
        let m = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]];
        assert!(inv(&m).is_none());
    }

    #[test]
    fn lerp_endpoints_are_exact() {
        let a = [[0.3, 0.1, 0.6], [0.2, 0.2, 0.6], [1.0 / 3.0; 3]];
        let b = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(lerp(&a, &b, 0.0), a);
        assert_eq!(lerp(&a, &b, 1.0), b);
    }
}
