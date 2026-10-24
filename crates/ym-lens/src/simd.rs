//! Four `f32` lanes with plain IEEE add, subtract and multiply: the vector the field optics' gather uses.
//!
//! SSE on x86-64 and NEON on AArch64 are part of those architectures' base instruction sets, so nothing is
//! detected at run time; any other architecture gets the same operations on an array. Every operation is one
//! correctly rounded IEEE single operation per lane, and nothing is fused, so all three give the same bits
//! (`docs/21` section 9). The unsafe blocks only load from and store to a `[f32; 4]`.

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::{
        __m128, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps, _mm_sub_ps,
    };

    /// Four `f32` lanes.
    #[derive(Clone, Copy)]
    pub struct F4(__m128);

    impl F4 {
        /// All lanes `x`.
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            // SAFETY: SSE is part of the x86-64 base instruction set; the intrinsic reads no memory.
            F4(unsafe { _mm_set1_ps(x) })
        }

        /// Loads four lanes.
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> F4 {
            // SAFETY: `a` is a valid reference to four `f32`; the load is unaligned.
            F4(unsafe { _mm_loadu_ps(a.as_ptr()) })
        }

        /// The four lanes.
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut out = [0.0f32; 4];
            // SAFETY: `out` is a valid place for four `f32`; the store is unaligned.
            unsafe { _mm_storeu_ps(out.as_mut_ptr(), self.0) };
            out
        }

        /// Lane-wise sum.
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            // SAFETY: register-only SSE operation on the x86-64 base instruction set.
            F4(unsafe { _mm_add_ps(self.0, o.0) })
        }

        /// Lane-wise difference.
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            // SAFETY: as `add`.
            F4(unsafe { _mm_sub_ps(self.0, o.0) })
        }

        /// Lane-wise product.
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            // SAFETY: as `add`.
            F4(unsafe { _mm_mul_ps(self.0, o.0) })
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32,
    };

    /// Four `f32` lanes.
    #[derive(Clone, Copy)]
    pub struct F4(float32x4_t);

    impl F4 {
        /// All lanes `x`.
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            // SAFETY: NEON is part of the AArch64 base instruction set; the intrinsic reads no memory.
            F4(unsafe { vdupq_n_f32(x) })
        }

        /// Loads four lanes.
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> F4 {
            // SAFETY: `a` is a valid reference to four `f32`.
            F4(unsafe { vld1q_f32(a.as_ptr()) })
        }

        /// The four lanes.
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut out = [0.0f32; 4];
            // SAFETY: `out` is a valid place for four `f32`.
            unsafe { vst1q_f32(out.as_mut_ptr(), self.0) };
            out
        }

        /// Lane-wise sum.
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            // SAFETY: register-only NEON operation on the AArch64 base instruction set.
            F4(unsafe { vaddq_f32(self.0, o.0) })
        }

        /// Lane-wise difference.
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            // SAFETY: as `add`.
            F4(unsafe { vsubq_f32(self.0, o.0) })
        }

        /// Lane-wise product (not fused with a following add).
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            // SAFETY: as `add`.
            F4(unsafe { vmulq_f32(self.0, o.0) })
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod imp {
    /// Four `f32` lanes.
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    impl F4 {
        /// All lanes `x`.
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            F4([x; 4])
        }

        /// Loads four lanes.
        #[inline(always)]
        pub fn load(a: &[f32; 4]) -> F4 {
            F4(*a)
        }

        /// The four lanes.
        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            self.0
        }

        /// Lane-wise sum.
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            F4(std::array::from_fn(|i| self.0[i] + o.0[i]))
        }

        /// Lane-wise difference.
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            F4(std::array::from_fn(|i| self.0[i] - o.0[i]))
        }

        /// Lane-wise product.
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            F4(std::array::from_fn(|i| self.0[i] * o.0[i]))
        }
    }
}

pub(crate) use imp::F4;

#[cfg(test)]
mod tests {
    use super::F4;

    #[test]
    fn lanes_follow_ieee_single_operations() {
        let a = F4::load(&[1.0, -2.5, 3.25, 0.1]);
        let b = F4::load(&[0.5, 4.0, -1.0, 0.2]);
        assert_eq!(a.add(b).to_array(), [1.5, 1.5, 2.25, 0.1f32 + 0.2f32]);
        assert_eq!(a.sub(b).to_array(), [0.5, -6.5, 4.25, 0.1f32 - 0.2f32]);
        assert_eq!(a.mul(b).to_array(), [0.5, -10.0, -3.25, 0.1f32 * 0.2f32]);
        assert_eq!(F4::splat(2.0).to_array(), [2.0; 4]);
    }
}
