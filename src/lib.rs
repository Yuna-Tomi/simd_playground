#![feature(stdsimd)]
#![feature(test)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)] // incomplete feature
#![feature(unboxed_closures)]
extern crate test;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::mem;

use crate::image::RgbImage;

pub mod consts;
pub mod image;
mod util;

pub mod test_util {
    pub use crate::util::test_util::*;
}

#[derive(Debug)]
struct ConvKernel<const K: usize> {
    inner: Vec<f32>,
    pub(crate) div: Option<f32>,
}

impl<const K: usize> ConvKernel<K> {
    pub fn new(filter: &[f32], avg: bool) -> Self {
        if filter.len() != K * K {
            panic!("inconsistent filter size {} for K={}", filter.len(), K);
        }
        if K % 2 == 0 || K < 3 {
            panic!("only odd number >= 3 is available for kernel size")
        }
        let div = if avg {
            let sum = filter.iter().sum();
            if sum == 0. {
                panic!("cannot calculate average on filter with weights of total 0.");
            }
            Some(sum)
        } else {
            None
        };

        Self {
            inner: filter.to_vec(),
            div,
        }
    }

    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.inner[i * K + j]
    }
}

#[derive(Debug)]
pub struct ConvProcessor<const K: usize> {
    kernel: ConvKernel<K>,
}

const C: usize = 3;
impl<const K: usize> ConvProcessor<K> {
    pub fn new(filter: &[f32], avg: bool) -> Self {
        Self {
            kernel: ConvKernel::<K>::new(filter, avg),
        }
    }

    pub fn naive1(&self, src: &RgbImage) -> RgbImage {
        let h = src.height;
        let w = src.width;
        let half = K / 2;
        let xend = w - half;
        let yend = h - half;
        let mut dst = vec![0u8; h * w * C]; // 0 padding

        for y in half..yend {
            for x in half..xend {
                for c in 0..C {
                    // RGB
                    let mut t: f32 = 0.;
                    for i in 0..K {
                        for j in 0..K {
                            let index = (y - half + i) * w * C + (x - half + j) * C + c;
                            t += src.content()[index] as f32 * self.kernel.at(i, j);
                        }
                    }
                    if let Some(div) = self.kernel.div {
                        t /= div;
                    }
                    let index = y * w * C + x * C + c;
                    dst[index] = t.clamp(u8::MIN as f32, u8::MAX as f32) as u8;
                }
            }
        }
        RgbImage::from_raw(dst, h, w)
    }

    pub fn naive2(&self, src: &RgbImage) -> RgbImage {
        let h = src.height;
        let w = src.width;
        let half = K / 2;
        let xend = w - half;
        let yend = h - half;
        let mut dst = vec![0u8; h * w * C]; // 0 padding

        for y in half..yend {
            for x in half..xend {
                let mut rgb: [f32; 3] = [0.; C];
                for i in 0..K {
                    for j in 0..K {
                        for (c, pix) in rgb.iter_mut().enumerate() {
                            let index = (y - half + i) * w * C + (x - half + j) * C + c;
                            *pix += src.content()[index] as f32 * self.kernel.at(i, j);
                        }
                    }
                }
                let base_index = y * w * C + x * C;
                for c in 0..C {
                    let mut t = rgb[c];
                    if let Some(div) = self.kernel.div {
                        t /= div;
                    }
                    dst[base_index + c] = t.clamp(u8::MIN as f32, u8::MAX as f32) as u8;
                }
            }
        }
        RgbImage::from_raw(dst, h, w)
    }

    #[cfg(all(any(target_arch = "aarch64"), target_feature = "neon"))]
    pub fn simd1(&self, src: &RgbImage) -> RgbImage {
        let h = src.height;
        let w = src.width;
        let half = K / 2;
        let xend = w - half;
        let yend = h - half;
        let mut dst = vec![0u8; h * w * C]; // 0 padding

        // calc 4 cells with simd in parallel
        // x coordinate of center pixel will be half+0~3, +4~7, ... half+(w-half*2 - (w-half*2)%4 -4 + 0~3)
        // remnants will be processed in serial (= peel loop)
        let simd_end = w - half - (w - 2 * half) % 4;

        let simd_loop = |x: usize, y: usize, dst: &mut [u8]| {
            let mut vt = unsafe { crate::util::init_float32x4x3(0.) };
            for i in 0..K {
                for j in 0..K {
                    let kern = unsafe { vdupq_n_f32(self.kernel.at(i, j)) };
                    let base_index = (y - half + i) * w * C + (x - half + j) * C;
                    let mut s4 = [0.; 4];
                    let mut prepare = |c: usize| -> float32x4_t {
                        // prepare simd register
                        for (z, s) in s4.iter_mut().enumerate() {
                            // +z in second axis and +c in third axis
                            *s = src.content()[base_index + z * C + c] as f32;
                        }
                        unsafe { vld1q_f32(s4.as_ptr()) }
                    };
                    let vs = float32x4x3_t(prepare(0), prepare(1), prepare(2));

                    unsafe {
                        vt.0 = vfmaq_f32(vt.0, vs.0, kern);
                        vt.1 = vfmaq_f32(vt.1, vs.1, kern);
                        vt.2 = vfmaq_f32(vt.2, vs.2, kern);
                    }
                }

                let base_index = y * w * C + x * C;
                let mut t4 = [0.; 4];
                for (c, &v) in [vt.0, vt.1, vt.2].iter().enumerate() {
                    unsafe {
                        vst1q_f32(t4.as_mut_ptr(), v);
                    }
                    for z in 0..4 {
                        let mut t = t4[z];
                        if let Some(div) = self.kernel.div {
                            t /= div;
                        }
                        dst[base_index + z * C + c] = t.clamp(u8::MIN as f32, u8::MAX as f32) as u8;
                    }
                }
            }
        };

        // main execution
        for y in half..yend {
            for x in (half..simd_end).step_by(4) {
                simd_loop(x, y, &mut dst);
            }

            for x in simd_end..xend {
                self.peel_loop(x, y, src, &mut dst);
            }
        }
        RgbImage::from_raw(dst, h, w)
    }

    fn peel_loop(&self, x: usize, y: usize, src: &RgbImage, dst: &mut [u8]) {
        let w = src.width;
        let half = K / 2;
        let mut rgb: [f32; 3] = [0.; C];
        for i in 0..K {
            for j in 0..K {
                for (c, pix) in rgb.iter_mut().enumerate() {
                    let index = (y - half + i) * w * C + (x - half + j) * C + c;
                    *pix += src.content()[index] as f32 * self.kernel.at(i, j);
                }
            }
        }
        let base_index = y * w * C + x * C;
        for c in 0..C {
            let mut t = rgb[c];
            if let Some(div) = self.kernel.div {
                t /= div;
            }
            dst[base_index + c] = t.clamp(u8::MIN as f32, u8::MAX as f32) as u8;
        }
    }
}

#[cfg(all(any(target_arch = "aarch64"), target_feature = "neon"))]
impl<const K: usize> ConvProcessor<K>
where
    [(); (K / 2 + 1) / 2 + 1]: Sized,
{
    pub fn simd2(&self, src: &RgbImage) -> RgbImage {
        let h = src.height;
        let w = src.width;
        let half = K / 2;
        let xend = w - half;
        let yend = h - half;
        let mut dst = vec![0u8; h * w * C]; // 0 padding

        // calc 4 cells with simd in parallel
        // x coordinate of center pixel will be half+0~3, +4~7, ... half+(w-half*2 - (w-half*2)%4 -4 + 0~3)
        // remnants will be processed in serial (= peel loop)
        let simd_end = w - half - (w - 2 * half) % 4;

        let simd_loop = |x: usize, y: usize, dst: &mut [u8]| {
            let mut vt = unsafe { crate::util::init_float32x4x3(0.) };
            for i in 0..K {
                // We process 2*half+4 elements(x3, RGB channel) in a row here
                // then number of simd registers simd register is ceil(half/2 + 1).
                let mut shared = unsafe { [mem::zeroed::<float32x4x3_t>(); (K / 2 + 1) / 2 + 1] };
                let len = shared.len();
                let base_index = (y - half + i) * w * C + (x - half) * C;
                let mut s4 = [0.; 4];

                let mut load = |k: usize, c: usize, ft: usize| -> float32x4_t {
                    // ft := four or two
                    debug_assert!(ft == 2 || ft == 4);
                    let base_index = base_index + k * 4 * C;
                    for (z, s) in s4.iter_mut().enumerate().take(ft) {
                        // +z in second axis and +c in third axis
                        *s = src.content()[base_index + z * C + c] as f32;
                    }
                    unsafe { vld1q_f32(s4.as_ptr()) }
                };

                // fill shared[k]
                let mut make = |k: usize, ft: usize| {
                    shared[k] = float32x4x3_t(load(k, 0, ft), load(k, 1, ft), load(k, 2, ft))
                };

                for k in 0..len - 1 {
                    make(k, 4)
                }
                let ft = if half % 2 == 1 { 2 } else { 4 };

                // have to care about 2 elements at the tail
                make(len - 1, ft);

                for j in 0..K {
                    let kern = unsafe { vdupq_n_f32(self.kernel.at(i, j)) };
                    let regi = j / 4;
                    let offset = j % 4;
                    let vext = match offset {
                        0 => vextq_f32::<0>,
                        1 => vextq_f32::<1>,
                        2 => vextq_f32::<2>,
                        3 => vextq_f32::<3>,
                        _ => unreachable!(),
                    };

                    let vs = if offset != 0 {
                        // here guaranteed that regi+1 is valid for index.
                        unsafe {
                            float32x4x3_t(
                                vext(shared[regi].0, shared[regi + 1].0),
                                vext(shared[regi].1, shared[regi + 1].1),
                                vext(shared[regi].2, shared[regi + 1].2),
                            )
                        }
                    } else {
                        shared[regi]
                    };

                    unsafe {
                        vt.0 = vfmaq_f32(vt.0, vs.0, kern);
                        vt.1 = vfmaq_f32(vt.1, vs.1, kern);
                        vt.2 = vfmaq_f32(vt.2, vs.2, kern);
                    }
                }

                let base_index = y * w * C + x * C;
                for (c, &v) in [vt.0, vt.1, vt.2].iter().enumerate() {
                    let mut t4 = [0.; 4];
                    unsafe {
                        vst1q_f32(t4.as_mut_ptr(), v);
                    }
                    for z in 0..4 {
                        let mut t = t4[z];
                        if let Some(div) = self.kernel.div {
                            t /= div;
                        }
                        dst[base_index + z * C + c] = t.clamp(u8::MIN as f32, u8::MAX as f32) as u8;
                    }
                }
            }
        };

        // main execution
        for y in half..yend {
            for x in (half..simd_end).step_by(4) {
                simd_loop(x, y, &mut dst);
            }

            for x in simd_end..xend {
                self.peel_loop(x, y, src, &mut dst);
            }
        }
        RgbImage::from_raw(dst, h, w)
    }
}

// Helper macro to pack float32x4_t into uint8x16_t
// Ugly hack: $c should be tuple indice.
// $v is expected to be
#[rustfmt::skip]
macro_rules! vec4_cvt {
    ($v:ident, $c:tt) => {{
        vqmovn_high_u16(
            vqmovn_u16(vqmovn_high_u32(vqmovn_u32(vcvtq_u32_f32($v[0].$c)),
                                                  vcvtq_u32_f32($v[1].$c))),
                       vqmovn_high_u32(vqmovn_u32(vcvtq_u32_f32($v[2].$c)),
                                                  vcvtq_u32_f32($v[3].$c)),
        )
    }};
}

#[cfg(all(any(target_arch = "aarch64"), target_feature = "neon"))]
impl<const K: usize> ConvProcessor<K>
where
    [(); (K + 1) / 4 + 4]: Sized,
    [(); K + 12]: Sized,
{
    pub fn simd3(&self, src: &RgbImage) -> RgbImage {
        let h = src.height;
        let w = src.width;
        let half = K / 2;
        let xend = w - half;
        let yend = h - half;
        let mut dst = vec![0u8; h * w * C]; // 0 padding

        // read/write 16 elements in parallel
        let simd_end = w - half - (w - 2 * half) % 16;

        let simd_loop = |x: usize, y: usize, dst: &mut [u8]| {
            let mut vts = unsafe { crate::util::init_multiple_float32x4x3::<4>(0.) };
            for i in 0..K {
                let mut shared = unsafe { [mem::zeroed::<float32x4x3_t>(); (K + 1) / 4 + 4] };
                let base_index = (y - half + i) * w * C + (x - half) * C;

                let load16 = |shared: &mut [float32x4x3_t], b: usize| {
                    let base_index = base_index + b * C;
                    // deinterleaved loading
                    let sc = unsafe { vld3q_u8(&src.content()[base_index]) };
                    #[rustfmt::skip]
                    let cvt = |z: usize, s: uint8x16_t| -> float32x4_t {
                        unsafe {
                            match z {
                                0 => vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(s))))),  // 0..4 th uint8 to float32
                                1 => vcvtq_f32_u32(vmovl_high_u16(        vmovl_u8(vget_low_u8(s)))),   // 4..8 th uint8 to float32
                                2 => vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_high_u8(       s)))),   // 9..12 th uint8 to float32
                                3 => vcvtq_f32_u32(vmovl_high_u16(        vmovl_high_u8(       s))),    // 12..15 th uint8 to float32
                                _ => unreachable!(),
                            }
                        }
                    };
                    for z in 0..4 {
                        shared[b + z].0 = cvt(z, sc.0);
                        shared[b + z].1 = cvt(z, sc.1);
                        shared[b + z].2 = cvt(z, sc.2);
                    }
                };

                let load8 = |shared: &mut [float32x4x3_t], b: usize| {
                    let base_index = base_index + b * C;
                    let sc = unsafe { vld3_u8(&src.content()[base_index]) };
                    #[rustfmt::skip]
                    let cvt = |z: usize, s: uint8x8_t| -> float32x4_t {
                        unsafe {
                            match z {
                                0 => vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(s)))),  // 0..4 th uint8 to float32
                                1 => vcvtq_f32_u32(vmovl_high_u16(        vmovl_u8(s))),   // 4..8 th uint8 to float32
                                _ => unreachable!(),
                            }
                        }
                    };
                    for z in 0..2 {
                        shared[b + z].0 = cvt(z, sc.0);
                        shared[b + z].1 = cvt(z, sc.1);
                        shared[b + z].2 = cvt(z, sc.2);
                    }
                };

                let load4or2 = |shared: &mut [float32x4x3_t], b: usize, ft: usize| {
                    debug_assert!(ft == 2 || ft == 4);
                    let base_index = base_index + b * 4 * C;
                    let mut s4 = [0.; 4];
                    let mut load = |c: usize| -> float32x4_t {
                        for (z, s) in s4.iter_mut().enumerate().take(ft) {
                            *s = src.content()[base_index + z * C + c] as f32;
                        }
                        unsafe { vld1q_f32(s4.as_ptr()) }
                    };
                    shared[b] = float32x4x3_t(load(0), load(1), load(2));
                };

                // If kernel size is very large, loading all first like below will impact performance,
                // (eviction from simd registers to stack may occur many times with large kernel)
                // but practically kernel size is 3x3 or 5x5 and we don't have to think it seriously.
                let mut base = 0;
                let mut remains = half * 2 + 16; // # of loading elements
                while remains > 0 {
                    match remains {
                        _r @ 16.. => {
                            load16(&mut shared, base);
                            remains -= 16;
                            base += 4;
                        }
                        _r @ 8.. => {
                            load8(&mut shared, base);
                            remains -= 8;
                            base += 2;
                        }
                        _r @ 4.. => {
                            load4or2(&mut shared, base, 4);
                            remains -= 4;
                            base += 1;
                        }
                        _r @ 2.. => {
                            load4or2(&mut shared, base, 2);
                            remains -= 2;
                        }
                        _ => unreachable!(),
                    }
                    if remains < 2 {
                        break;
                    }
                }

                /* below inplementation is slow, maybe due to eviction from registers to stack memories.  */
                // let mut vss = unsafe { [mem::zeroed::<float32x4x3_t>(); K + 12] };
                // for (s, vs) in vss.iter_mut().enumerate() {
                //     let regi = s / 4;
                //     let offset = s % 4;
                //     let vext = match offset {
                //         0 => vextq_f32::<0>,
                //         1 => vextq_f32::<1>,
                //         2 => vextq_f32::<2>,
                //         3 => vextq_f32::<3>,
                //         _ => unreachable!(),
                //     };

                //     // here guaranteed that regi+1 is valid for index.
                //     *vs = if offset != 0 {
                //         unsafe {
                //             float32x4x3_t(
                //                 vext(shared[regi].0, shared[regi + 1].0),
                //                 vext(shared[regi].1, shared[regi + 1].1),
                //                 vext(shared[regi].2, shared[regi + 1].2),
                //             )
                //         }
                //     } else {
                //         shared[regi]
                //     };
                // }

                for j in 0..K {
                    let kern = unsafe { vdupq_n_f32(self.kernel.at(i, j)) };
                    for (z, vt) in vts.iter_mut().enumerate().take(4) {
                        let s = z * 4 + j;
                        let regi = s / 4;
                        let offset = s % 4;
                        let vext = match offset {
                            0 => vextq_f32::<0>,
                            1 => vextq_f32::<1>,
                            2 => vextq_f32::<2>,
                            3 => vextq_f32::<3>,
                            _ => unreachable!(),
                        };

                        // here guaranteed that regi+1 is valid for index.
                        let vs = if offset != 0 {
                            unsafe {
                                float32x4x3_t(
                                    vext(shared[regi].0, shared[regi + 1].0),
                                    vext(shared[regi].1, shared[regi + 1].1),
                                    vext(shared[regi].2, shared[regi + 1].2),
                                )
                            }
                        } else {
                            shared[regi]
                        };

                        unsafe {
                            vt.0 = vfmaq_f32(vt.0, vs.0, kern);
                            vt.1 = vfmaq_f32(vt.1, vs.1, kern);
                            vt.2 = vfmaq_f32(vt.2, vs.2, kern);
                        }

                        /* see comments on commented out implementations above */
                        // unsafe {
                        //     vts[z].0 = vfmaq_f32(vts[z].0, vss[s].0, kern);
                        //     vts[z].1 = vfmaq_f32(vts[z].1, vss[s].1, kern);
                        //     vts[z].2 = vfmaq_f32(vts[z].2, vss[s].2, kern);
                        // }
                    }
                }
            }
            if let Some(div) = self.kernel.div {
                let vdiv = unsafe { vdupq_n_f32(div) };
                for vt in &mut vts {
                    unsafe {
                        vt.0 = vdivq_f32(vt.0, vdiv);
                        vt.1 = vdivq_f32(vt.1, vdiv);
                        vt.2 = vdivq_f32(vt.2, vdiv);
                    }
                }
            }
            let base_index = y * w * C + x * C;
            unsafe {
                vst3q_u8(
                    &mut dst[base_index],
                    uint8x16x3_t(vec4_cvt!(vts, 0), vec4_cvt!(vts, 1), vec4_cvt!(vts, 2)),
                );
            }
        };

        // main execution
        for y in half..yend {
            for x in (half..simd_end).step_by(16) {
                simd_loop(x, y, &mut dst);
            }

            for x in simd_end..xend {
                self.peel_loop(x, y, src, &mut dst);
            }
        }
        RgbImage::from_raw(dst, h, w)
    }
}

#[cfg(test)]
pub mod tests {

    use std::io;

    use super::*;
    use crate::util::test_util::{test, FilterType};

    // check filters for ConvProcessor::$method
    // use macro here due to test multiple constant generic parameter
    macro_rules! check {
        ($method:ident, $($k:literal)*) => {{
            for &ty in [ $(FilterType::Box($k),)* FilterType::Sobel,].iter() {
                match ty.size() {
                    $(
                        // run test with assertion enabled
                        $k => test(None, true, ty, ConvProcessor::<$k>::$method)?,
                    )*
                    _ => unreachable!(),
                }
            }
            Ok(())
        }};
    }

    macro_rules! config  {
        ($macro_name:ident, $($k:literal),* $(,)?) => {
            macro_rules! $macro_name {
                ($method:ident) => {{
                    check!($method, $($k)*)
                }};
            }
        };
    }

    // you can specify which size of kernels are tested by adding odd numbers inside check!()
    config!(check_all, 3, 5, 7, 9, 11, 13, 15, 17, 19);

    #[test]
    fn naive2() -> io::Result<()> {
        check_all!(naive2)
    }

    #[cfg(all(any(target_arch = "aarch64"), all(target_feature = "neon")))]
    mod simd_tests {
        use super::*;

        #[test]
        fn simd1() -> io::Result<()> {
            check_all!(simd1)
        }

        #[test]
        fn simd2() -> io::Result<()> {
            check_all!(simd2)
        }

        #[test]
        fn simd3() -> io::Result<()> {
            check_all!(simd3)
        }
    }
}
