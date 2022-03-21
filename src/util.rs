use std::mem;

#[cfg(all(any(target_arch = "aarch64"), target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(any(target_arch = "aarch64"), target_feature = "neon"))]
#[inline]
pub unsafe fn init_multiple_float32x4x3<const N: usize>(value: f32) -> [float32x4x3_t; N] {
    let mut init = [mem::zeroed::<float32x4x3_t>(); N];
    for elem in init.iter_mut() {
        *elem = init_float32x4x3(value);
    }
    init
}

#[cfg(all(any(target_arch = "aarch64"), target_feature = "neon"))]
#[inline]
pub unsafe fn init_float32x4x3(value: f32) -> float32x4x3_t {
    float32x4x3_t(vdupq_n_f32(value), vdupq_n_f32(value), vdupq_n_f32(value))
}

pub mod test_util {
    use std::io;

    use test::Bencher;

    use crate::{consts::*, image::RgbImage, ConvProcessor};

    #[derive(Debug, Clone, Copy)]
    pub enum FilterType {
        Box(usize),
        Sobel,
    }

    impl FilterType {
        pub fn answer_path(&self) -> String {
            match self {
                FilterType::Box(k) => format!("img/box_ans_{}x{}.png", k, k),
                FilterType::Sobel => SOBEL_ANS.to_string(),
            }
        }

        pub fn filter(&self) -> Vec<f32> {
            match self {
                &FilterType::Box(k) => vec![1.; k * k],
                FilterType::Sobel => SOBEL_FILTER.to_vec(),
            }
        }

        pub const fn avg(&self) -> bool {
            match self {
                FilterType::Box(_) => true,
                FilterType::Sobel => false,
            }
        }

        pub const fn size(&self) -> usize {
            match self {
                &FilterType::Box(k) => k,
                FilterType::Sobel => 3,
            }
        }
    }

    // confirm answer image is valid before test
    fn make<const K: usize>(ty: FilterType) -> io::Result<(RgbImage, ConvProcessor<K>)> {
        let img = RgbImage::load(ORIGINAL)?;
        let layer = ConvProcessor::<K>::new(&ty.filter(), ty.avg());
        layer.naive1(&img).save(ty.answer_path())?;
        Ok((img, layer))
    }

    pub fn test<const K: usize, F>(
        b: Option<&mut Bencher>,
        enable_assertion: bool,
        ty: FilterType,
        f: F,
    ) -> io::Result<()>
    where
        F: Fn(&ConvProcessor<K>, &RgbImage) -> RgbImage,
    {
        let (img, layer) = make::<K>(ty)?;
        let processed = &mut RgbImage::empty(); // initialize with dummy
        *processed = f(&layer, &img);

        if enable_assertion && *processed != RgbImage::load(ty.answer_path())? {
            processed.save(DEBUG)?;
            panic!("invalid calculation in {:?}", ty);
        }

        if let Some(b) = b {
            b.iter(|| *processed = f(&layer, &img));
        }
        Ok(())
    }
}
