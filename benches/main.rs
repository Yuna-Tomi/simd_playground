#![feature(test)]
extern crate test;

use std::io;

use test::Bencher;

use simd_playground as simd;

use simd::{
    test_util::{test, FilterType},
    ConvProcessor,
};

macro_rules! bench {
    ($bencher:ident, $const_filter_type:expr, $method:ident) => {{
        const FIL_TY: FilterType = $const_filter_type;
        const K: usize = FIL_TY.size();
        // run test with assertion disabled
        test(Some($bencher), false, FIL_TY, ConvProcessor::<K>::$method)
    }};
}

mod image {
    use super::*;

    use simd::{consts::*, image::RgbImage};

    #[bench]
    fn load(b: &mut Bencher) -> io::Result<()> {
        b.iter(|| {
            RgbImage::load(ORIGINAL).expect("cannot load");
        });
        Ok(())
    }

    #[bench]
    fn save(b: &mut Bencher) -> io::Result<()> {
        let img = RgbImage::load(ORIGINAL)?;
        b.iter(|| {
            img.save(BACKUP).expect("cannot save");
        });
        Ok(())
    }
}

mod naive_benches {
    use super::*;

    #[bench]
    fn box3_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(3), naive2)
    }

    #[bench]
    fn box5_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(5), naive2)
    }

    #[bench]
    fn box7_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(7), naive2)
    }

    #[bench]
    fn box9_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(9), naive2)
    }

    #[bench]
    fn box11_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(11), naive2)
    }

    #[bench]
    fn box13_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(13), naive2)
    }

    #[bench]
    fn box15_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(15), naive2)
    }

    #[bench]
    fn box17_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(17), naive2)
    }

    #[bench]
    fn box19_naive2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(19), naive2)
    }
}

#[cfg(all(any(target_arch = "aarch64"), all(target_feature = "neon")))]
mod simd_benches {
    use super::*;

    #[bench]
    fn box3_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(3), simd1)
    }

    #[bench]
    fn box5_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(5), simd1)
    }

    #[bench]
    fn box7_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(7), simd1)
    }

    #[bench]
    fn box9_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(9), simd1)
    }

    #[bench]
    fn box11_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(11), simd1)
    }

    #[bench]
    fn box13_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(13), simd1)
    }

    #[bench]
    fn box15_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(15), simd1)
    }

    #[bench]
    fn box17_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(17), simd1)
    }

    #[bench]
    fn box19_simd1(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(19), simd1)
    }

    #[bench]
    fn box3_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(3), simd2)
    }

    #[bench]
    fn box5_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(5), simd2)
    }

    #[bench]
    fn box7_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(7), simd2)
    }

    #[bench]
    fn box9_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(9), simd2)
    }

    #[bench]
    fn box11_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(11), simd2)
    }

    #[bench]
    fn box13_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(13), simd2)
    }

    #[bench]
    fn box15_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(15), simd2)
    }

    #[bench]
    fn box17_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(17), simd2)
    }

    #[bench]
    fn box19_simd2(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(19), simd2)
    }

    #[bench]
    fn box3_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(3), simd3)
    }

    #[bench]
    fn box5_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(5), simd3)
    }

    #[bench]
    fn box7_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(7), simd3)
    }

    #[bench]
    fn box9_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(9), simd3)
    }

    #[bench]
    fn box11_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(11), simd3)
    }

    #[bench]
    fn box13_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(13), simd3)
    }

    #[bench]
    fn box15_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(15), simd3)
    }

    #[bench]
    fn box17_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(17), simd3)
    }

    #[bench]
    fn box19_simd3(b: &mut Bencher) -> io::Result<()> {
        bench!(b, FilterType::Box(19), simd3)
    }
}
