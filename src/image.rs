use std::fs::OpenOptions;
use std::io::{self, BufWriter};
use std::path::Path;

use png::{BitDepth, ColorType, Decoder, Encoder};

#[derive(Debug)]
pub struct RgbImage {
    pub(crate) inner: Vec<u8>,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl RgbImage {
    pub const fn empty() -> Self {
        Self {
            inner: vec![],
            height: 0,
            width: 0,
        }
    }

    pub const fn from_raw(content: Vec<u8>, height: usize, width: usize) -> Self {
        Self {
            inner: content,
            height,
            width,
        }
    }

    pub fn load<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        let f = OpenOptions::new().read(true).open(path)?;
        let decoder = Decoder::new(f);
        let mut reader = decoder.read_info()?;
        let len = reader.output_buffer_size();
        let mut buf = vec![0; len];
        let info = reader.next_frame(&mut buf)?;
        match info.color_type {
            ColorType::Rgb => {}
            ColorType::Rgba => {
                for i in 0..len / 4 {
                    for j in 0..3 {
                        buf[i * 3 + j] = buf[i * 4 + j];
                    }
                }
                buf.truncate(3 * len / 4);
            }
            _ => panic!("unsupported format."),
        }

        Ok(Self {
            inner: buf,
            height: info.height as usize,
            width: info.width as usize,
        })
    }

    pub fn save<P>(&self, path: P) -> io::Result<()>
    where
        P: AsRef<Path>,
    {
        let f = OpenOptions::new().write(true).create(true).open(path)?;
        let w = BufWriter::new(f);
        let mut encoder = Encoder::new(w, self.width as u32, self.height as u32);
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(self.content())?;
        Ok(())
    }

    pub fn content(&self) -> &[u8] {
        &self.inner
    }

    pub fn content_mut(&mut self) -> &mut [u8] {
        &mut self.inner
    }
}

impl PartialEq for RgbImage {
    fn eq(&self, other: &Self) -> bool {
        if self.height != other.height || self.width != other.width {
            false
        } else {
            self.inner == other.inner
        }
    }
}

#[cfg(test)]
mod tests {
    use test::Bencher;

    use super::*;
    use crate::consts::*;

    #[test]
    fn eq() -> io::Result<()> {
        let img = RgbImage::load(ORIGINAL)?;
        let dummy = RgbImage {
            inner: vec![0u8; img.height * img.width],
            height: img.height,
            width: img.width,
        };
        assert_ne!(img, dummy);
        Ok(())
    }

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
