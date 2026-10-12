//! A very small TIFF/DNG reader for the export tests: directories, tag values and uncompressed
//! strips. Enough to check what the writer put in a file without a second TIFF library.
#![allow(dead_code)]

/// One directory entry.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub tag: u16,
    pub typ: u16,
    pub count: u32,
    /// Where the 4-byte value/offset field is.
    field_at: usize,
}

pub struct Tiff<'a> {
    pub bytes: &'a [u8],
    pub le: bool,
}

impl<'a> Tiff<'a> {
    pub fn new(bytes: &'a [u8]) -> Option<Self> {
        let le = match bytes.get(..4)? {
            b"II*\0" => true,
            b"MM\0*" => false,
            _ => return None,
        };
        Some(Self { bytes, le })
    }

    fn u16_at(&self, at: usize) -> u16 {
        let b: [u8; 2] = self.bytes[at..at + 2].try_into().expect("2 bytes");
        if self.le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        }
    }

    fn u32_at(&self, at: usize) -> u32 {
        let b: [u8; 4] = self.bytes[at..at + 4].try_into().expect("4 bytes");
        if self.le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        }
    }

    pub fn ifd0(&self) -> usize {
        self.u32_at(4) as usize
    }

    pub fn entries(&self, ifd: usize) -> Vec<Entry> {
        let n = self.u16_at(ifd) as usize;
        (0..n)
            .map(|i| {
                let at = ifd + 2 + 12 * i;
                Entry {
                    tag: self.u16_at(at),
                    typ: self.u16_at(at + 2),
                    count: self.u32_at(at + 4),
                    field_at: at + 8,
                }
            })
            .collect()
    }

    pub fn find(&self, ifd: usize, tag: u16) -> Option<Entry> {
        self.entries(ifd).into_iter().find(|e| e.tag == tag)
    }

    fn type_size(typ: u16) -> usize {
        match typ {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => 1,
        }
    }

    /// The bytes of an entry's value.
    pub fn data(&self, e: &Entry) -> &'a [u8] {
        let len = Self::type_size(e.typ) * e.count as usize;
        if len <= 4 {
            &self.bytes[e.field_at..e.field_at + len]
        } else {
            let at = self.u32_at(e.field_at) as usize;
            &self.bytes[at..at + len]
        }
    }

    /// SHORT and LONG values as numbers.
    pub fn uints(&self, e: &Entry) -> Vec<u32> {
        let d = self.data(e);
        match e.typ {
            3 => (0..e.count as usize)
                .map(|i| {
                    let b: [u8; 2] = d[2 * i..2 * i + 2].try_into().expect("2");
                    u32::from(if self.le {
                        u16::from_le_bytes(b)
                    } else {
                        u16::from_be_bytes(b)
                    })
                })
                .collect(),
            4 => (0..e.count as usize)
                .map(|i| {
                    let b: [u8; 4] = d[4 * i..4 * i + 4].try_into().expect("4");
                    if self.le {
                        u32::from_le_bytes(b)
                    } else {
                        u32::from_be_bytes(b)
                    }
                })
                .collect(),
            other => panic!("not an integer entry (type {other})"),
        }
    }

    pub fn uint(&self, ifd: usize, tag: u16) -> Option<u32> {
        self.find(ifd, tag).map(|e| self.uints(&e)[0])
    }

    /// The offsets in the `SubIFDs` tag of `ifd`.
    pub fn sub_ifds(&self, ifd: usize) -> Vec<usize> {
        self.find(ifd, 330)
            .map(|e| self.uints(&e).into_iter().map(|v| v as usize).collect())
            .unwrap_or_default()
    }

    /// Width, height, samples per pixel and bits of an uncompressed 16-bit RGB image, and its
    /// samples (all strips joined).
    pub fn uncompressed_u16_rgb(&self, ifd: usize) -> (usize, usize, Vec<u16>) {
        assert_eq!(self.uint(ifd, 259), Some(1), "compression must be none");
        let w = self.uint(ifd, 256).expect("width") as usize;
        let h = self.uint(ifd, 257).expect("height") as usize;
        assert_eq!(self.uint(ifd, 277), Some(3), "three samples");
        let offsets = self.uints(&self.find(ifd, 273).expect("strip offsets"));
        let counts = self.uints(&self.find(ifd, 279).expect("strip counts"));
        let mut raw = Vec::new();
        for (o, c) in offsets.iter().zip(&counts) {
            raw.extend_from_slice(&self.bytes[*o as usize..(*o + *c) as usize]);
        }
        let samples: Vec<u16> = raw
            .chunks_exact(2)
            .map(|b| {
                let b: [u8; 2] = [b[0], b[1]];
                if self.le {
                    u16::from_le_bytes(b)
                } else {
                    u16::from_be_bytes(b)
                }
            })
            .collect();
        assert_eq!(samples.len(), w * h * 3);
        (w, h, samples)
    }
}

/// The XMP packet of a file in any of the formats (it is plain text in all of them).
pub fn xmp_of(bytes: &[u8]) -> Option<String> {
    let begin = find(bytes, b"<?xpacket begin")?;
    let end_marker = find(&bytes[begin..], b"<?xpacket end")?;
    let end = begin
        + end_marker
        + bytes[begin + end_marker..]
            .iter()
            .position(|&b| b == b'>')?
        + 1;
    Some(String::from_utf8_lossy(&bytes[begin..end]).into_owned())
}

/// The text of one XMP property (`ym:Look`) from a packet.
pub fn xmp_property(xmp: &str, name: &str) -> Option<String> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let from = xmp.find(&open)? + open.len();
    let to = from + xmp[from..].find(&close)?;
    Some(xmp[from..to].to_owned())
}

pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
