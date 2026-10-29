//! Native looks of a GR IV 1.11 image: the tables of `grkit_core::native_tone`, and an
//! effect (`grkit_core::native_look`) that gives a look of the image to another camera.
use anyhow::{ensure, Context, Result};
use grkit_core::effects::Effect;
use grkit_core::native_look::{NativeLook, BANKS, LEVELS};
use grkit_core::native_tone::{Curve, CurveSet, ToneTables};

use crate::firmware::FirmwareImage;
use crate::target;

/// The addresses of one firmware build.
struct Layout {
    /// Nine u32 input gains of the key steps.
    gains: u32,
    /// Pointers to the three families; each family is five curve pointers.
    families: u32,
    /// 97 u16 ease weights.
    ease: u32,
    /// The u16 unit of mid gray.
    unit: u32,
    /// The type table of curve sets, three blocks of 34 entries.
    curve_sets: u32,
    /// The default key, contrast, highlight, and shadow values (0..=8, 4 neutral) of the
    /// types whose defaults are not neutral. The firmware writes them as instruction
    /// immediates into its defaults record (GetDefaultValueImageKey and the like read
    /// it), so they are fixed for each build and checked against the camera.
    default_tone: &'static [(u32, [u8; 4])],
}

const GR4_111: Layout = Layout {
    gains: 0x53b4_d644,
    families: 0x5503_2890,
    ease: 0x5407_b1f4,
    unit: 0x5407_e710,
    curve_sets: 0x5505_3138,
    // Cinema (Yellow) and Cinema (Green).
    default_tone: &[(18, [3, 7, 4, 2]), (19, [4, 5, 4, 1])],
};
const CURVE_SET_BLOCK: u32 = 34 * 4;

fn layout(image: &FirmwareImage) -> Result<&'static Layout> {
    ensure!(
        image.target.name == target::GR4_111.name,
        "Native tone tables are known for GR IV 1.11 only, not {}",
        image.target.firmware
    );
    Ok(&GR4_111)
}

fn word(image: &FirmwareImage, address: u32) -> Result<u32> {
    Ok(u32::from_le_bytes(image.read(address, 4)?.try_into()?))
}

fn curve(image: &FirmwareImage, address: u32) -> Result<Curve> {
    let bytes = image.read(address, 512)?;
    Ok(std::array::from_fn(|i| {
        u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])
    }))
}

/// The tone tables of the image, checked for their shape.
pub fn tone_tables(image: &FirmwareImage) -> Result<ToneTables> {
    let layout = layout(image)?;
    let mut gains = [0; 9];
    for (i, gain) in gains.iter_mut().enumerate() {
        *gain = word(image, layout.gains + 4 * i as u32)?;
    }
    let mut families = [[[0; 256]; 5]; 3];
    for (f, family) in families.iter_mut().enumerate() {
        let pointers = word(image, layout.families + 4 * f as u32)?;
        for (c, slot) in family.iter_mut().enumerate() {
            *slot = curve(image, word(image, pointers + 4 * c as u32)?)?;
        }
    }
    let ease = image.read(layout.ease, 2 * 97)?;
    let unit = image.read(layout.unit, 2)?;
    let tables = ToneTables {
        gains,
        families,
        ease: std::array::from_fn(|i| u16::from_le_bytes([ease[2 * i], ease[2 * i + 1]])),
        unit: u16::from_le_bytes([unit[0], unit[1]]),
    };
    tables
        .validate()
        .with_context(|| format!("Unexpected tone tables in {}", image.target.firmware))?;
    Ok(tables)
}

/// The curve set of an Image Control type. The three blocks of the type table must agree.
pub fn curve_set(image: &FirmwareImage, image_type: u32) -> Result<CurveSet> {
    let layout = layout(image)?;
    ensure!(
        image_type < 34,
        "Image Control type {image_type} out of range"
    );
    let address = word(image, layout.curve_sets + 4 * image_type)?;
    for block in 1..3 {
        ensure!(
            word(
                image,
                layout.curve_sets + block * CURVE_SET_BLOCK + 4 * image_type
            )? == address,
            "The curve set blocks differ for type {image_type}"
        );
    }
    let flags = image.read(address + 4, 2)?;
    let set = CurveSet {
        enabled: flags[0] != 0,
        weight: flags[1],
        curves: [
            curve(image, word(image, address + 8)?)?,
            curve(image, word(image, address + 12)?)?,
            curve(image, word(image, address + 16)?)?,
        ],
    };
    ensure!(
        set.weight <= 3
            && set
                .curves
                .iter()
                .all(|c| c.windows(2).all(|pair| pair[0] <= pair[1] + 64) && c[255] <= 0x4000),
        "Unexpected curve set for type {image_type}"
    );
    Ok(set)
}

/// The offsets of a type's default key, contrast, highlight, and shadow values from neutral.
pub fn tone_offsets(image: &FirmwareImage, image_type: u32) -> Result<[i8; 4]> {
    let layout = layout(image)?;
    Ok(layout
        .default_tone
        .iter()
        .find(|(t, _)| *t == image_type)
        .map_or([0; 4], |(_, values)| values.map(|v| v as i8 - 4)))
}

fn rows(m: &[i16; 9]) -> [[f64; 3]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| f64::from(m[r * 3 + c]) / 8192.0))
}

fn product(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

fn inverse(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let cofactor = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]
    };
    let det: f64 = (0..3).map(|c| m[0][c] * cofactor(0, c)).sum();
    ensure!(det.abs() > 1e-9, "Singular look matrix");
    Ok(std::array::from_fn(|r| {
        std::array::from_fn(|c| cofactor(c, r) / det)
    }))
}

/// The look matrix (Q13, at record +4) of an Image Control type.
fn look_matrix(image: &FirmwareImage, image_type: u32) -> Result<[i16; 9]> {
    let tables = image
        .target
        .image_control
        .context("The image has no Image Control")?
        .tables;
    let record = word(image, tables[0] + 4 * image_type)?;
    let bytes = image.read(word(image, record + 4)?, 18)?;
    Ok(std::array::from_fn(|k| {
        i16::from_le_bytes([bytes[2 * k], bytes[2 * k + 1]])
    }))
}

/// The MultiAxial set of an Image Control type with `tone_offsets`.
fn multiaxial(image: &FirmwareImage, image_type: u32, tone_offsets: [i8; 4]) -> Result<NativeLook> {
    let tables = image
        .target
        .image_control
        .context("The image has no Image Control")?
        .tables;
    let set = word(image, tables[1] + 4 * image_type)?;
    let pointer = |k: u32| word(image, set + 4 * k);
    let u16s = |address: u32, n: usize| -> Result<Vec<u16>> {
        let bytes = image.read(address, n * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect())
    };
    let bounds = u16s(pointer(0)?, 16)?;
    let selectors = image.read(pointer(1)?, 20)?;
    let control = u16s(pointer(4)?, 21)?;
    let matrices = image.read(word(image, pointer(2)? + 0x10)?, BANKS * LEVELS * 36)?;
    let at = |bank: usize, level: usize, k: usize| {
        let offset = ((bank * LEVELS + level) * 9 + k) * 4;
        i32::from_le_bytes(matrices[offset..offset + 4].try_into().unwrap())
    };
    let levels: [[[i32; 9]; LEVELS]; BANKS] =
        std::array::from_fn(|b| std::array::from_fn(|l| std::array::from_fn(|k| at(b, l, k))));
    NativeLook::from_levels(
        tone_offsets,
        std::array::from_fn(|k| bounds[k]),
        std::array::from_fn(|k| selectors[k]),
        std::array::from_fn(|k| control[k]),
        &levels,
    )
    .with_context(|| format!("Unexpected MultiAxial set for type {image_type}"))
}

/// An effect that gives Image Control type `image_type` of the image to a camera whose
/// Standard look matrix is `standard` (`grkit_core::fit::STANDARD_LOOK` for GR III).
///
/// The matrix is the type's look matrix relative to the image's Standard, on the target's
/// Standard. The curves are the type's curve set after its weight; the target camera
/// applies them to its own base curve, which its tone generator makes from the tone
/// offsets. The native look holds the type's MultiAxial set and tone offsets. A type
/// without its own curve set keeps linear curves.
pub fn effect(
    image: &FirmwareImage,
    image_type: u32,
    standard: &[i16; 9],
    id: &str,
    name: &str,
) -> Result<Effect> {
    let set = curve_set(image, image_type)?;
    let curves = if set.enabled {
        set.mixed()
    } else {
        [std::array::from_fn(|i| (i * 16383 / 255) as u16); 3]
    };
    ensure!(
        curves.iter().flatten().all(|v| *v <= 16383),
        "A curve of type {image_type} leaves the 14-bit range"
    );
    let source = rows(&look_matrix(image, image_type)?);
    let base = rows(&look_matrix(image, 11)?);
    let matrix = product(&product(&source, &inverse(&base)?), &rows(standard));
    let mut q13 = [0i16; 9];
    for (k, slot) in q13.iter_mut().enumerate() {
        let v = (matrix[k / 3][k % 3] * 8192.0).round();
        ensure!(
            (-32768.0..=32767.0).contains(&v),
            "The look matrix leaves the Q13 range"
        );
        *slot = v as i16;
    }
    Ok(Effect {
        id: id.to_owned(),
        name: name.to_owned(),
        flags: 0,
        matrix: q13,
        curves,
        post_matrix: None,
        selective: None,
        hue_sectors: None,
        native_look: Some(multiaxial(
            image,
            image_type,
            tone_offsets(image, image_type)?,
        )?),
    })
}
