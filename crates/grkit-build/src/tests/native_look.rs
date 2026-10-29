use super::*;
use crate::firmware::{verify_firmware, FirmwareImage};
use grkit_core::native_tone::ToneSteps;
use sha2::{Digest, Sha256};

fn digest(curves: &[[u16; 256]]) -> String {
    let mut hash = Sha256::new();
    for value in curves.iter().flatten() {
        hash.update(value.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

/// The tone tables and curve sets of GR IV 1.11 give the live-view tables of the camera:
/// Standard with neutral steps, and Cinema (Yellow), type 18, with its steps. The digests
/// stand for the camera's tables, so the test holds no firmware data.
#[test]
#[ignore = "Requires local GRKIT_TEST_FIRMWARE_GR4"]
fn gr4_native_tone_matches_the_camera() -> Result<()> {
    let firmware = PathBuf::from(
        std::env::var("GRKIT_TEST_FIRMWARE_GR4").context("Set GRKIT_TEST_FIRMWARE_GR4")?,
    );
    let image = verify_firmware(&firmware, &target::GR4_111)?;
    let tables = crate::native_look::tone_tables(&image)?;
    let steps = |key, contrast, shadow, highlight| ToneSteps {
        key,
        contrast,
        shadow,
        highlight,
        input_scale: 0x2158,
    };
    let standard = tables.base_curve(&steps(0, 0, 0, 0));
    assert_eq!(
        digest(&[standard]),
        "a7551a5811f906b587c2dd8e0495a4943c0a1a0a734fdbba83bde4ce34f01748"
    );
    assert!(!crate::native_look::curve_set(&image, 11)?.enabled);

    let yellow = crate::native_look::curve_set(&image, 18)?;
    assert!(yellow.enabled && yellow.weight == 1);
    let base = tables.base_curve(&steps(-1, 3, 2, 0));
    assert_eq!(
        digest(&yellow.apply(&base)),
        "f472ad7dc0424ee9a1328575adc6a9aefe46701f96e08fd9de2b90f003df406a"
    );
    assert!(crate::native_look::curve_set(&image, 19)?.enabled);
    Ok(())
}

#[test]
fn native_tone_tables_need_gr4_111() -> Result<()> {
    let image = FirmwareImage {
        target: &target::GR3_210,
        body: Vec::new(),
        sha256: String::new(),
        decoded_sha256: String::new(),
    };
    let error = crate::native_look::tone_tables(&image).unwrap_err();
    assert!(error.to_string().contains("GR IV 1.11 only"), "{error}");
    Ok(())
}

/// Cinema (Yellow) and Cinema (Green) of GR IV 1.11 as effects for GR III: the tone offsets
/// of their default adjustments, their own MultiAxial sets (every level of every bank, as
/// INSTALL expands them), and their curve sets.
#[test]
#[ignore = "Requires local GRKIT_TEST_FIRMWARE_GR4"]
fn gr4_cinema_looks_become_effects() -> Result<()> {
    let firmware = PathBuf::from(
        std::env::var("GRKIT_TEST_FIRMWARE_GR4").context("Set GRKIT_TEST_FIRMWARE_GR4")?,
    );
    let image = verify_firmware(&firmware, &target::GR4_111)?;
    let standard = grkit_core::fit::STANDARD_LOOK;
    let icon = vec![0; grkit_core::effects::ICON_SIZE];
    for (image_type, offsets) in [(18, [-1, 3, 0, -2]), (19, [0, 1, 0, -3])] {
        let effect = crate::native_look::effect(&image, image_type, &standard, "cinema", "Cinema")?;
        let look = effect.native_look.clone().unwrap();
        assert_eq!(look.tone_offsets, offsets);
        assert_eq!(look.gains, [1024, 960, 896, 832, 768]);
        let set = crate::native_look::curve_set(&image, image_type)?;
        assert_eq!(effect.curves, set.mixed());
        let bytes = effect.encode(&icon)?;
        let (decoded, _) = grkit_core::effects::decode(&bytes)?;
        assert_eq!(decoded, effect);
    }
    // Standard has neutral offsets, and its look matrix converts to the target's Standard.
    let standard_effect = crate::native_look::effect(&image, 11, &standard, "std", "Std")?;
    assert_eq!(standard_effect.native_look.unwrap().tone_offsets, [0; 4]);
    for (got, want) in standard_effect.matrix.iter().zip(standard) {
        assert!(
            (i32::from(*got) - i32::from(want)).abs() <= 1,
            "{got} vs {want}"
        );
    }
    Ok(())
}
