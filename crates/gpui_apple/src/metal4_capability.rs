//! Runtime capability probe for the Metal 4 renderer backend.
//!
//! Metal 4 requires **both** macOS 26+ *and* a GPU reporting the
//! `MTLGPUFamily::Metal4` family. Neither check alone is sufficient: the OS
//! gate exists because every `MTL4*` selector is simply unresolved on an
//! older system (invoking one would raise `doesNotRecognizeSelector:` rather
//! than fail gracefully), and the family gate exists because a sufficiently
//! new OS can still be running on a GPU that never gained Metal 4 support.
//!
//! This module intentionally knows nothing about how the renderer is
//! selected — see `renderer_select`, which is the only caller.

use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
#[cfg(target_os = "macos")]
use objc2_metal::MTLCopyAllDevices;
#[cfg(target_os = "ios")]
use objc2_metal::MTLCreateSystemDefaultDevice;
use objc2_metal::{MTLDevice, MTLGPUFamily};

const MIN_OS_VERSION: NSOperatingSystemVersion = NSOperatingSystemVersion {
    majorVersion: 26,
    minorVersion: 0,
    patchVersion: 0,
};

/// Whether this machine can run the Metal 4 renderer backend.
///
/// Checks the running OS version first (cheap, no GPU enumeration) and only
/// falls through to a device query when that gate passes.
pub fn metal4_available() -> bool {
    let os_supports_metal4 =
        NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(MIN_OS_VERSION);
    if !os_supports_metal4 {
        return false;
    }

    #[cfg(target_os = "macos")]
    {
        MTLCopyAllDevices()
            .iter()
            .any(|device| device.supportsFamily(MTLGPUFamily::Metal4))
    }
    // iOS / iPadOS: the one GPU is the system default device (iOS 26 aligns with macOS 26).
    #[cfg(target_os = "ios")]
    {
        MTLCreateSystemDefaultDevice()
            .is_some_and(|device| device.supportsFamily(MTLGPUFamily::Metal4))
    }
}
