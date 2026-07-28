#![no_main]
use libfuzzer_sys::fuzz_target;

// Runs a region's selected rebuild stage end to end. This is the widest entry
// point: the header's flag bits steer it into whichever stage the input asks
// for, so one harness reaches every subsystem.
fuzz_target!(|data: &[u8]| {
    let _ = ymir::rebuild(data);
});
