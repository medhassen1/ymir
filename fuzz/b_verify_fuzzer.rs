#![no_main]
use libfuzzer_sys::fuzz_target;

// Structural validation only: parse the region, decode every column, and walk
// the structure templates. Stops before any derived data is rebuilt, so it
// exercises the decoder and the template recursion without the stage modules.
fuzz_target!(|data: &[u8]| {
    let _ = ymir::verify(data);
});
