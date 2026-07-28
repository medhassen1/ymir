//! Rebuild a single region file and print its digest. Used by the fixture
//! validation script to isolate which fixture trips a sanitizer.
fn main() {
    let path = std::env::args().nth(1).expect("usage: one_fixture <file>");
    let data = std::fs::read(&path).expect("readable fixture");
    let status = ymir::verify(&data);
    let digest = ymir::rebuild(&data);
    println!("{status} 0x{digest:016x}");
}
