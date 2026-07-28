#![no_main]
use libfuzzer_sys::fuzz_target;

// Drives the column decoder and the passes that read a decoded column directly,
// bypassing the header's stage selection. A region whose flags steer `rebuild`
// somewhere else still gets its columns meshed, relit and repacked here.
fuzz_target!(|data: &[u8]| {
    let region = match ymir::parse::parse(data) {
        Ok(r) => r,
        Err(_) => return,
    };
    let n = region.worked_chunks();

    // Budget derivation folds the whole decoded region.
    let slots = ymir::budget::pool_slots(&region, n);

    for cid in 0..n {
        let col = match ymir::chunk::decode(&region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let _ = ymir::seccache::flatten_column(&col);
        let _ = ymir::height::scan_column(&col);
        let _ = ymir::palette::repack_column(&col);
        let _ = ymir::relight::column_levels(&col);

        let builder = ymir::mesh::mesh_column(&col, slots);
        let table = ymir::weld::weld(builder.vertices());
        let _ = ymir::weld::triangulate(&table);
    }
});
