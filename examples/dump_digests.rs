fn main() {
    let dir = "tests/fixtures";
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".ymr"))
        .collect();
    names.sort();
    for n in names {
        let path = format!("{dir}/{n}");
        let data = std::fs::read(&path).unwrap();
        let stem = n.trim_end_matches(".ymr");
        let status = ymir::verify(&data);
        let digest = ymir::rebuild(&data);
        println!("{stem:14} stage={:9} verify={:12} digest=0x{digest:016x}",
                 ymir::stage_name(u16::from_be_bytes([data[6], data[7]])), status.code());
    }
}
