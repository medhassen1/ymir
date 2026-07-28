//! `ymir-inspect` — dump the structure of a `.ymr` region file.
//!
//! ```text
//! ymir-inspect <region.ymr> [--columns] [--quiet]
//! ```
//!
//! Prints the header, the section directory, and a per-column summary, then
//! runs the region's selected rebuild stage and reports its digest. Exits
//! non-zero if the region fails structural validation.

use std::process::ExitCode;

use ymir::chunk;
use ymir::common::Status;
use ymir::format;
use ymir::parse;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = match args.iter().find(|a| !a.starts_with("--")) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: ymir-inspect <region.ymr> [--columns] [--quiet]");
            return ExitCode::from(2);
        }
    };
    let show_columns = args.iter().any(|a| a == "--columns");
    let quiet = args.iter().any(|a| a == "--quiet");

    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ymir-inspect: {path}: {e}");
            return ExitCode::from(2);
        }
    };

    let region = match parse::parse(&data) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ymir-inspect: {path}: not a valid region ({e})");
            return ExitCode::from(1);
        }
    };

    if !quiet {
        println!("{path}  ({} bytes)", data.len());
        println!("  version      {}", region.version);
        println!(
            "  flags        {:#06x}  -> stage `{}`",
            region.flags,
            ymir::stage_name(region.flags)
        );
        println!("  region       ({}, {})", region.region_x, region.region_z);
        println!("  chunks       {}", region.num_chunks);
        println!("  seed         {:#010x}", region.seed);
        println!("  world height {}", region.world_height);
        println!("  palette bits {}", region.bits);
        println!("  depth budget {}", region.depth);
        println!("  checksum     {:#06x} (advisory)", region.csum);

        println!("  sections:");
        for (name, s) in [
            (format::tag::CMAP, region.cmap),
            (format::tag::CDAT, region.cdat),
            (format::tag::PALT, region.palt),
            (format::tag::ENTS, region.ents),
            (format::tag::TILE, region.tile),
            (format::tag::LGTS, region.lgts),
            (format::tag::HGTS, region.hgts),
            (format::tag::BIOM, region.biom),
            (format::tag::STRC, region.strc),
            (format::tag::TICK, region.tick),
        ] {
            if s.is_present() {
                println!(
                    "    {:<6} offset {:>7}  len {:>7}",
                    format::tag_name(name),
                    s.offset,
                    s.len
                );
            }
        }
    }

    let n = region.worked_chunks();
    let mut ok = 0usize;
    let mut failed = 0usize;
    let mut sections_total = 0usize;
    let mut blocks_total = 0usize;

    for cid in 0..n {
        match chunk::decode(&region, cid) {
            Ok(col) => {
                ok += 1;
                sections_total += col.sections.len();
                blocks_total += col.block_count();
                if show_columns && !quiet {
                    println!(
                        "    column {cid:<5} sections {:<3} blocks {:<7} base_y {:<5} light {}",
                        col.sections.len(),
                        col.block_count(),
                        col.base_y,
                        col.has_light()
                    );
                }
            }
            Err(e) => {
                failed += 1;
                if show_columns && !quiet {
                    println!("    column {cid:<5} FAILED: {e}");
                }
            }
        }
    }

    if !quiet {
        println!(
            "  columns      {ok} decoded, {failed} failed, {sections_total} sections, {blocks_total} blocks"
        );
    }

    let status = ymir::verify(&data);
    let digest = ymir::rebuild(&data);
    println!("  verify       {status}");
    println!("  digest       {digest:#018x}");

    if status == Status::Ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
