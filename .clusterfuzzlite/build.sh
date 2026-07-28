#!/bin/bash -eu
# Build every ymir fuzz harness into $OUT.
#
# The repo root is the ymir crate; the fuzz project sits at ./fuzz and declares
# its own targets. Each harness is also given its seed corpus, packaged under
# the OSS-Fuzz naming convention so the runner picks it up.

cd "$SRC"

TARGETS="a_rebuild_fuzzer b_verify_fuzzer c_column_fuzzer"

cargo fuzz build -O

for t in $TARGETS; do
  cp "fuzz/target/x86_64-unknown-linux-gnu/release/$t" "$OUT/"
done

for t in $TARGETS; do
  seeds="fuzz/corpus/$t"
  if [ -d "$seeds" ]; then
    (cd "$seeds" && zip -q -r "$OUT/${t}_seed_corpus.zip" .)
  fi
done
