# nanoBragg GPU↔CPU parity harness

A permanent, in-repo parity harness. Point it at a GPU kernel binary and read
PASS/FAIL. The test corpus is *compiled* from a compact source spec, so coverage
is countable and reproducible.

## Two-zone layout (strict)

Committed source — pristine, read-only. Nothing built or run ever lands here:

    cuda/test/
      run.sh            entrypoint / orchestrator
      gen_cpu.sh        builds + behaviorally verifies the CPU reference
      Makefile          OUT-OF-SOURCE build (binaries go to the workbench)
      src/gen_cells.c   the compiler   (spec -> cells)
      src/metrics.c     the comparator (float32 image pair -> corr/sum_ratio/…)
      spec/             base.json, dimensions.jsonl, groups.json, plan.json
      cells/            compiled + LOCKED cells (committed source-of-truth)

Ephemeral build + run — gitignored, disposable:

    cuda/workbench/testrun/
      bin/            compiled binaries: gen_cells, metrics
      cpu/            CPU reference images (cached, keyed by cpu_args hash)
      out/            transient GPU output images
      cpu_build/      from-source CPU-reference build scratch
      results.tsv     run output
      cpu_manifest.txt CPU-reference provenance

`make` compiles `src/*.c` into `cuda/workbench/testrun/bin/`. `run.sh` reads
`spec/` + `cells/`, invokes the binaries, and writes cpu/out/results into the
workbench. The ONLY thing that ever writes into `cuda/test/` is a deliberate
`gen_cells` regeneration of `cells/`, reviewed and committed like a lockfile.

## The compile model

`spec/` is SOURCE. `gen_cells` is the COMPILER. `cells/*.jsonl` are the COMPILED,
LOCKED output — regenerate deliberately and review the diff. A cell is a scenario:
an exact GPU/CPU CLI plus id/axes/tags/gate/cost metadata. Different plans compile
to different suites.

    make                                                  # build tools -> testrun/bin
    testrun/bin/gen_cells grid320 spec cells/grid320.jsonl # compile the 320-suite

`plan.json` names the suites:

- `grid320` — Cartesian cross of crystal × crystal_size × grid320 × orientation
  (4×4×5×4 = 320). Reproduces the canonical parity grid cell-for-cell.
- `coverage` — main-effects: a baseline cell, then one cell per (dimension, value)
  that differs from that dimension's baseline (only that dimension changed).

## Running

    ./gen_cpu.sh                                   # establish/verify the CPU reference
    ./run.sh grid320 <gpu-kernel-binary>           # render, compare, gate, TSV
    column -t < ../workbench/testrun/results.tsv   # human-readable view

`run.sh <suite> <kernel-binary> [workdir]` asserts the device before rendering a
single pixel (see below), builds the C tools out-of-source, and for each cell:
renders the GPU kernel (`gpu_args`) and the CPU reference (`cpu_args`, cached by
hash), compares with `metrics`, applies the gate, and appends a TSV row
`cell corr sum_ratio ms verdict note`. It emits a final
`TIER <suite> PASS|FAIL n_pass/n_total` line and exits nonzero on any FAIL.

`NB_RANGE="lo-hi"` restricts a run to a 1-based inclusive cell-index range so a
long suite can run as several foreground invocations that append to one
`results.tsv`; the TIER tally is computed over the whole file.

## The gate (uniform, all precisions)

    PASS iff corr >= 0.9999 AND sum_ratio in [0.999, 1.001]

Pearson correlation + sum-ratio, identical for fp32/df64/fp64. fp32 is held to the
same gate deliberately, so float failures stay visible. `metrics` also prints
`max_rel worst_pixel_frac peak_max_rel worst_is_peak` as diagnostics (not gated).

## Invariants (every cell)

- Detector locked at 2048×2048.
- Desktop RTX 5090 only. This box has two GPUs and the CUDA runtime enumerates
  them INVERTED vs nvidia-smi; under `CUDA_DEVICE_ORDER=FASTEST_FIRST` the desktop
  card sorts first, so `CUDA_VISIBLE_DEVICES=0` selects it. `run.sh` proves the
  active device is the 170-SM RTX 5090 and prints it as proof; a mismatch is a
  hard refusal, not a warning.
- Deterministic: explicit step counts; `-nonoise` always (noise is host-only and
  non-deterministic — a separate regression, not here).
- Thickness cells pass `-oversample_thick` to the CPU side only (group `cpu_extra`).
- All cells pass an explicit `-hkl`, and renders run in an isolated working
  directory so nanoBragg's `Fdump.bin` auto-write/auto-load (a hardcoded filename
  in the CWD) cannot collide across cells.

## CPU reference provenance (gen_cpu.sh)

The CPU oracle must be `main` + `fix/phi0-stale-rotation` + `fix/subpixel-oversampling`.
Provenance is BEHAVIORAL, not by branch name: two canary cells — a phi-sweep cell
(sensitive to the phi0 fix) and an `-oversample_thick`, oversample>1 cell
(sensitive to the subpixel fix) — are rendered and compared against a "gold"
binary built from source (`main` + both branch diffs applied to `nanoBragg.c`) on
every invocation. Building gold from source makes the check non-circular (it
trusts no cached image) and self-adapting: if the fixes are already merged into
`main` the diffs are empty and gold == main, so the check still passes and the
eventual PR merge needs no edits here. If `cuda/workbench/nanoBragg_root` already
reproduces gold on both canaries it is reused; otherwise the freshly built gold is
installed. Provenance is recorded in `testrun/cpu_manifest.txt`. A parity result
measured against a reference that lacks either fix is invalid.

## Data

Uses the existing local `.hkl`/`.mat` inputs under `{data_root}` (repo-relative
`cuda/workbench`): the crystals at `{data_root}/crystals/*.hkl` and the AMAT inputs
at `{data_root}/A.mat` (105 B) and `{data_root}/scaled.hkl`. These large inputs are
gitignored and already present on the owner's box; this harness commits no crystal
data and touches no PDB. `run.sh` resolves `{data_root}` to an absolute path at run
time.
