# Out-of-source build for the nanoBragg parity harness.
#
# Compiles src/*.c into the workbench, never into this committed tree:
#   cuda/test/src/gen_cells.c  -> $(BINDIR)/gen_cells
#   cuda/test/src/metrics.c    -> $(BINDIR)/metrics
#
# BINDIR defaults to the ephemeral workbench run dir and may be overridden
# (run.sh points it at its workdir). No object files or binaries ever land
# under cuda/test/.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

# This Makefile lives in cuda/test/; '..' is cuda/, so the default target is
# cuda/workbench/testrun/bin.
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BINDIR  ?= $(abspath $(HERE)../workbench/testrun/bin)

# Build-time commit bake (Layer-2 ledger provenance): metrics is stamped with the
# commit HEAD was at when it was last compiled, queried at runtime via
# `metrics --build-commit`. gen_cells stays commit-agnostic -- its output is
# reviewed by diff, not stamped.
BUILD_COMMIT := $(shell git -C $(HERE) rev-parse HEAD 2>/dev/null || echo unknown)

BINS := $(BINDIR)/gen_cells $(BINDIR)/metrics

.PHONY: all clean gen_cells metrics
all: $(BINS)

# Convenience aliases so 'make metrics' / 'make gen_cells' work regardless of BINDIR.
gen_cells: $(BINDIR)/gen_cells
metrics:   $(BINDIR)/metrics

$(BINDIR)/gen_cells: src/gen_cells.c | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BINDIR)/metrics: src/metrics.c | $(BINDIR)
	$(CC) $(CFLAGS) -DNB_BUILD_COMMIT='"$(BUILD_COMMIT)"' -o $@ $< $(LDLIBS)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -f $(BINS)
