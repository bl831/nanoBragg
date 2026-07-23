#!/bin/bash
# =============================================================================
# ledger_append.sh -- ingest one run's results.tsv into the Layer-2 ledger.
#
#   ledger_append.sh <results.tsv> [ledger_dir=cuda/test/ledger]
#
# Parses the "# key=value" trailer run.sh appends after the final chunk (TIER
# line, kernel/cpu_ref md5, device, driver/cuda, utc/host, commit, and -- once
# emitted by a run built with this Phase-2 run.sh -- gpu_name/cpu_model/
# precision) into:
#   - ONE row appended to ledger/runs.tsv   (the run's provenance)
#   - one row per DATA line appended to ledger/results.tsv, mapped from
#     results.tsv columns (cell corr sum_ratio ms verdict note) to
#     run_id cell verdict corr sum_ratio ms. REJECT/SKIP/BLOCKED rows already
#     carry corr/sum_ratio as "-" in results.tsv and pass through unchanged.
#
# run_id = <UTCcompact>-<kernel_md5's first 8 hex>-<host>  (UTCcompact strips
# '-' and ':' from the utc timestamp). Idempotent: re-running on a results.tsv
# already ingested (same run_id) prints "already ingested: <run_id>" and exits
# 0 without appending anything twice.
#
# Pure shell + awk. No jq/python.
# =============================================================================
set -u

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # cuda/test

die(){ echo "ERROR: $*" >&2; exit 2; }

[ $# -ge 1 ] || die "usage: ledger_append.sh <results.tsv> [ledger_dir]"
RESULTS="$1"
LEDGER_DIR="${2:-$TEST_DIR/ledger}"
[ -s "$RESULTS" ] || die "results.tsv missing/empty: $RESULTS"

RUNS_TSV="$LEDGER_DIR/runs.tsv"
RES_TSV="$LEDGER_DIR/results.tsv"
[ -f "$RUNS_TSV" ] || die "ledger runs.tsv missing: $RUNS_TSV"
[ -f "$RES_TSV" ]  || die "ledger results.tsv missing: $RES_TSV"

# --- parse the results.tsv "# key=value" trailer -----------------------------
PROV_UTC="$(sed -n 's/^# utc=\([^ ]*\).*/\1/p' "$RESULTS" | head -1)"
PROV_HOST="$(sed -n 's/.*host=\([^ ]*\).*/\1/p' "$RESULTS" | head -1)"
PROV_DRIVER="$(sed -n 's/^# driver=\([^ ]*\).*/\1/p' "$RESULTS" | head -1)"
PROV_COMMIT="$(sed -n 's/^# commit=\(.*\)/\1/p' "$RESULTS" | head -1)"
PROV_SUITE="$(sed -n 's/^# TIER \([^ ]*\) .*/\1/p' "$RESULTS" | head -1)"
KERNEL_MD5="$(sed -n 's/^# kernel=.*md5=\([0-9a-fA-F]*\).*/\1/p' "$RESULTS" | head -1)"
CPU_MD5="$(sed -n 's/^# cpu_ref=.*md5=\([0-9a-fA-F]*\).*/\1/p' "$RESULTS" | head -1)"
# gpu_name: prefer the clean key; fall back to the name="..." embedded in # device=.
PROV_GPUNAME="$(sed -n 's/^# gpu_name=\(.*\)/\1/p' "$RESULTS" | head -1)"
[ -n "$PROV_GPUNAME" ] || PROV_GPUNAME="$(sed -n 's/.*device=.*name="\([^"]*\)".*/\1/p' "$RESULTS" | head -1)"
PROV_CPUMODEL="$(sed -n 's/^# cpu_model=\(.*\)/\1/p' "$RESULTS" | head -1)"
PROV_PRECISION="$(sed -n 's/^# precision=\(.*\)/\1/p' "$RESULTS" | head -1)"

: "${PROV_UTC:=unknown}" "${PROV_HOST:=unknown}" "${PROV_DRIVER:=unknown}" \
  "${PROV_COMMIT:=unknown}" "${PROV_SUITE:=unknown}" "${KERNEL_MD5:=unknown}" \
  "${CPU_MD5:=unknown}" "${PROV_GPUNAME:=unknown}" "${PROV_CPUMODEL:=unknown}" \
  "${PROV_PRECISION:=unknown}"

UTC_COMPACT="$(tr -d ':-' <<<"$PROV_UTC")"
RUN_ID="${UTC_COMPACT}-${KERNEL_MD5:0:8}-${PROV_HOST}"

# --- idempotency: a run_id already present in runs.tsv is a no-op -----------
if awk -F'\t' -v id="$RUN_ID" '/^#/{next} $1==id{f=1} END{exit !f}' "$RUNS_TSV"; then
  echo "already ingested: $RUN_ID"
  exit 0
fi

# --- append ONE runs.tsv row (11 cols, header order) -------------------------
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$RUN_ID" "$PROV_UTC" "$PROV_HOST" "$PROV_GPUNAME" "$PROV_DRIVER" "$PROV_CPUMODEL" \
  "$PROV_COMMIT" "$PROV_SUITE" "$PROV_PRECISION" "$KERNEL_MD5" "$CPU_MD5" >> "$RUNS_TSV"

# --- append one results.tsv row per DATA line ---------------------------------
# results.tsv columns: cell corr sum_ratio ms verdict note
# ledger row:          run_id cell verdict corr sum_ratio ms
awk -F'\t' -v OFS='\t' -v id="$RUN_ID" '
  /^#/ { next }
  NF < 6 { next }
  $1 == "cell" { next }
  { print id, $1, $5, $2, $3, $4 }
' "$RESULTS" >> "$RES_TSV"

NAPPENDED="$(awk -F'\t' '!/^#/ && NF>=6 && $1!="cell"' "$RESULTS" | wc -l)"
echo "ledger: appended run $RUN_ID (1 runs row + $NAPPENDED results rows)"
