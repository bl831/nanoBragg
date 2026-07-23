#!/bin/bash
# =============================================================================
# reseed_golden.sh -- widen/refresh a golden file's data rows from a validated
# results.tsv, while PRESERVING the old golden's '#' header comment lines and
# refusing to silently change any cell's expected verdict.
#
#   reseed_golden.sh <results.tsv> <old_golden.tsv> <new_golden.tsv>
#
# For each DATA row of results.tsv (columns: cell corr sum_ratio ms verdict
# note), writes a 4-col golden row 'cell verdict corr sum_ratio' (tab-sep) to
# <new_golden.tsv>, preceded by every '#' comment line copied verbatim from
# <old_golden.tsv> (in order). If any cell's verdict in results.tsv differs
# from its verdict in the OLD golden, this is a FLIP: the script aborts
# nonzero and prints every flipped cell instead of writing <new_golden.tsv> --
# a verdict change must be reviewed and re-seeded deliberately, never silently
# picked up as a side effect of widening columns.
#
# Pure awk; no jq/python. Never overwrites <new_golden.tsv> on abort.
# =============================================================================
set -u

die(){ echo "ERROR: $*" >&2; exit 2; }

[ $# -ge 3 ] || die "usage: reseed_golden.sh <results.tsv> <old_golden.tsv> <new_golden.tsv>"
RESULTS="$1"; OLD="$2"; NEW="$3"
[ -s "$RESULTS" ] || die "results.tsv missing/empty: $RESULTS"
[ -s "$OLD" ]     || die "old golden missing/empty: $OLD"

TMP="$(mktemp "${NEW}.XXXXXX" 2>/dev/null)" || die "mktemp failed for $NEW"

awk -F'\t' -v OFS='\t' '
  # pass 1: OLD golden (first file) -- collect header comments verbatim, and
  # the OLD verdict per cell (2-col or 4-col row, verdict is always $2).
  FNR==NR {
    if ($0 ~ /^#/) {
      # rewrite the column-header comment to the widened 4-col form; copy the rest verbatim
      if ($0 ~ /^# cell([ \t]|$)/) comments[++nc] = "# cell" OFS "verdict" OFS "corr" OFS "sum_ratio"
      else comments[++nc] = $0
      next
    }
    if (NF >= 2 && $1 != "") { oldv[$1] = $2 }
    next
  }
  # pass 2: results.tsv (second file) -- data rows only.
  /^#/ { next }
  NF < 6 { next }
  {
    cell = $1; corr = $2; sr = $3; verdict = $5
    if (cell == "" || cell == "cell") next
    if (!(cell in newv)) order[++no] = cell   # first occurrence wins (chunked reruns may repeat a cell)
    newv[cell] = verdict; newcorr[cell] = corr; newsr[cell] = sr
  }
  END {
    nflip = 0; flipmsg = ""
    for (i = 1; i <= no; i++) {
      c = order[i]
      if ((c in oldv) && oldv[c] != newv[c]) {
        nflip++
        flipmsg = flipmsg "\n  " c " : golden=" oldv[c] " actual=" newv[c]
      }
    }
    if (nflip > 0) {
      print "ABORT: " nflip " verdict flip(s) vs old golden (not writing new golden):" flipmsg > "/dev/stderr"
      exit 1
    }
    for (i = 1; i <= nc; i++) print comments[i]
    for (i = 1; i <= no; i++) {
      c = order[i]
      print c, newv[c], newcorr[c], newsr[c]
    }
    print "reseed_golden: " no " cell(s), 0 flips vs " ARGV[1] > "/dev/stderr"
  }
' "$OLD" "$RESULTS" > "$TMP"
rc=$?

if [ $rc -ne 0 ]; then
  rm -f "$TMP"
  exit "$rc"
fi

mv "$TMP" "$NEW"
echo "reseed_golden: wrote $NEW"
