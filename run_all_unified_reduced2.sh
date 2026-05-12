#!/usr/bin/env bash
# Stage A driver: refit the unified 13-term equation (--variant f --term_set v2)
# on every animal under {3,8,10,16,28}weeks/, writing into runs_unified/.
# Stage B (Unified_age_summary.py) is invoked at the end.
#
# Usage:
#   ./run_all_unified.sh                       # run all 12 animals + summary
#   ./run_all_unified.sh --skip_fit            # only re-run Stage B (summary)
#   ./run_all_unified.sh --variant g           # base+cross without rec/disp
#   PYTHON=python3.11 ./run_all_unified.sh     # override interpreter

set -euo pipefail

# Avoid the "OMP: Error #15: libomp.dylib already initialized" abort that hits
# on macOS when matplotlib (or another lib) drags in a second OpenMP runtime
# mid-run. Affects only the OS-level guard, not the model — every animal
# completes consistently with this set.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/Juvenile_SEDF_final_reduced2.py"
SUMMARY="$SCRIPT_DIR/Unified_age_summary.py"

VARIANT="b"
TERM_SET="v2"
OUT_ROOT="runs_sokolis"
SKIP_FIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)   VARIANT="$2";   shift 2 ;;
    --term_set)  TERM_SET="$2";  shift 2 ;;
    --out_root)  OUT_ROOT="$2";  shift 2 ;;
    --skip_fit)  SKIP_FIT=1;     shift ;;
    -h|--help)
      grep "^#" "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

ANIMALS=(
  "3weeks/Lucid"      "3weeks/Rosie"
  "8weeks/Martini"
  "10weeks/Gus"       "10weeks/Mojito"   "10weeks/Pablo"
  "16weeks/Audi"      "16weeks/Oreon"    "16weeks/Sagittarius"
  "28weeks/Dodge"     "28weeks/Lexus"    "28weeks/Mercedes"
)

if [[ "$SKIP_FIT" -eq 0 ]]; then
  for rel in "${ANIMALS[@]}"; do
    dir="$SCRIPT_DIR/$rel"
    if [[ ! -d "$dir/data" ]]; then
      echo "  [skip] $rel — no data/ directory"; continue
    fi
    echo
    echo "==================================================="
    echo "  Stage A: $rel  (variant=$VARIANT, term_set=$TERM_SET)"
    echo "==================================================="
    (
      cd "$dir"
      "$PYTHON" "$SCRIPT" \
        --data_root data \
        --out_root "$OUT_ROOT" \
        --variant "$VARIANT" \
        --term_set "$TERM_SET"
    )
  done
fi

echo
echo "==================================================="
echo "  Stage B: aggregate parameters vs age"
echo "==================================================="
"$PYTHON" "$SUMMARY" \
  --root "$SCRIPT_DIR" \
  --out_root "$OUT_ROOT" \
  --variant "$VARIANT" \
  --term_set "$TERM_SET"
