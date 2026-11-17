#!/usr/bin/env bash
# The scale validation of the spectral energy engine (plan 005, the author's decision 5): the runs that decide whether
# settings.engine = "spectral" may become a size-based default. Prepared, not run: it needs the europa slot.
#
#   europa claim INFERENCES <mission>
#   EUROPA_MISSION=<mission> tools/europa_detached.sh start 'export PYGEOINF0_WORKERS=30; bash benchmarks/validate_spectral.sh'
#   tools/europa_detached.sh wait <dir>; tools/europa_detached.sh fetch <dir> <local>
#
# Runs from the staged src/ ($PY = the inferences-main python, which has finufft 2.5.1). Every run prints its stages
# with wall time and peak memory (this process and the workers), the engine's claims and stats, and its accuracy
# against settings.accuracy; validate_spectral.py ends with a JSON line. Logs: validate_spectral/<run>.log, all of them
# also on stdout (job.log). Estimated 2-3 h in all at 30 workers, most of it the exact dense energies at 1e4 and the
# 1e5 rung; the runs are independent (a failure is logged and the next one starts).
#
#   every validate_spectral.py run compares two correction sets with one dense route: none (every pair closer than
#   r_corr = 0.25 l) and 0.3 (between rays only the pairs at sin(angle) < 0.3: pg.settings.collinear)
#   L3_3000, fan_3000      dense vs spectral: products, predicted data, ladder.solve end to end (8 segments per ray)
#   L3_10000, fan_10000    the same at 1e4 (exact energies ~10 min each at 30 workers)
#   L2_2000                the 2-D rung (straight chords, s = 3/2: the derived rate p = 3, every crossing corrected)
#   L3_1e5                 benchmarks/rung.py L3 100000 --segments 27 --engine spectral --collinear 0.3: the user code
#                          end to end at 1e5 rays (2.7e6 segments), no reference; the plan's counts and timings (pairs,
#                          corrections in the workers, transforms) and the estimate of the predicted-data error at the
#                          posterior mean. Not with every close pair: at 1e5 rays of 8 segments they are ~1.25e8
#                          (EXP-025), growing like n^2, which settings.memory_gb refuses or hours of corrections cost
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.:tests
export PYGEOINF0_WORKERS=${PYGEOINF0_WORKERS:-30}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16}
PY=${PY:-python}
OUT=validate_spectral
mkdir -p "$OUT"
run() {
  local name=$1; shift
  echo "=== $name: $* ($(date -u +%H:%M:%S) UTC, workers $PYGEOINF0_WORKERS)"
  local t0=$SECONDS
  "$PY" "$@" 2>&1 | tee "$OUT/$name.log"
  echo "=== $name: rc ${PIPESTATUS[0]}, $((SECONDS - t0)) s"
}
run L3_3000 benchmarks/validate_spectral.py L3 3000 --segments 8 --collinear none 0.3
run fan_3000 benchmarks/validate_spectral.py fan 3000 --segments 8 --collinear none 0.3
run L2_2000 benchmarks/validate_spectral.py L2 2000 --collinear none 0.3
run L3_10000 benchmarks/validate_spectral.py L3 10000 --segments 8 --collinear none 0.3
run fan_10000 benchmarks/validate_spectral.py fan 10000 --segments 8 --collinear none 0.3
run L3_1e5 benchmarks/rung.py L3 100000 --segments 27 --engine spectral --collinear 0.3
