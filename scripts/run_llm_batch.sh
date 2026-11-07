#!/usr/bin/env bash
# Run LLM-designer runs through the subscription CLI (`claude -p`, never an API). Resumable: rerun the same command.
#   scripts/run_llm_batch.sh ARM SEED [SEED ...]
#   ARM: llm_param | llm_param_scalar | llm_free
# Environment: MODEL (default opus), EFFORT (optional), BUDGET (default 64), GEN_SIZE (default 3), EXP (default m1).
# Each generation is one `claude -p` call in a fresh empty directory under /var/tmp. Replies are cached before they are
# evaluated, so a stop (rate limit, Ctrl-C, reboot) never loses a paid reply or re-spends budget.
set -u
HERE="$(dirname "$(readlink -f "$0")")"
ROOT="$(dirname "$HERE")"
[ $# -ge 2 ] || { echo "usage: $0 ARM SEED [SEED ...]" >&2; exit 2; }
arm="$1"
shift
for seed in "$@"; do
  PYTHONPATH="$ROOT" "$HOME/projects/bin/sim-run" "$ROOT/.venv/bin/python" -m dumbrobot.m1 --exp "${EXP:-m1}" run \
    --arm "$arm" --seed "$seed" --budget "${BUDGET:-64}" --gen-size "${GEN_SIZE:-3}" --model "${MODEL:-opus}" --clips \
    ${EFFORT:+--effort "$EFFORT"}
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "stopped at $arm seed $seed (exit $rc, 3 = designer could not answer, 4 = disk floor); rerun the same command to resume" >&2
    exit "$rc"
  fi
done
