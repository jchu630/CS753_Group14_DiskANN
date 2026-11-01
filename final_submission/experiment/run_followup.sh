#!/bin/bash
# Wait for the 1M run, then rerun the 100K seeds (deterministic) so they include config D'.
cd "$(dirname "$0")"
until grep -q "done" log_n1000000_s0.txt 2>/dev/null; do sleep 20; done
for s in 0 1 2; do python3 run_experiment.py 100000 1000 $s > log_n100000_s$s.txt 2>&1; done
echo FOLLOWUP_DONE
