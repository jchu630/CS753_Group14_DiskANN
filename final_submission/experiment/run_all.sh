#!/bin/bash
# Three seeds at 100K points, then one run at 1M points.
set -e
cd "$(dirname "$0")"
for s in 0 1 2; do python3 run_experiment.py 100000 1000 $s > log_n100000_s$s.txt 2>&1; done
python3 run_experiment.py 1000000 1000 0 > log_n1000000_s0.txt 2>&1
echo ALL_DONE
