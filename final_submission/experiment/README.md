# Layout-aware pruning experiment (Section V-C of the report)

A small, self-contained simulation of disk reads for a DiskANN-style graph index.
Vectors stay in RAM; every time the search needs a vertex's neighbour list it
"reads" the 4 KB block holding that vertex, and we count those reads.

## What it compares

| Config | Graph | Layout | Search |
|---|---|---|---|
| A | Vamana (2 passes) | by vertex ID | DiskANN vertex search |
| B | Vamana (2 passes) | by vertex ID | Starling block search |
| C | Vamana (2 passes) | BNF (Starling) | block search |
| D' | same graph as D | C's BNF, fixed | block search |
| D | Vamana + 1 standard pass (control) | its own BNF | block search |
| E | Vamana + 1 pass with rule V1 (favour co-located, tau = 1.25) | C's BNF, fixed | block search |
| E' | same graph as E | its own BNF | block search |
| F | Vamana + 1 pass with rule V2 (block-mates as implicit edges) | C's BNF, fixed | block search |

Settings follow Starling's BIGANN segment: R = 31, 256-byte records, 16 records
per 4 KB block; Vamana L = 75, alpha = 1.2; BNF 8 iterations; block search
sigma = 0.3. Distances are exact (no PQ) for routing and re-ranking.

Data are synthetic (SIFT1M was not downloadable in the environment used):
100 Gaussian clusters in 128-d, random orientation, variances decaying as 1/j,
1,000 held-out queries, exact ground truth.

Naming note: the report's Table 3 row "D" is config D' here (`D2_control_extra_pass_fixed_layout`),
because E and F also keep C's layout fixed, which makes D' the like-for-like control. Config D here
(own BNF layout) is used only for the layout re-run comparison in the report's results.

## How to run

```
pip install numpy numba
python3 run_experiment.py 100000 1000 0     # n, queries, seed  (~4 min on 2 cores)
python3 run_experiment.py 1000000 1000 0    # ~45 min on 2 cores
python3 aggregate.py                        # prints the numbers used in the report
```

Each run writes `results_n{N}_s{SEED}.json` (summary per config) and
`sweeps_n{N}_s{SEED}.csv` (recall and mean reads for every search list size).

## Caveats (also stated in the report)

* Synthetic data, not SIFT/BIGANN.
* Exact distances instead of PQ for routing.
* Reads are counted, time is not: block search computes up to 16 distances per
  read, vertex search 1, and that compute cost is not in these numbers.
* Unpaired, differences of about 10% between C, D, E and F are within seed-to-seed
  variation at n = 100K. Paired by seed, F needs 2-7% fewer reads than D' in every
  seed and at every recall level (`aggregate.py` prints the paired ratios), and E
  matches D' because V1 changes only about 1% of edges, almost all to block-mates.
