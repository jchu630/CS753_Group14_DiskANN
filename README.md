# Disk-Resident Graph Indexes for Billion-Scale ANNS

COMPSCI 753 (University of Auckland) research report, Group 14:
Zalán Varga, Jadon Chu, Xiangrong Jiang, Igor Kim.

The report reviews DiskANN (the seed paper), HNSW, NSG, SPANN, Starling, Filtered-DiskANN and BANG. It then tests whether pruning and disk layout should be designed together, using a small simulation of disk reads.

## Contents

| Path | What it is |
|---|---|
| [`final_submission/report.pdf`](final_submission/report.pdf) | The report as submitted |
| `final_submission/report.typ`, `refs.bib`, `fig_reads_recall.svg` | Typst source, bibliography and Fig. 2 |
| [`final_submission/experiment/`](final_submission/experiment/) | Code, logs and results for the experiment in Section V-C |

## Reproducing Table III

```
cd final_submission/experiment
pip install numpy numba
python3 aggregate.py      # prints Table III and the paired ratios from the stored results (seconds)
```

To rerun from scratch, see [`experiment/README.md`](final_submission/experiment/README.md). A run takes about 4 minutes at 100K points and about 45 minutes at 1M points on 2 cores.

The data are synthetic, distances are exact rather than PQ, and the code counts block reads rather than measuring time. The report and the experiment README discuss these limits.
