"""
Runs the layout / pruning comparison and writes results.json and sweeps.csv.

Usage:  python3 run_experiment.py N_POINTS [N_QUERIES] [SEED]
"""
import json
import sys
import time

import numpy as np

from ann_layout import (EPS, R_DEG, block_members, bnf, bnp, build_vamana, ground_truth,
                        id_layout, make_data, overlap_ratio, reads_at_recall, sweep,
                        vamana_pass)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ALPHA, L_BUILD, TAU, SIGMA = 1.2, 75, 1.25, 0.3
Ls = [10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 320]
out = {"n": n, "nq": nq, "seed": seed, "R": R_DEG, "eps": EPS, "alpha": ALPHA,
       "L_build": L_BUILD, "tau": TAU, "sigma": SIGMA, "timings_s": {}, "configs": {}}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tick(name, t0):
    out["timings_s"][name] = round(time.time() - t0, 1)
    log(name, out["timings_s"][name], "s")


t = time.time()
X, Q = make_data(n, nq, center_scale=0.3, spec_decay=1.0, seed=seed)
gt = ground_truth(X, Q, k=100)
tick("data_and_ground_truth", t)

t = time.time()
G0, D0, start = build_vamana(X, R=R_DEG, L=L_BUILD, alpha=ALPHA, seed=seed, batch_size=max(2000, n // 200))
tick("build_vamana_2_passes", t)

t = time.time()
P_id = id_layout(n)
P_bnp = bnp(G0, D0, n, EPS)
P_bnf = bnf(G0, D0, P_bnp)
tick("layouts_bnp_bnf", t)


def third_pass(mode, layout, tau=1.0):
    g, d = G0.copy(), D0.copy()
    vamana_pass(X, g, d, start, L_BUILD, ALPHA, R_DEG, np.random.default_rng(seed + 7),
                mode=mode, tau=tau, blk=layout, batch_size=max(2000, n // 200))
    return g, d


t = time.time(); Gc, Dc = third_pass(0, P_id); tick("control_extra_standard_pass", t)
t = time.time(); G1, D1 = third_pass(1, P_bnf, TAU); tick("v1_favour_colocated_pass", t)
t = time.time(); G2, D2 = third_pass(2, P_bnf); tick("v2_implicit_blockmates_pass", t)

t = time.time()
P_c = bnf(Gc, Dc, bnp(Gc, Dc, n, EPS))     # control graph gets its own BNP -> BNF layout
P_1r = bnf(G1, D1, bnp(G1, D1, n, EPS))     # V1 graph, its own BNP -> BNF layout
tick("layouts_for_new_graphs", t)

configs = [
    # name, graph, degrees, layout, block search?
    ("A_diskann_vertex_search_id_layout", G0, D0, P_id, False),
    ("B_block_search_id_layout", G0, D0, P_id, True),
    ("C_starling_block_search_bnf", G0, D0, P_bnf, True),
    ("D_control_extra_pass_bnf", Gc, Dc, P_c, True),
    ("D2_control_extra_pass_fixed_layout", Gc, Dc, P_bnf, True),
    ("E1_v1_favour_colocated_fixed_layout", G1, D1, P_bnf, True),
    ("E2_v1_favour_colocated_reshuffled", G1, D1, P_1r, True),
    ("F_v2_implicit_blockmates_fixed_layout", G2, D2, P_bnf, True),
]
rows_csv = ["config,L,recall_at_10,mean_block_reads,relevant_records_per_read"]
for name, g, d, layout, block in configs:
    t = time.time()
    _, bs = block_members(layout)
    rows = sweep(X, Q, gt, g, d, start, layout, Ls, block=block, sigma=SIGMA)
    res = {
        "overlap_ratio": round(float(overlap_ratio(layout, g, d, bs)), 4),
        "avg_out_degree": round(float(d.mean()), 2),
        "block_search": block,
        "reads_at_recall_0.90": round(float(reads_at_recall(rows, 0.90)), 1),
        "reads_at_recall_0.95": round(float(reads_at_recall(rows, 0.95)), 1),
        "reads_at_recall_0.98": round(float(reads_at_recall(rows, 0.98)), 1),
        "max_recall": round(max(r[1] for r in rows), 4),
        "relevant_per_read_near_0.95": round(float(min(rows, key=lambda r: abs(r[1] - 0.95))[3]), 3),
    }
    out["configs"][name] = res
    for Lc, r, io, rel in rows:
        rows_csv.append(f"{name},{Lc},{r:.4f},{io:.2f},{rel:.4f}")
    tick("search_" + name, t)
    log(name, res)

out["layout_overlap_ratio_on_G0"] = {
    nm: round(float(overlap_ratio(P, G0, D0, block_members(P)[1])), 4)
    for nm, P in [("id", P_id), ("bnp", P_bnp), ("bnf", P_bnf)]
}
with open(f"results_n{n}_s{seed}.json", "w") as f:
    json.dump(out, f, indent=2)
with open(f"sweeps_n{n}_s{seed}.csv", "w") as f:
    f.write("\n".join(rows_csv) + "\n")
log("done")
