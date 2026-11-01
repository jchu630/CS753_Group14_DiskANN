"""Summarise results_*.json into the numbers quoted in the report."""
import glob
import json

import numpy as np

NAMES = {
    "A_diskann_vertex_search_id_layout": "A. DiskANN: vertex search, ID layout",
    "B_block_search_id_layout": "B. Block search, ID layout",
    "C_starling_block_search_bnf": "C. Starling: block search, BNF",
    "D_control_extra_pass_bnf": "D. Control: + 1 standard pass, BNF",
    "D2_control_extra_pass_fixed_layout": "D'. Control pass, C's layout fixed",
    "E1_v1_favour_colocated_fixed_layout": "E. V1 favour co-located",
    "E2_v1_favour_colocated_reshuffled": "E'. V1, re-shuffled",
    "F_v2_implicit_blockmates_fixed_layout": "F. V2 implicit block-mates",
}
KEYS = ["reads_at_recall_0.90", "reads_at_recall_0.95", "reads_at_recall_0.98"]

runs = {}
for f in sorted(glob.glob("results_n*_s*.json")):
    d = json.load(open(f))
    runs[(d["n"], d["seed"])] = d

for (n, s), d in sorted(runs.items()):
    print(f"\n=== n={n} seed={s}  OR on G0: {d['layout_overlap_ratio_on_G0']}")
    for k, v in d["configs"].items():
        print(f"  {NAMES[k]:40s} OR={v['overlap_ratio']:.3f} deg={v['avg_out_degree']:.2f} "
              + " ".join(f"{v[x]:7.1f}" for x in KEYS)
              + f"  rel/read@~0.95={v['relevant_per_read_near_0.95']:.2f} maxR={v['max_recall']}")

# Paired ratios across 100K seeds
small = [d for (n, s), d in runs.items() if n == 100_000]
if small:
    print(f"\n=== n=100000, {len(small)} seeds: ratio of reads (mean, min, max)")
    pairs = [("A_diskann_vertex_search_id_layout", "C_starling_block_search_bnf"),
             ("A_diskann_vertex_search_id_layout", "B_block_search_id_layout"),
             ("C_starling_block_search_bnf", "D_control_extra_pass_bnf"),
             ("D_control_extra_pass_bnf", "E1_v1_favour_colocated_fixed_layout"),
             ("D_control_extra_pass_bnf", "F_v2_implicit_blockmates_fixed_layout"),
             ("D2_control_extra_pass_fixed_layout", "F_v2_implicit_blockmates_fixed_layout"),
             ("D2_control_extra_pass_fixed_layout", "E1_v1_favour_colocated_fixed_layout"),
             ("C_starling_block_search_bnf", "F_v2_implicit_blockmates_fixed_layout")]
    for a, b in pairs:
        for key in KEYS:
            if not all(a in d["configs"] and b in d["configs"] for d in small):
                continue
            r = np.array([d["configs"][a][key] / d["configs"][b][key] for d in small])
            print(f"  {a[:12]:12s}/{b[:14]:14s} {key[-4:]}: {r.mean():.3f} [{r.min():.3f}, {r.max():.3f}]")
    for key in KEYS:
        vals = {k: np.array([d["configs"][k][key] for d in small]) for k in NAMES
                if all(k in d["configs"] for d in small)}
        print("  " + key + ": " + ", ".join(f"{k[:2]} {v.mean():.1f}±{v.std(ddof=1):.1f}" for k, v in vals.items()))
