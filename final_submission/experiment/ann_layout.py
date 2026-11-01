"""
Small-scale study of disk layout and pruning for a DiskANN-style graph index.

Everything here is a *simulation of disk I/O*: vectors live in RAM, but every
time the search needs a vertex's adjacency list it "reads" the 4 KB block that
holds that vertex, and we count those reads. Records are sized as in Starling's
BIGANN example (128-byte vector + 4-byte degree + 31 x 4-byte neighbour IDs =
256 bytes), so a 4 KB block holds EPS = 16 vertex records.

Components
  * make_data        synthetic clustered 128-d data (SIFT1M is not downloadable here)
  * vamana_pass      one Vamana pass (GreedySearch + RobustPrune + back edges),
                     batched so it can use both cores
  * robust_prune     DiskANN's rule, plus two layout-aware variants (mode 1, 2)
  * bnp / bnf        Starling's Block Neighbor Padding / Block Neighbor Frequency
  * search_all       DiskANN vertex search or Starling block search, counting reads

Distances are exact (no PQ) for both routing and re-ranking, so the comparison
isolates graph + layout effects. That is a simplification relative to DiskANN
and Starling, which route with PQ distances.
"""
import numpy as np
from numba import njit, prange

EPS = 16          # vertex records per 4 KB block (256-byte records)
R_DEG = 31        # maximum out-degree, as in Starling's BIGANN segment setup


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_data(n, nq, d=128, n_clusters=100, center_scale=1.2, within_scale=1.0,
              spec_decay=1.0, seed=0):
    """Gaussian mixture with anisotropic, power-law-spectrum clusters.

    Each cluster has a random orientation and per-direction variances that
    decay as j^-spec_decay, so the data has a low intrinsic dimension inside a
    128-d ambient space (a rough stand-in for SIFT-like descriptors). Point IDs
    are in random order with respect to geometry, as in real datasets.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, center_scale, size=(n_clusters, d)).astype(np.float32)
    spec = np.arange(1, d + 1, dtype=np.float64) ** (-spec_decay)
    spec = spec / spec.mean()
    sds = np.sqrt(spec) * within_scale
    bases = []
    for _ in range(n_clusters):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        bases.append((q * sds[None, :]).astype(np.float32))
    probs = rng.dirichlet(np.full(n_clusters, 2.0))

    def sample(m):
        lab = rng.choice(n_clusters, size=m, p=probs)
        out = np.empty((m, d), np.float32)
        for c in range(n_clusters):
            idx = np.nonzero(lab == c)[0]
            if idx.size == 0:
                continue
            z = rng.standard_normal((idx.size, d)).astype(np.float32)
            out[idx] = centers[c] + z @ bases[c].T
        return out

    return sample(n), sample(nq)


def ground_truth(X, Q, k=100, chunk=20000):
    """Exact k nearest neighbours of each query (squared L2)."""
    nq = Q.shape[0]
    best_d = np.full((nq, k), np.inf, np.float32)
    best_i = np.full((nq, k), -1, np.int64)
    qn = (Q.astype(np.float64) ** 2).sum(1)[:, None]
    for lo in range(0, X.shape[0], chunk):
        xb = X[lo:lo + chunk].astype(np.float64)
        dist = qn - 2.0 * Q.astype(np.float64) @ xb.T + (xb ** 2).sum(1)[None, :]
        ids = np.arange(lo, lo + xb.shape[0])[None, :].repeat(nq, 0)
        alld = np.concatenate([best_d, dist.astype(np.float32)], 1)
        alli = np.concatenate([best_i, ids], 1)
        sel = np.argpartition(alld, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(alld, sel, 1)
        best_i = np.take_along_axis(alli, sel, 1)
    order = np.argsort(best_d, 1)
    return np.take_along_axis(best_i, order, 1).astype(np.int32)


def medoid(X):
    mu = X.mean(0)
    return int(np.argmin(((X - mu) ** 2).sum(1)))


# ---------------------------------------------------------------------------
# Distance helpers (squared L2)
# ---------------------------------------------------------------------------
@njit(fastmath=True, inline="always")
def d2_vec(X, i, q):
    s = 0.0
    for t in range(X.shape[1]):
        dd = X[i, t] - q[t]
        s += dd * dd
    return s


@njit(fastmath=True, inline="always")
def d2_pair(X, i, j):
    s = 0.0
    for t in range(X.shape[1]):
        dd = X[i, t] - X[j, t]
        s += dd * dd
    return s


# ---------------------------------------------------------------------------
# GreedySearch used during construction (returns the expanded/visited set)
# ---------------------------------------------------------------------------
@njit(fastmath=True)
def greedy_visit(X, nbrs, deg, start, q, L, seen, stamp, vis, vis_cap):
    cid = np.empty(L, np.int32)
    cd = np.empty(L, np.float32)
    cexp = np.zeros(L, np.uint8)
    cnt = 1
    cid[0] = start
    cd[0] = d2_vec(X, start, q)
    seen[start] = stamp
    nvis = 0
    while True:
        pos = -1
        for i in range(cnt):
            if cexp[i] == 0:
                pos = i
                break
        if pos < 0 or nvis >= vis_cap:
            break
        cexp[pos] = 1
        u = cid[pos]
        vis[nvis] = u
        nvis += 1
        for k in range(deg[u]):
            v = nbrs[u, k]
            if seen[v] == stamp:
                continue
            seen[v] = stamp
            dv = d2_vec(X, v, q)
            if cnt == L and dv >= cd[cnt - 1]:
                continue
            if cnt < L:
                i = cnt
                cnt += 1
            else:
                i = L - 1
            while i > 0 and cd[i - 1] > dv:
                cd[i] = cd[i - 1]
                cid[i] = cid[i - 1]
                cexp[i] = cexp[i - 1]
                i -= 1
            cd[i] = dv
            cid[i] = v
            cexp[i] = 0
    return nvis


# ---------------------------------------------------------------------------
# RobustPrune and its layout-aware variants
#   mode 0: DiskANN.  Drop p' if  alpha * d(p*, p') <= d(p, p').
#   mode 1: "favour co-located" (Stage 1 proposal as written). Candidates in the
#           same block as p get alpha * tau, so they are harder to prune away.
#   mode 2: "block-mates as implicit edges". Every block-mate of p is read for
#           free whenever p's block is read, so block-mates are merged into the
#           candidate list as occluders that prune farther candidates but are
#           never stored and never consume out-degree.
# Comparisons use squared distances, so alpha2 = alpha^2 and tau2 = tau^2.
# ---------------------------------------------------------------------------
@njit(fastmath=True)
def robust_prune(X, p, cand, ncand, alpha2, R, mode, tau2, blk, bmem, bsize, out):
    bp = blk[p] if mode > 0 else -1
    extra = bsize[bp] if mode == 2 else 0
    ids = np.empty(ncand + extra, np.int32)
    m = 0
    srt = np.sort(cand[:ncand])
    last = -1
    for i in range(ncand):
        v = srt[i]
        if v == p or v == last:
            continue
        last = v
        ids[m] = v
        m += 1
    imp = np.zeros(m + extra, np.uint8)
    if mode == 2:
        for i in range(m):
            if blk[ids[i]] == bp:
                imp[i] = 1
        for t in range(bsize[bp]):
            b = bmem[bp, t]
            if b == p:
                continue
            dup = False
            for i in range(m):
                if ids[i] == b:
                    dup = True
                    break
            if not dup:
                ids[m] = b
                imp[m] = 1
                m += 1
    dist = np.empty(m, np.float32)
    for i in range(m):
        dist[i] = d2_pair(X, p, ids[i])
    order = np.argsort(dist)
    alive = np.ones(m, np.uint8)
    nsel = 0
    for a in range(m):
        i = order[a]
        if alive[i] == 0:
            continue
        c = ids[i]
        if imp[i] == 0:
            out[nsel] = c
            nsel += 1
            if nsel == R:
                break
        for b in range(a + 1, m):
            j = order[b]
            if alive[j] == 0:
                continue
            a2 = alpha2
            if mode == 1 and blk[ids[j]] == bp:
                a2 = alpha2 * tau2
            if a2 * d2_pair(X, c, ids[j]) <= dist[j]:
                alive[j] = 0
    return nsel


# ---------------------------------------------------------------------------
# One Vamana pass, processed in batches (points in a batch search the same
# snapshot of the graph; back edges are merged per target afterwards).
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _batch_forward(X, nbrs, deg, start, batch, L, alpha2, R, mode, tau2,
                   blk, bmem, bsize, seen, stamps, newl, newd, T):
    B = batch.shape[0]
    chunk = (B + T - 1) // T
    vis_cap = 8 * L
    for t in prange(T):
        vis = np.empty(vis_cap, np.int32)
        cand = np.empty(vis_cap + R, np.int32)
        lo = t * chunk
        hi = min(B, lo + chunk)
        for b in range(lo, hi):
            p = batch[b]
            stamps[t] += 1
            nv = greedy_visit(X, nbrs, deg, start, X[p], L, seen[t], stamps[t], vis, vis_cap)
            m = 0
            for i in range(nv):
                cand[m] = vis[i]
                m += 1
            for k in range(deg[p]):
                cand[m] = nbrs[p, k]
                m += 1
            newd[b] = robust_prune(X, p, cand, m, alpha2, R, mode, tau2, blk, bmem, bsize, newl[b])


@njit(parallel=True, fastmath=True)
def _batch_backward(X, nbrs, deg, js, ps, starts, alpha2, R, mode, tau2, blk, bmem, bsize):
    G = starts.shape[0] - 1
    for g in prange(G):
        lo = starts[g]
        hi = starts[g + 1]
        j = js[lo]
        cand = np.empty(R + (hi - lo), np.int32)
        m = 0
        for k in range(deg[j]):
            cand[m] = nbrs[j, k]
            m += 1
        for x in range(lo, hi):
            p = ps[x]
            if mode == 2 and blk[p] == blk[j]:
                continue
            present = False
            for k in range(m):
                if cand[k] == p:
                    present = True
                    break
            if not present:
                cand[m] = p
                m += 1
        if m <= R:
            for k in range(m):
                nbrs[j, k] = cand[k]
            deg[j] = m
        else:
            out = np.empty(R, np.int32)
            nd = robust_prune(X, j, cand, m, alpha2, R, mode, tau2, blk, bmem, bsize, out)
            for k in range(nd):
                nbrs[j, k] = out[k]
            deg[j] = nd


def vamana_pass(X, nbrs, deg, start, L, alpha, R, rng, mode=0, tau=1.0,
                blk=None, batch_size=2000, T=2):
    n = X.shape[0]
    if blk is None:
        blk = id_layout(n)   # unused when mode == 0, but must be a valid layout
    bmem, bsize = block_members(blk)
    seen = np.zeros((T, n), np.int32)
    stamps = np.zeros(T, np.int64)
    perm = rng.permutation(n).astype(np.int32)
    alpha2, tau2 = float(alpha) ** 2, float(tau) ** 2
    for lo in range(0, n, batch_size):
        batch = perm[lo:lo + batch_size]
        B = batch.shape[0]
        newl = np.full((B, R), -1, np.int32)
        newd = np.zeros(B, np.int32)
        _batch_forward(X, nbrs, deg, start, batch, L, alpha2, R, mode, tau2,
                       blk, bmem, bsize, seen, stamps, newl, newd, T)
        nbrs[batch] = newl
        deg[batch] = newd
        # back edges (j <- p for every new edge p -> j)
        js = np.concatenate([newl[b, :newd[b]] for b in range(B)])
        ps = np.repeat(batch, newd)
        order = np.argsort(js, kind="stable")
        js, ps = js[order], ps[order]
        if js.size == 0:
            continue
        brk = np.nonzero(np.diff(js))[0] + 1
        starts = np.concatenate([[0], brk, [js.size]]).astype(np.int64)
        _batch_backward(X, nbrs, deg, js.astype(np.int32), ps.astype(np.int32), starts,
                        alpha2, R, mode, tau2, blk, bmem, bsize)


def build_vamana(X, R=R_DEG, L=75, alpha=1.2, seed=0, batch_size=2000, T=2):
    """DiskANN's two passes: alpha = 1 first, then the user alpha."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    nbrs = rng.integers(0, n, size=(n, R)).astype(np.int32)
    deg = np.full(n, R, np.int32)
    start = medoid(X)
    vamana_pass(X, nbrs, deg, start, L, 1.0, R, rng, batch_size=batch_size, T=T)
    vamana_pass(X, nbrs, deg, start, L, alpha, R, rng, batch_size=batch_size, T=T)
    return nbrs, deg, start


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def id_layout(n, eps=EPS):
    return (np.arange(n) // eps).astype(np.int32)


@njit
def _block_members(blk, nb, eps):
    bmem = np.full((nb, eps), -1, np.int32)
    bsize = np.zeros(nb, np.int32)
    for u in range(blk.shape[0]):
        b = blk[u]
        bmem[b, bsize[b]] = u
        bsize[b] += 1
    return bmem, bsize


def block_members(blk, eps=EPS):
    nb = int(blk.max()) + 1
    return _block_members(blk, nb, eps)


@njit
def overlap_ratio(blk, nbrs, deg, bsize):
    """Starling's OR(G): mean over u of |B(u) ∩ N(u)| / (|B(u)| - 1)."""
    n = blk.shape[0]
    tot = 0.0
    for u in range(n):
        s = bsize[blk[u]]
        if s <= 1:
            continue
        c = 0
        for k in range(deg[u]):
            v = nbrs[u, k]
            if v != u and blk[v] == blk[u]:
                c += 1
        tot += c / (s - 1)
    return tot / n


@njit
def bnp(nbrs, deg, n, eps):
    """Starling Algorithm I: Block Neighbor Padding."""
    blk = np.full(n, -1, np.int32)
    cur = 0
    fill = 0
    for u in range(n):
        if blk[u] >= 0:
            continue
        blk[u] = cur
        fill += 1
        if fill == eps:
            cur += 1
            fill = 0
        for k in range(deg[u]):
            v = nbrs[u, k]
            if blk[v] < 0:
                blk[v] = cur
                fill += 1
                if fill == eps:
                    cur += 1
                    fill = 0
    return blk


@njit
def _bnf_iter(nbrs, deg, n, eps, D, nb):
    size = np.zeros(nb, np.int32)
    newb = np.full(n, -1, np.int32)
    hb = np.empty(64, np.int32)
    hc = np.empty(64, np.int32)
    ptr = 0
    for u in range(n):
        h = 0
        for k in range(deg[u]):
            x = D[nbrs[u, k]]
            found = False
            for t in range(h):
                if hb[t] == x:
                    hc[t] += 1
                    found = True
                    break
            if not found:
                hb[h] = x
                hc[h] = 1
                h += 1
        assigned = False
        while h > 0:
            best = 0
            for t in range(1, h):
                if hc[t] > hc[best]:
                    best = t
            x = hb[best]
            if size[x] < eps:
                newb[u] = x
                size[x] += 1
                assigned = True
                break
            hb[best] = hb[h - 1]
            hc[best] = hc[h - 1]
            h -= 1
        if not assigned:
            while size[ptr] >= eps:
                ptr += 1
            newb[u] = ptr
            size[ptr] += 1
    return newb


def bnf(nbrs, deg, init_blk, eps=EPS, iters=8):
    """Starling Algorithm 1: Block Neighbor Frequency, beta = iters iterations.

    On our graphs the first iteration from a BNP layout lowers OR(G) before later
    iterations raise it past BNP, so instead of Starling's gain-threshold stop
    we run all beta = 8 iterations (Starling's default) and keep the best layout.
    """
    n = init_blk.shape[0]
    nb = (n + eps - 1) // eps
    best = init_blk.copy()
    _, bs = block_members(best, eps)
    best_or = overlap_ratio(best, nbrs, deg, bs)
    cur = best
    for _ in range(iters):
        cur = _bnf_iter(nbrs, deg, n, eps, cur, nb)
        _, bs = block_members(cur, eps)
        cur_or = overlap_ratio(cur, nbrs, deg, bs)
        if cur_or > best_or:
            best, best_or = cur, cur_or
    return best


# ---------------------------------------------------------------------------
# Search with read counting
#   block=False: DiskANN-style vertex search (one block read per expanded vertex)
#   block=True : Starling-style block search without the I/O pipeline: after
#                reading u's block, the top round((eps-1)*sigma) block-mates by
#                distance to q are also expanded with no further read.
# ---------------------------------------------------------------------------
@njit(fastmath=True)
def _search_one(X, q, nbrs, deg, start, blk, bmem, bsize, Lc, k, block, nprune,
                seen, done, isgt, stamp, res, resd):
    cid = np.empty(Lc, np.int32)
    cd = np.empty(Lc, np.float32)
    cnt = 1
    cid[0] = start
    cd[0] = d2_vec(X, start, q)
    seen[start] = stamp
    nres = 0
    ios = 0
    rel = 0
    mdist = np.empty(64, np.float32)
    mid = np.empty(64, np.int32)
    while True:
        pos = -1
        for i in range(cnt):
            if done[cid[i]] != stamp:
                pos = i
                break
        if pos < 0:
            break
        u = cid[pos]
        done[u] = stamp
        ios += 1
        b = blk[u]
        for t in range(bsize[b]):
            if isgt[bmem[b, t]] == stamp:
                rel += 1
        # expand u (full vector + neighbour list come with the block)
        if nres < res.shape[0]:
            res[nres] = u
            resd[nres] = cd[pos]
            nres += 1
        nexp = 1
        mid[0] = u
        if block:
            nm = 0
            for t in range(bsize[b]):
                v = bmem[b, t]
                if v == u:
                    continue
                mdist[nm] = d2_vec(X, v, q)
                mid[nm + 1] = v
                nm += 1
            # pick nprune closest block-mates (selection sort on a tiny array)
            take = min(nprune, nm)
            for a in range(take):
                bi = a
                for c in range(a + 1, nm):
                    if mdist[c] < mdist[bi]:
                        bi = c
                tmpd = mdist[a]; mdist[a] = mdist[bi]; mdist[bi] = tmpd
                tmpi = mid[a + 1]; mid[a + 1] = mid[bi + 1]; mid[bi + 1] = tmpi
                v = mid[a + 1]
                if done[v] != stamp:
                    done[v] = stamp
                    if nres < res.shape[0]:
                        res[nres] = v
                        resd[nres] = mdist[a]
                        nres += 1
                    mid[nexp] = v
                    nexp += 1
        for e in range(nexp):
            w = mid[e]
            for kk in range(deg[w]):
                v = nbrs[w, kk]
                if seen[v] == stamp or done[v] == stamp:
                    continue
                seen[v] = stamp
                dv = d2_vec(X, v, q)
                if cnt == Lc and dv >= cd[cnt - 1]:
                    continue
                if cnt < Lc:
                    i = cnt
                    cnt += 1
                else:
                    i = Lc - 1
                while i > 0 and cd[i - 1] > dv:
                    cd[i] = cd[i - 1]
                    cid[i] = cid[i - 1]
                    i -= 1
                cd[i] = dv
                cid[i] = v
    # top-k of evaluated vertices by exact distance
    order = np.argsort(resd[:nres])
    kk = min(k, nres)
    out = np.empty(kk, np.int32)
    for i in range(kk):
        out[i] = res[order[i]]
    return out, ios, rel


@njit(parallel=True, fastmath=True)
def _search_batch(X, Q, gt, nbrs, deg, start, blk, bmem, bsize, Lc, k, block, nprune,
                  n_rel_gt, T, recall_out, ios_out, rel_out):
    nq = Q.shape[0]
    n = X.shape[0]
    chunk = (nq + T - 1) // T
    for t in prange(T):
        seen = np.zeros(n, np.int32)
        done = np.zeros(n, np.int32)
        isgt = np.zeros(n, np.int32)
        res = np.empty(Lc * 64, np.int32)
        resd = np.empty(Lc * 64, np.float32)
        lo = t * chunk
        hi = min(nq, lo + chunk)
        stamp = 0
        for qi in range(lo, hi):
            stamp += 1
            for j in range(n_rel_gt):
                isgt[gt[qi, j]] = stamp
            out, ios, rel = _search_one(X, Q[qi], nbrs, deg, start, blk, bmem, bsize, Lc, k,
                                        block, nprune, seen, done, isgt, stamp, res, resd)
            hit = 0
            for a in range(out.shape[0]):
                for b in range(k):
                    if out[a] == gt[qi, b]:
                        hit += 1
                        break
            recall_out[qi] = hit / k
            ios_out[qi] = ios
            rel_out[qi] = rel


def search_all(X, Q, gt, nbrs, deg, start, blk, Lc, k=10, block=False, sigma=0.3,
               eps=EPS, n_rel_gt=100, T=2):
    bmem, bsize = block_members(blk, eps)
    nprune = int(np.floor((eps - 1) * sigma + 0.5))
    nq = Q.shape[0]
    rec = np.zeros(nq)
    ios = np.zeros(nq, np.int64)
    rel = np.zeros(nq, np.int64)
    _search_batch(X, Q, gt, nbrs, deg, start, blk, bmem, bsize, Lc, k, block, nprune,
                  n_rel_gt, T, rec, ios, rel)
    return rec.mean(), ios.mean(), rel.sum() / max(ios.sum(), 1)


def sweep(X, Q, gt, nbrs, deg, start, blk, Ls, **kw):
    rows = []
    for Lc in Ls:
        r, io, rel = search_all(X, Q, gt, nbrs, deg, start, blk, Lc, **kw)
        rows.append((Lc, r, io, rel))
    return rows


def reads_at_recall(rows, target):
    """Linear interpolation of mean block reads at a target mean recall@k."""
    rows = sorted(rows, key=lambda r: r[1])
    for (l0, r0, i0, _), (l1, r1, i1, _) in zip(rows, rows[1:]):
        if r0 <= target <= r1:
            if r1 == r0:
                return i0
            return i0 + (target - r0) * (i1 - i0) / (r1 - r0)
    return float("nan")
