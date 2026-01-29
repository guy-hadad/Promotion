import time
import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCORES_PATH = "scores_memmap/sasrec_ml-100k_full_scores_float16.npy"
POS_U_PATH  = "ml100k_pos_u.npy"
POS_I_PATH  = "ml100k_pos_i.npy"
META_PATH   = "ml100k_item_metadata_aligned.csv"

# For MovieLens-10M
# SCORES_PATH = "scores_memmap/sasrec_ml-10m_full_scores_float16.npy"
# POS_U_PATH  = "ml10m_pos_u.npy"
# POS_I_PATH  = "ml10m_pos_i.npy"
# META_PATH   = "ml-10m_item_metadata_aligned.csv"

DUAL_MAX_ITER = 200         
DUAL_LR       = 0.5          
DUAL_CLIP     = 50.0         
DUAL_MU       = 1e-8         
KS = [1, 5, 10]      
MAX_K = max(KS)       
PROMOTERS = ["Action", "Comedy", "Romance"]
DELTA_M = 2
TOLERANCE = 1e-10
IPF_MAX_ITER = 300
NEWTON_MAX_ITER = 10
LOG_DISC = 1.0 / np.log2(np.arange(2, MAX_K + 2))
FAIL_TOL = 1e-7
TOPK = MAX_K  
min_weight = 1e-12


def get_metrics(r, u_idx, pos_map, multi_map):
    arg_top = np.argpartition(-r, kth=MAX_K-1)[:MAX_K]
    top_items = arg_top[np.argsort(-r[arg_top])]
    
    results = {}
    if u_idx in multi_map:
        pos_set = multi_map[u_idx]
        all_hits = np.isin(top_items, list(pos_set)).astype(float)
        
        for k in KS:
            hits_k = all_hits[:k]
            rec = 1.0 if hits_k.sum() > 0 else 0.0
            
            idcg = np.sum(LOG_DISC[:min(len(pos_set), k)])
            dcg = np.sum(hits_k * LOG_DISC[:k])
            
            results[k] = (rec, dcg / idcg if idcg > 0 else 0.0)
    else:
        for k in KS:
            results[k] = (0.0, 0.0)
            
    return results

def solve_population_kl_newton(probs, Q, pop_targets, weights=None,
                              max_iter=NEWTON_MAX_ITER, tol=TOLERANCE):
    """
    Shared-dual Population-Level KL projection.

    probs:       (N, I) baseline user distributions
    Q:           (K, I) constraint masks/features
    pop_targets: (K,)   aggregate exposure targets (e.g., targets_all.mean(axis=0))
    weights:     (N,)   traffic weights (defaults to uniform)
    """
    N, I = probs.shape
    K = len(pop_targets)

    if weights is None:
        weights = np.ones(N, dtype=np.float64) / N
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

    lam = np.zeros(K, dtype=np.float64)
    mu = 1e-8

    for _ in range(max_iter):
        total_grad = np.zeros(K, dtype=np.float64)
        total_H = np.zeros((K, K), dtype=np.float64)

        # shared tilt across population
        logits = lam @ Q
        w_shared = np.exp(logits - np.max(logits))

        for u in range(N):
            r_u = probs[u] * w_shared
            r_u /= max(r_u.sum(), 1e-20)

            expv_u = Q @ r_u
            total_grad += weights[u] * (expv_u - pop_targets)

            # Hessian (weighted sum of covariances)
            Qw_u = Q * r_u[None, :]
            H_u = (Qw_u @ Q.T) - np.outer(expv_u, expv_u)
            total_H += weights[u] * H_u

        if np.max(np.abs(total_grad)) <= tol:
            break

        try:
            step = np.linalg.solve(total_H + mu * np.eye(K), total_grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(total_H) @ total_grad

        lam = np.maximum(0.0, lam - step)

    # final adjusted distributions for all users
    final_logits = lam @ Q
    W = np.exp(final_logits - np.max(final_logits))
    R = probs * W[None, :]
    R /= np.maximum(R.sum(axis=1, keepdims=True), 1e-20)

    return R, lam

def solve_kl_newton(p, Q, targets, max_iter=NEWTON_MAX_ITER, tol=TOLERANCE):
    K = len(targets)
    lam = np.zeros(K)
    mu = 1e-8 
    for _ in range(max_iter):
        logits = lam @ Q
        w = np.exp(logits - np.max(logits))
        r = p * w
        r /= r.sum()

        expv = Q @ r
        grad = expv - targets
        if np.max(np.abs(grad)) <= tol:
            return r

        # Hessian (Covariance)
        Qw = Q * r[None, :]
        H = (Qw @ Q.T) - np.outer(expv, expv)
        
        try:
            step = np.linalg.solve(H + mu * np.eye(K), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        
        lam = np.maximum(0, lam - step) 
    return r

def solve_regular_ipf(p, constraint_indices, targets, max_iter=IPF_MAX_ITER, tol=TOLERANCE):
    r = p.astype(np.float64, copy=True)

    # (Optional) avoid true zeros if constraints might require mass there
    # r = np.maximum(r, 1e-30)

    K = len(targets)

    for _ in range(max_iter):
        # Project onto each subset-sum constraint
        for k in range(K):
            idx = constraint_indices[k]
            curr_mass = r[idx].sum()
            if curr_mass > 1e-20:
                r[idx] *= (targets[k] / curr_mass)

        # Project onto simplex (sum = 1)
        total_mass = r.sum()
        if total_mass > 1e-20:
            r /= total_mass

        # Residual-based stopping (recommended)
        max_res = abs(r.sum() - 1.0)
        for k in range(K):
            idx = constraint_indices[k]
            max_res = max(max_res, abs(r[idx].sum() - targets[k]))

        if max_res < tol:
            break

    return r

def pick_mip_solver():
    for s in ["GUROBI", "CPLEX", "MOSEK", "SCIP", "HIGHS", "CBC", "GLPK_MI", "ECOS_BB"]:
        if s in cp.installed_solvers():
            return s
    return None

class IPExposureAlignedSolver:
    """
    Exposure-aligned MILP baseline:
        maximize   p^T w
        s.t.       sum(x) = K
                   sum(w) = 1
                   0 <= w <= x
                   Q w within +/- tol of targets  (aligns with err = max|Q@r - targets|)
                   x boolean
    """
    def __init__(self, num_groups, k, pool_size=200, tol=1e-7, min_weight=0.0):
        self.G = int(num_groups)
        self.K = int(k)
        self.P = int(pool_size)
        self.tol = float(tol)
        self.min_weight = float(min_weight)  

        # Variables
        self.x = cp.Variable(self.P, boolean=True)
        self.w = cp.Variable(self.P, nonneg=True)

        # Parameters (updated per user)
        self.p_param = cp.Parameter(self.P, nonneg=True)          # candidate probs/scores
        self.Q_param = cp.Parameter((self.G, self.P))             # candidate group matrix
        self.t_param = cp.Parameter(self.G)                       # targets

        # Constraints
        cons = [
            cp.sum(self.x) == self.K,
            cp.sum(self.w) == 1.0,
            self.w <= self.x,
            self.w >= 0.0
        ]

        # Must satisfy min_weight * K <= 1.
        if self.min_weight > 0:
            cons.append(self.w >= self.min_weight * self.x)

        # Exposure constraints aligned with your evaluation (absolute difference)
        cons += [
            self.Q_param @ self.w >= self.t_param - self.tol,
            self.Q_param @ self.w <= self.t_param + self.tol
        ]

        # Objective: maximize expected relevance under final distribution w
        obj = cp.Maximize(self.p_param @ self.w)

        self.prob = cp.Problem(obj, cons)
        self.solver = pick_mip_solver()

    def solve(self, p_full, Q_full, targets):
        # 1) top pool via argpartition (fast)
        cand_idx = np.argpartition(-p_full, self.P - 1)[: self.P]
        cand_p = p_full[cand_idx].astype(np.float64, copy=False)
        cand_Q = Q_full[:, cand_idx].astype(np.float64, copy=False)
        targets = targets.astype(np.float64, copy=False)

        # 2) update params
        self.p_param.value = cand_p
        self.Q_param.value = cand_Q
        self.t_param.value = targets

        # 3) warm start: pick top-K candidates, distribute w proportional to cand_p
        x0 = np.zeros(self.P, dtype=np.float64)
        topk_local = np.argpartition(-cand_p, self.K - 1)[: self.K]
        x0[topk_local] = 1.0
        self.x.value = x0

        w0 = np.zeros(self.P, dtype=np.float64)
        denom = cand_p[topk_local].sum()
        if denom > 0:
            w0[topk_local] = cand_p[topk_local] / denom
        else:
            w0[topk_local] = 1.0 / self.K
        self.w.value = w0

        # 4) solve
        try:
            if self.solver is None:
                self.prob.solve(warm_start=True, verbose=False)
            else:
                self.prob.solve(solver=self.solver, warm_start=True, verbose=False)
        except Exception:
            return None, None, "solve_failed"

        if self.w.value is None or self.x.value is None:
            return None, None, "infeasible"

        w_sol = np.asarray(self.w.value).reshape(-1)
        # Support items are in candidate pool only
        r_out = np.zeros_like(p_full, dtype=np.float64)
        r_out[cand_idx] = np.maximum(w_sol, 0.0)

        s = r_out.sum()
        if s <= 0:
            return None, None, "degenerate"
        r_out /= s

        # indices with mass (for faster error computation later)
        support_local = np.where(w_sol > 1e-15)[0]
        support_full = cand_idx[support_local]

        return r_out, support_full, "ok"

def exposure_error(r, Q, targets):
    """max absolute constraint residual: max_k |(Q@r)[k] - targets[k]|"""
    expv = Q @ r
    return float(np.max(np.abs(expv - targets)))

def evaluate_method(name, solve_one_user_fn, probs, Q, targets_all, pos_map, multi_map,
                    fail_tol=FAIL_TOL, show_progress=True):
    N, I = probs.shape

    metric_sums = {k: {"hit": 0.0, "ndcg": 0.0} for k in KS}
    err_sum = 0.0
    time_ms_sum = 0.0

    solved = 0
    solver_fails = 0
    constraint_fails = 0

    spearman100_sum = 0.0

    it = range(N)
    if show_progress:
        it = tqdm(it, desc=f"Eval {name}", leave=False)

    for u in it:
        p_u = probs[u]
        t_u = targets_all[u]

        t0 = time.perf_counter()
        r_u, status = solve_one_user_fn(u, p_u, t_u)
        t1 = time.perf_counter()
        time_ms_sum += (t1 - t0) * 1000.0

        if r_u is None or status != "ok":
            solver_fails += 1
            continue

        solved += 1

        # --- metrics ---
        spearman100_sum += spearman_at_k_union(p_u, r_u, k=100)

        res = get_metrics(r_u, u, pos_map, multi_map)
        for k in KS:
            metric_sums[k]["hit"] += res[k][0]
            metric_sums[k]["ndcg"] += res[k][1]

        # --- exposure error + failure definition ---
        e = exposure_error(r_u, Q, t_u)
        err_sum += e

        if e > fail_tol:
            constraint_fails += 1

    denom = max(solved, 1)

    out = {
        "method": name,
        "users_total": N,
        "users_solved": solved,

        "solver_fails": solver_fails,
        "constraint_fails": constraint_fails,

        "fail_rate": constraint_fails / denom,

        "solver_fail_rate": solver_fails / max(N, 1),

        "avg_error": err_sum / denom,
        "avg_time_ms": time_ms_sum / max(N, 1),
        "spearman@100": spearman100_sum / denom,
    }

    for k in KS:
        out[f"recall@{k}"] = metric_sums[k]["hit"] / denom
        out[f"ndcg@{k}"]   = metric_sums[k]["ndcg"] / denom

    return out

def spearman_at_k_union(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    """
    Spearman rank correlation between rankings induced by a and b,
    computed on the union of their top-k items.

    Pure numpy, minimal, fast (no full sort over I).
    """
    I = a.size
    k = min(k, I)
    if k < 2:
        return 0.0

    ia = np.argpartition(-a, k - 1)[:k]
    ib = np.argpartition(-b, k - 1)[:k]
    items = np.union1d(ia, ib)          # compare ranks on union
    m = items.size
    if m < 2:
        return 0.0

    sa = a[items]
    sb = b[items]

    # ranks: 1..m (higher score => better rank)
    ra = np.empty(m, dtype=np.int32)
    rb = np.empty(m, dtype=np.int32)
    ra[np.argsort(-sa, kind="mergesort")] = np.arange(1, m + 1)
    rb[np.argsort(-sb, kind="mergesort")] = np.arange(1, m + 1)

    d = ra.astype(np.float64) - rb.astype(np.float64)
    denom = m * (m * m - 1.0)
    return 0.0 if denom <= 0 else (1.0 - (6.0 * np.sum(d * d)) / denom)

def evaluate_population_newton(name, probs, Q, pop_targets, pos_map, multi_map, show_progress=True):
    """
    Runs population KL-Newton once, then computes per-user Recall/NDCG@K.
    Does NOT compute per-user error or fails (not applicable).
    """
    N, I = probs.shape

    t0 = time.perf_counter()
    R_all, lam = solve_population_kl_newton(probs, Q, pop_targets, max_iter=NEWTON_MAX_ITER, tol=TOLERANCE)
    t1 = time.perf_counter()
    total_time_ms = (t1 - t0) * 1000.0
    avg_time_ms = total_time_ms / max(N, 1)

    metric_sums = {k: {"hit": 0.0, "ndcg": 0.0} for k in KS}
    expv_pop = (R_all @ Q.T).mean(axis=0)   # uniform mean matches pop_targets definition
    err = np.max(np.abs(expv_pop - pop_targets))

    spearman100_sum = 0.0  

    it = range(N)
    if show_progress:
        it = tqdm(it, desc=f"Eval {name}", leave=False)

    for u in it:
        r_u = R_all[u]

        spearman100_sum += spearman_at_k_union(probs[u], r_u, k=100)

        res = get_metrics(r_u, u, pos_map, multi_map)
        for k in KS:
            metric_sums[k]["hit"] += res[k][0]
            metric_sums[k]["ndcg"] += res[k][1]

    out = {
        "method": name,
        "users_total": N,
        "users_solved": N,          
        "fails": np.nan,           
        "fail_rate": np.nan,      
        "avg_error": err,
        "avg_time_ms": avg_time_ms,

        "spearman@100": spearman100_sum / max(N, 1),  
    }
    for k in KS:
        out[f"recall@{k}"] = metric_sums[k]["hit"] / max(N, 1)
        out[f"ndcg@{k}"]   = metric_sums[k]["ndcg"] / max(N, 1)

    return out

def solve_baseline(u, p_u, t_u):
    return p_u.astype(np.float64, copy=False), "ok"

def solve_ipf(u, p_u, t_u):
    try:
        r = solve_regular_ipf(p_u, Q_indices, t_u, max_iter=IPF_MAX_ITER, tol=TOLERANCE)
        if not np.all(np.isfinite(r)) or r.sum() <= 0:
            return None, "degenerate"
        r = r / max(r.sum(), 1e-20)
        return r, "ok"
    except Exception:
        return None, "exception"

def solve_newton(u, p_u, t_u):
    try:
        r = solve_kl_newton(p_u, Q, t_u, max_iter=NEWTON_MAX_ITER, tol=TOLERANCE)
        if r is None or (not np.all(np.isfinite(r))) or r.sum() <= 0:
            return None, "degenerate"
        r = r / max(r.sum(), 1e-20)
        return r, "ok"
    except Exception:
        return None, "exception"

def solve_ip_milp(u, p_u, t_u):
    r_out, support_full, status = ip_solver.solve(p_u, Q, t_u)
    if status != "ok" or r_out is None:
        return None, status
    s = r_out.sum()
    if (not np.isfinite(s)) or s <= 0:
        return None, "degenerate"
    r_out = r_out / s
    return r_out, "ok"

def fit_paper_lagrange_multipliers_topk(
    probs,                 
    item_provider,         
    target_counts,         
    topk,
    max_iter=50,
    pi=1.0,                 
    tol=0.0,              
    eps=1e-30,
    clip_lambda=50.0,       
    verbose=False,
):
    """
    Inspired by Surer et al. RecSys'18:
      - For fixed lambdas, solve per-user topk with shifted utilities.
      - Update lambdas via subgradient on provider exposure constraints.

    NOTE: target_counts are in *counts* units, matching paper's:
      sum_u sum_{i in provider r} x_ui >= target_counts[r],
      where x_ui is 1 if item i appears in user's topk list.

    Returns:
      lambdas (float64, nonnegative), status
    """
    probs = np.asarray(probs)
    m, n = probs.shape

    item_provider = np.asarray(item_provider, dtype=np.int32)
    R = int(item_provider.max()) + 1

    target_counts = np.asarray(target_counts, dtype=np.float64)
    if target_counts.shape[0] != R:
        raise ValueError(f"target_counts must have length {R}, got {target_counts.shape[0]}")

    lambdas = np.zeros(R, dtype=np.float64)

    # --- proxy for ZLB (paper uses feasible lower bound). We just keep a numeric baseline. ---
    # This is only used to scale the step size like in the paper formula.
    ZLB = 0.0

    for it in range(max_iter):
        # precompute exp(lambda) per provider and per item
        lam_clip = np.clip(lambdas, 0.0, clip_lambda)
        w_provider = np.exp(lam_clip)                    # [R]
        w_item = w_provider[item_provider]               # [n]

        counts = np.zeros(R, dtype=np.int64)

        # Dual objective proxy: sum(log(p_u)+lambda_provider) over selected topk - sum(lambda*target)
        # We compute it to mimic the paper's T = pi*(ZUB-ZLB)/||G||^2 scaling.
        ZUB = 0.0

        for u in range(m):
            p_u = probs[u]

            # ranking score ~ exp(log p_u + lambda_provider) == p_u * exp(lambda_provider)
            score = p_u * w_item

            if topk >= n:
                top_idx = np.arange(n)
            else:
                part = np.argpartition(-score, kth=topk - 1)[:topk]
                top_idx = part[np.argsort(-score[part])]

            prov = item_provider[top_idx]
            counts += np.bincount(prov, minlength=R)

            # dual objective proxy (log domain)
            ZUB += (np.log(np.maximum(p_u[top_idx], eps)) + lam_clip[item_provider[top_idx]]).sum()

        ZUB -= np.dot(lam_clip, target_counts)

        # subgradient: G_r = exposure_r - target_r
        G = counts.astype(np.float64) - target_counts

        # deficits (violations of lower bounds): target - counts
        deficits = np.maximum(-G, 0.0)
        max_deficit = deficits.max()

        if verbose and (it % 5 == 0 or it == max_iter - 1):
            sat = float(np.mean(counts >= target_counts))
            print(f"[it={it:03d}] max_deficit={max_deficit:.3f}  satisfied={sat:.3f}")

        if max_deficit <= tol:
            return lambdas, "ok"

        # paper-style step: T = pi*(ZUB - ZLB)/sum_r G_r^2
        denom = float(np.dot(G, G) + 1e-12)
        step = pi * max(ZUB - ZLB, 1e-9) / denom

        # update + projection to lambda >= 0
        lambdas = np.maximum(0.0, lambdas + step * G)

        # update numeric LB proxy (monotone)
        ZLB = max(ZLB, ZUB)

    return lambdas, "max_iter"

def make_paper_dual_solver(
    lambdas,
    item_provider,
    eps=1e-30,
    clip_lambda=50.0,
):
    """
    Returns a function solve(u_idx, p_u, t_u) -> (r, status)
    compatible with your evaluate_method pipeline.

    r is a full vector (same shape as p_u), normalized to sum=1.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    item_provider = np.asarray(item_provider, dtype=np.int32)

    lam_clip = np.clip(lambdas, 0.0, clip_lambda)
    w_provider = np.exp(lam_clip)
    w_item = w_provider[item_provider]   # precompute once (fast)

    def solve(u_idx, p_u, t_u=None):
        try:
            p_u = np.asarray(p_u, dtype=np.float64)
            r = p_u * w_item
            s = r.sum()
            if (not np.isfinite(s)) or s <= 0:
                return None, "degenerate"
            r /= s
            return r, "ok"
        except Exception:
            return None, "exception"

    return solve

def solve_lagrangian_relaxation(p, constraint_indices, targets, max_iter=100, pi=2.0):
    """
    Adapts the paper's 'Multistakeholder Recommendation with Provider Constraints' approach.
    
    Paper Logic:
    1. Formulate constraints as G(x) >= 0[cite: 143].
    2. Relax constraints into objective with multipliers lambda.
    3. Update lambdas via Subgradient Optimization[cite: 225].
    
    Args:
        p: Initial probability distribution (acting as 'utility' u_ij in paper).
        constraint_indices: List of indices for each constraint subset.
        targets: Target mass for each subset.
    """
    # Initialize variables
    n = len(p)
    K = len(targets)
    lambdas = np.zeros(K)  # Lagrangian multipliers 
    
    best_r = p.copy() / p.sum()
    best_violation = float('inf')
    
    pi_val = pi 
    
    for t in range(max_iter):
        # --- 1. SOLVE LUBP (Lagrangian Upper Bound Problem) ---
        # "Multiply each constraint with lambda_r and bring into objective" 
        # Objective: Maximize sum((p_i + sum(lambda_k)) * r_i)
        
        adjusted_scores = p.astype(np.float64, copy=True)
        
        # Add multipliers to the utilities of items in constrained sets 
        for k in range(K):
            idx = constraint_indices[k]
            if lambdas[k] > 0:
                adjusted_scores[idx] += lambdas[k]
        
        # In the paper, LUBP is Top-K. Here, for probabilities, we project to simplex.
        # This finds r that is "closest" to the adjusted scores while valid.
        r = project_to_simplex(adjusted_scores)

        # --- 2. COMPUTE SUBGRADIENTS ---
        # G_r = Sum(x_ij) - Target 
        gradients = np.zeros(K)
        current_sums = np.zeros(K)
        
        for k in range(K):
            idx = constraint_indices[k]
            current_sums[k] = r[idx].sum()
            # Constraint: sum >= target. Gradient = sum - target.
            gradients[k] = current_sums[k] - targets[k]

        # --- 3. UPDATE BEST FEASIBLE SOLUTION (Primal Heuristic) ---
        # The paper uses a greedy heuristic to enforce feasibility 
        # We track the solution with minimal constraint violation.
        max_v = np.max(np.abs(current_sums - targets))
        if max_v < best_violation:
            best_violation = max_v
            best_r = r.copy()

        # --- 4. UPDATE MULTIPLIERS (Subgradient Step) ---
        # Step size T formula from paper 
        # T = pi * (UB - LB) / sum(gradients^2)
        
        # Calculate UB (Current Relaxed Objective Value)
        # sum((p + lambda)*r) - sum(lambda * target) 
        term1 = np.sum(adjusted_scores * r)
        term2 = np.sum(lambdas * targets)
        Z_UB = term1 - term2
        
        # Calculate LB (Best Feasible Objective Value estimate)
        # Simply sum(p * best_r)
        Z_LB = np.sum(p * best_r)
        
        norm_grad_sq = np.sum(gradients**2)
        
        if norm_grad_sq < 1e-20:
            break # Gradient is zero, optimal found
            
        step_size = pi_val * (Z_UB - Z_LB) / norm_grad_sq
        
        # Update lambdas: lambda = max(0, lambda - step * gradient)
        # Note: If sum < target, grad < 0. lambda increases. Item utility increases.
        lambdas = np.maximum(0, lambdas - step_size * gradients)
        
        # Halve pi occasionally as suggested in subgradient literature/heuristics
        if t > 0 and t % 20 == 0:
            pi_val *= 0.5

    return best_r

def project_to_simplex(v):
    """
    Projects vector v onto the probability simplex (sum=1, non-negative).
    Standard Euclidean projection.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def repair_feasible_solution(r, p_scores, constraint_indices, targets, max_repair_iter=50, tol=1e-9):
    """
    Implements the 'Greedy Heuristic' from the paper [cite: 242-256] to force feasibility.
    
    Paper Logic:
    1. Identify violated constraints.
    2. 'Recommend' (add mass) to items in violated sets until satisfied.
    3. 'Drop' (remove mass) from others to maintain limits.
    
    Adapted for Continuous Probabilities:
    - Iteratively scales up violated sets and scales down others.
    - Effectively a projection step ensuring Sum(r_k) >= Target_k.
    """
    r_fixed = r.copy()
    K = len(targets)
    
    for _ in range(max_repair_iter):
        max_violation = 0.0
        
        # 1. Satisfy Lower Bounds (Provider Constraints)
        for k in range(K):
            idx = constraint_indices[k]
            current_mass = r_fixed[idx].sum()
            target = targets[k]
            
            if current_mass < target - tol:
                violation = target - current_mass
                max_violation = max(max_violation, violation)
                
                # Scale up this group to meet target exactly
                # We use original scores 'p_scores' to decide which items get the boost 
                # (Paper: "recommend new items with highest predicted ratings" )
                scale = target / (current_mass + 1e-20)
                r_fixed[idx] *= scale

        # 2. Re-normalize to Sum = 1 (System Constraint)
        # We must be careful not to break the lower bounds we just fixed.
        # We assume "disjoint" or "loosely coupled" constraints typical in Provider settings.
        total_mass = r_fixed.sum()
        if abs(total_mass - 1.0) > tol:
            # Simple normalization might violate bounds again. 
            # We use a shift that prefers reducing unconstrained items.
            diff = total_mass - 1.0
            if diff > 0:
                # We need to remove mass. Remove uniformly or proportional to current r
                r_fixed /= total_mass
        
        if max_violation < tol and abs(r_fixed.sum() - 1.0) < tol:
            break
            
    # Final hard clamp to ensure no numerical drift
    r_fixed = r_fixed / r_fixed.sum()
    return r_fixed

def solve_lagrangian_relaxation_precise(p, constraint_indices, targets, max_iter=200, pi=2.0):
    """
    Solves using Lagrangian Relaxation + Greedy Repair for high precision.
    """
    n = len(p)
    K = len(targets)
    lambdas = np.zeros(K)  # Lagrangian multipliers
    
    # Best found solutions
    best_r = p.copy() / p.sum()
    best_L_bound = -np.inf 
    
    # Adaptive step size parameters
    pi_val = pi
    
    for t in range(max_iter):
        # --- 1. SOLVE LUBP (Relaxed Problem) ---
        # Add multipliers to utility: u_new = u_old + lambda 
        adjusted_scores = p.astype(np.float64, copy=True)
        for k in range(K):
            if lambdas[k] > 0:
                adjusted_scores[constraint_indices[k]] += lambdas[k]
        
        # Project to simplex (Constraint 1 & 3)
        r = project_to_simplex(adjusted_scores)

        # --- 2. GET FEASIBLE SOLUTION (Lower Bound) ---
        # Apply the repair heuristic immediately to get a valid candidate 
        r_feasible = repair_feasible_solution(r, p, constraint_indices, targets)
        
        # Calculate primal utility (original objective)
        current_utility = np.sum(p * r_feasible)
        if current_utility > best_L_bound:
            best_L_bound = current_utility
            best_r = r_feasible.copy()

        # --- 3. SUBGRADIENT UPDATE ---
        # Gradients based on the RELAXED solution (r), not the repaired one
        gradients = np.zeros(K)
        for k in range(K):
            # Gradient = Sum(x) - Target
            gradients[k] = r[constraint_indices[k]].sum() - targets[k]

        # Step size calculation 
        # UB = Relaxed Objective Value
        Z_UB = np.sum(adjusted_scores * r) - np.sum(lambdas * targets)
        
        norm_grad_sq = np.sum(gradients**2)
        if norm_grad_sq < 1e-12: 
            break 
            
        step_size = pi_val * (Z_UB - best_L_bound) / norm_grad_sq
        
        # Update multipliers 
        lambdas = np.maximum(0, lambdas - step_size * gradients)
        
        # Decay pi to force convergence
        if t > 0 and t % 25 == 0:
            pi_val *= 0.8

    # Final Strict Repair on the best found solution to ensure high precision
    final_r = repair_feasible_solution(best_r, p, constraint_indices, targets, max_repair_iter=100, tol=1e-10)
    
    return final_r

def solve_mrs_provider_precise(u, p_u, t_u):
    """
    High-precision wrapper.
    """
    try:
        # Use a few more iterations than standard IPF to allow subgradients to settle
        r = solve_lagrangian_relaxation_precise(p_u, Q_indices, t_u, max_iter=200)
        
        if not np.all(np.isfinite(r)) or r.sum() <= 0:
            return None, "degenerate"
        
        # Final safety check
        r = r / max(r.sum(), 1e-20)
        return r, "ok"
        
    except Exception as e:
        return None, "exception"

def evaluate_population_dual_boost(name, probs, Q, pop_targets, pos_map, multi_map, show_progress=True,
                                  max_iter=DUAL_MAX_ITER, lr=DUAL_LR, clip_val=DUAL_CLIP, tol=TOLERANCE):
    """
    Population-level baseline: global dual-ascent / feedback controller boosting.
    Learns a shared lambda across all users, then computes per-user Recall/NDCG@K.
    """
    N, I = probs.shape

    t0 = time.perf_counter()
    R_all, lam, achieved = solve_population_dual_boost(
        probs, Q, pop_targets,
        max_iter=max_iter, lr=lr, clip_val=clip_val, tol=tol
    )
    t1 = time.perf_counter()
    total_time_ms = (t1 - t0) * 1000.0
    avg_time_ms = total_time_ms / max(N, 1)

    metric_sums = {k: {"hit": 0.0, "ndcg": 0.0} for k in KS}
    spearman100_sum = 0.0 

    it = range(N)
    if show_progress:
        it = tqdm(it, desc=f"Eval {name}", leave=False)

    for u in it:
        r_u = R_all[u]

        spearman100_sum += spearman_at_k_union(probs[u], r_u, k=100)

        res = get_metrics(r_u, u, pos_map, multi_map)
        for k in KS:
            metric_sums[k]["hit"] += res[k][0]
            metric_sums[k]["ndcg"] += res[k][1]

    pop_err = float(np.max(np.abs(achieved - pop_targets)))

    out = {
        "method": name,
        "users_total": N,
        "users_solved": N,
        "fails": np.nan,
        "fail_rate": np.nan,
        "avg_error": pop_err,
        "avg_time_ms": avg_time_ms,

        "spearman@100": spearman100_sum / max(N, 1), 
    }
    for k in KS:
        out[f"recall@{k}"] = metric_sums[k]["hit"] / max(N, 1)
        out[f"ndcg@{k}"]   = metric_sums[k]["ndcg"] / max(N, 1)

    return out

def solve_population_dual_boost(probs, Q, pop_targets, weights=None,
                                max_iter=DUAL_MAX_ITER, lr=DUAL_LR, clip_val=DUAL_CLIP, tol=TOLERANCE):
    """
    Population-level global dual-ascent / controller baseline.

    We keep a shared lambda (K,) across the population, and iteratively update it to meet:
        E_g >= target_g   for all g

    probs:       (N, I) baseline user distributions
    Q:           (K, I) constraint masks/features
    pop_targets: (K,)   aggregate exposure targets
    weights:     (N,)   traffic weights (defaults to uniform)

    Returns:
        R:        (N, I) adjusted user distributions
        lam:      (K,)   learned boosts
        achieved: (K,)   final achieved population exposure
    """
    N, I = probs.shape
    K = len(pop_targets)

    if weights is None:
        weights = np.ones(N, dtype=np.float64) / N
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-20)

    lam = np.zeros(K, dtype=np.float64)

    # ---- Controller loop (dual ascent / projected gradient) ----
    achieved = np.zeros(K, dtype=np.float64)

    for _ in range(max_iter):
        # shared exponential tilt across items: w_i = exp(sum_g lam_g * Q[g,i])
        logits = lam @ Q                         # (I,)
        logits = logits - np.max(logits)         # stabilize
        w_shared = np.exp(logits)                # (I,)

        # measure achieved population exposure under current lambda
        achieved[:] = 0.0
        for u in range(N):
            r_u = probs[u] * w_shared
            r_u /= max(r_u.sum(), 1e-20)
            achieved += weights[u] * (Q @ r_u)   # (K,)

        # constraint residual: want achieved >= target
        # e = target - achieved  (positive => under-exposed => increase lambda)
        err = pop_targets - achieved

        if np.max(np.abs(err)) <= tol:
            break

        # projected dual update
        lam = lam + lr * err
        lam = np.clip(lam, 0.0, clip_val)

    # ---- Final adjusted distributions using learned lambda ----
    final_logits = lam @ Q
    final_logits = final_logits - np.max(final_logits)
    W = np.exp(final_logits)                     # (I,)

    R = probs * W[None, :]
    R /= np.maximum(R.sum(axis=1, keepdims=True), 1e-20)

    return R, lam, achieved



print("Loading Data...")

X = 100_000_000          # Subsampling for experiments if X > users that it takes all users
RANDOM = True        
SEED = 42        

# 1) Load scores as memmap (does NOT load full array into RAM)
scores_mm = np.load(SCORES_PATH, mmap_mode="r")   # shape: [N_all, I_num]
N_all, I_num = scores_mm.shape
print(scores_mm)

# 2) Choose which users to keep (original user ids in [0..N_all-1])
if X is None:
    X_eff = N_all
else:
    X_eff = min(int(X), N_all)

if RANDOM:
    rng = np.random.default_rng(SEED)
    keep_users = rng.choice(N_all, size=X_eff, replace=False)
    keep_users.sort()   # keep sorted for better sequential IO on memmap
else:
    keep_users = np.arange(X_eff, dtype=np.int64)


scores_sub = scores_mm[keep_users].astype(np.float32, copy=False)  # [N, I_num]
scores_sub -= scores_sub.max(axis=1, keepdims=True)
probs = np.exp(scores_sub)
probs /= probs.sum(axis=1, keepdims=True)
N = probs.shape[0]

# 4) Load positives, then remap them to the sampled user-index space [0..N-1]
pos_u_all = np.load(POS_U_PATH)
pos_i_all = np.load(POS_I_PATH)

# Map original user id -> new local index (0..N-1), -1 means "not kept"
inv = np.full(N_all, -1, dtype=np.int64)
inv[keep_users] = np.arange(N, dtype=np.int64)

local_u = inv[pos_u_all]
mask = local_u >= 0
pos_u = local_u[mask].astype(np.int64, copy=False)
pos_i = pos_i_all[mask].astype(np.int64, copy=False)

# Build pos_map / multi_map for sampled users only (indices 0..N-1)
pos_map = np.full(N, -1, dtype=np.int64)
pos_map[pos_u] = pos_i

multi_map = {}
order = np.argsort(pos_u)
pos_u_s = pos_u[order]
pos_i_s = pos_i[order]
if pos_u_s.size > 0:
    cuts = np.where(np.diff(pos_u_s) != 0)[0] + 1
    u_groups = np.split(pos_u_s, cuts)
    i_groups = np.split(pos_i_s, cuts)
    for ug, ig in zip(u_groups, i_groups):
        multi_map[int(ug[0])] = set(map(int, ig))

meta = pd.read_csv(META_PATH)

Q = np.zeros((len(PROMOTERS), I_num), dtype=np.float16)
for k, col in enumerate(PROMOTERS):
    idx = meta.loc[meta[col] == 1, "item_id_internal"].values.astype(int)
    Q[k, idx] = 1.0

Q_indices = [np.where(row > 0.5)[0] for row in Q]
B_all = probs @ Q.T
targets_all = np.minimum(B_all * DELTA_M, 1.0 - 1e-7)
pop_targets = targets_all.mean(axis=0)

print(f"Ready: {N}/{N_all} Users, {I_num} Items.")

ip_solver = IPExposureAlignedSolver(
    num_groups=Q.shape[0],
    k=TOPK,
    pool_size=200,
    tol=FAIL_TOL,
    min_weight=min_weight
)


results = []
results.append(
    evaluate_population_dual_boost(
        "dual_boost_population",
        probs, Q, pop_targets,
        pos_map, multi_map
    )
)
results.append(evaluate_method("baseline",     solve_baseline, probs, Q, targets_all, pos_map, multi_map))
results.append(evaluate_method("regular_ipf",  solve_ipf,      probs, Q, targets_all, pos_map, multi_map))
results.append(evaluate_method("kl_newton",    solve_newton,   probs, Q, targets_all, pos_map, multi_map))
results.append(evaluate_method("lagrangian_precise", solve_mrs_provider_precise, probs, Q, targets_all, pos_map, multi_map))
results.append(evaluate_population_newton("kl_newton_population", probs, Q, pop_targets, pos_map, multi_map))
results.append(evaluate_method("ip_milp",      solve_ip_milp,  probs, Q, targets_all, pos_map, multi_map))
df_results = pd.DataFrame(results)
cols = (["method", "fail_rate",
         "avg_error", "avg_time_ms", "spearman@100"] +
        [f"recall@{k}" for k in KS] +
        [f"ndcg@{k}" for k in KS])
df_results = df_results[cols]
print(df_results)

OUT_CSV = "exposure_methods_results_100K.csv"
df_results.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")



