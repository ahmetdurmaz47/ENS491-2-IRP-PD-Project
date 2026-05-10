"""
================================================================
 IRP-PD MIP-LNS IMPROVEMENT HEURISTIC
================================================================
Reads:
  --excel    : original problem data  (instance_expanded.xlsx)
  --solution : a .txt file containing the ROUTES section copied
               from main_lazy.py or main_miller.py output

Applies a 5-phase MIP-based Large Neighborhood Search.  The phases are
chosen so each operator targets a DIFFERENT structural dimension of the
solution; redundant random / overlapping operators were intentionally
removed.

  Phase A : Quantity-only re-optimization (LP, fix all x,y)
            -> service-amount dimension
  Phase B : Period-by-period destroy & MIP repair
            -> time dimension, intra-period routing
  Phase C : Pair-of-periods destroy & MIP repair
            -> inter-period coupling (inventory/service trade-offs)
  Phase E : Shaw / related-customer (spatial) destroy & MIP repair
            -> spatial / geographic dimension
  Phase F : Worst-detour customer destroy & MIP repair (greedy targeted)
            -> cost-contribution dimension (weakest insertions)

Prints the improved solution in the SAME format as the MIP scripts.
The heuristic NEVER worsens the input solution: every sub-problem
contains the input as a feasible point, so the worst case is "no
improvement, return original".

USAGE:
  python improvement_heuristic.py \
      --excel instance_expanded.xlsx \
      --solution mip_output.txt \
      --time_limit 300

Author: project-internal heuristic for the IRP-PD project.
================================================================
"""

from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import argparse
import random
import re
import sys
import time


# ============================================================
# 1) DATA LOADER  (mirrors main_lazy.py / main_miller.py)
# ============================================================

def load_irppd_from_excel(xlsx_path,
                          depot_inventory_bigM=1e9,
                          depot_holding_cost=0.0):
    expanded = pd.read_excel(xlsx_path, sheet_name="ExpandedData", engine="openpyxl")
    meta_df  = pd.read_excel(xlsx_path, sheet_name="Meta",         engine="openpyxl")
    dist     = pd.read_excel(xlsx_path, sheet_name="DistanceMatrix",
                             header=None, engine="openpyxl").values

    if meta_df.empty:
        raise ValueError("Meta sheet is empty.")
    meta = meta_df.iloc[0].to_dict()

    T_count   = int(meta["T"])
    Q         = float(meta["Q"])
    K_max     = int(meta.get("K", len(expanded)))
    n_nodes   = int(meta["n_nodes"])

    expanded  = expanded.sort_values("Customer").reset_index(drop=True)
    orig_ids  = expanded["Customer"].astype(int).tolist()
    n_custs   = len(orig_ids)

    if n_custs != n_nodes - 1:
        raise ValueError(f"ExpandedData has {n_custs} customers but Meta says n_nodes={n_nodes}")
    if dist.shape[0] != n_nodes or dist.shape[1] != n_nodes:
        raise ValueError(f"DistanceMatrix shape {dist.shape} != ({n_nodes},{n_nodes})")

    depot = 0
    N     = list(range(1, n_custs + 1))
    N_all = [depot] + N
    T     = list(range(1, T_count + 1))

    internal_to_orig = {i: orig_ids[i - 1] for i in N}

    I0 = {depot: float(depot_inventory_bigM)}
    L  = {depot: 0.0}
    U  = {depot: float("inf")}
    h  = {depot: float(depot_holding_cost)}
    r  = {}

    for idx, row in expanded.iterrows():
        i = idx + 1
        r[i]  = float(row["ri"])
        L[i]  = float(row["LB"])
        U[i]  = float(row["UB"])
        I0[i] = float(row["I0"])
        h[i]  = float(row["h"]) if not pd.isna(row["h"]) else 0.0

    d = {}
    for idx, row in expanded.iterrows():
        i = idx + 1
        for t in T:
            d[(i, t)] = int(row[f"d_{t}"])
    for t in T:
        d[(depot, t)] = 0

    c = {}
    for i in N_all:
        for j in N_all:
            if i != j:
                c[(i, j)] = float(dist[i, j])

    return {
        "N": N, "T": T, "Q": Q, "m": K_max,
        "d": d, "h": h, "L": L, "U": U, "I0": I0, "c": c, "r": r,
        "internal_to_orig": internal_to_orig, "depot": depot,
    }


# ============================================================
# 2) PARSE THE MIP OUTPUT TEXT FILE
# ============================================================

def parse_solution_txt(txt_path):
    """
    Parses the routes and service amounts from the routes-section of
    main_lazy.py / main_miller.py output. Robust to extra lines.

    Returns:
      routes_by_t  : { t : [[depot, i1, i2, ..., depot], ...] }   (internal ids)
      service_by_t : { t : { i_internal : (q_plus, q_minus) } }
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    routes_by_t  = {}
    service_by_t = {}

    period_starts = list(re.finditer(r"Period\s+t\s*=\s*(\d+)", text))
    if not period_starts:
        raise ValueError("No 'Period t=...' lines found in the input txt.")

    for k, m in enumerate(period_starts):
        t      = int(m.group(1))
        start  = m.end()
        end    = period_starts[k + 1].start() if k + 1 < len(period_starts) else len(text)
        block  = text[start:end]

        routes_by_t[t]  = []
        service_by_t[t] = {}

        # Routes
        for rm in re.finditer(r"Route\s+\d+\s*:\s*([^\n]+)", block):
            tokens = [tok.strip() for tok in rm.group(1).split("->")]
            route  = []
            for tok in tokens:
                m2 = re.match(r"(\d+)", tok)
                if m2:
                    route.append(int(m2.group(1)))
            if route:
                routes_by_t[t].append(route)

        # Service amounts
        for sm in re.finditer(
            r"i\s*=\s*(\d+)(?:\([^)]*\))?\s*:\s*q_plus\s*=\s*([\-0-9.]+)\s*,\s*q_minus\s*=\s*([\-0-9.]+)",
            block,
        ):
            i  = int(sm.group(1))
            qp = float(sm.group(2))
            qm = float(sm.group(3))
            service_by_t[t][i] = (qp, qm)

    return routes_by_t, service_by_t


# ============================================================
# 3) ROUTES  →  x, y  DICTIONARIES
# ============================================================

def routes_to_xy(data, routes_by_t):
    N     = data["N"]
    T     = data["T"]
    depot = data["depot"]
    N_all = [depot] + list(N)

    x_init = {(i, j, t): 0
              for i in N_all for j in N_all if i != j for t in T}
    y_init = {(i, t): 0 for i in N for t in T}

    for t, routes in routes_by_t.items():
        for route in routes:
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                if i != j:
                    x_init[(i, j, t)] = 1
            for node in route:
                if node != depot:
                    y_init[(node, t)] = 1
    return x_init, y_init


# ============================================================
# 4) BUILD THE MASTER MIP (MTZ-based, used for every sub-problem)
# ============================================================

def build_master_model(data, verbose=0):
    N     = data["N"]
    T     = data["T"]
    Q     = data["Q"]
    m_max = data["m"]
    d     = data["d"]
    h     = data["h"]
    L     = data["L"]
    U     = data["U"]
    I0    = data["I0"]
    c     = data["c"]
    depot = data["depot"]

    N_all = [depot] + list(N)
    A     = [(i, j) for i in N_all for j in N_all if i != j]
    Kcap  = {i: min(Q, U[i] - L[i]) for i in N}

    model = Model("IRP_PD_LNS_MASTER")
    model.Params.OutputFlag = verbose

    T0     = [0] + T
    I      = model.addVars(N_all, T0, vtype=GRB.CONTINUOUS, name="I")
    y      = model.addVars(N, T,      vtype=GRB.BINARY,     name="y")
    y0     = model.addVars(T,         vtype=GRB.INTEGER,    name="y0")
    x      = model.addVars(A, T,      vtype=GRB.BINARY,     name="x")
    l      = model.addVars(A, T,      vtype=GRB.CONTINUOUS, name="l")
    q_plus  = model.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0.0, name="q_plus")
    q_minus = model.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0.0, name="q_minus")

    n_cust = len(N)
    u = model.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0.0, ub=n_cust, name="u")

    # ---- Objective ----
    model.setObjective(
        quicksum(c[i, j] * x[i, j, t] for (i, j) in A for t in T) +
        quicksum(h[i] * I[i, t]       for i in N_all for t in T),
        GRB.MINIMIZE,
    )

    # ---- Standard IRP-PD constraints (same as main_miller.py) ----
    for t in T:
        model.addConstr(quicksum(x[depot, j, t] for j in N) == y0[t])

    for t in T:
        for i in N_all:
            model.addConstr(
                quicksum(x[i, j, t] for j in N_all if i != j) -
                quicksum(x[j, i, t] for j in N_all if i != j) == 0
            )

    for t in T:
        for i in N:
            model.addConstr(quicksum(x[i, j, t] for j in N_all if i != j) == y[i, t])

    for t in T:
        for i in N:
            model.addConstr(
                I[i, t] - I[i, t - 1] - q_plus[i, t] + q_minus[i, t] + d[(i, t)] == 0
            )

    for t in T:
        model.addConstr(
            I[depot, t] - I[depot, t - 1]
            + quicksum(q_plus[i, t]  for i in N)
            - quicksum(q_minus[i, t] for i in N) == 0
        )

    for i in N_all:
        model.addConstr(I[i, 0] == I0[i])

    for t in T:
        for i in N:
            model.addConstr(I[i, t] >= L[i])
            model.addConstr(I[i, t] <= U[i])

    for t in T:
        for i in N:
            model.addConstr(q_plus[i, t]  <= Kcap[i] * y[i, t])
            model.addConstr(q_minus[i, t] <= Kcap[i] * y[i, t])

    # type by sign of d (delivery / pickup / none in each (i,t))
    for t in T:
        for i in N:
            dit = d[(i, t)]
            if dit > 0:
                model.addConstr(q_minus[i, t] == 0.0)
            elif dit < 0:
                model.addConstr(q_plus[i, t] == 0.0)
            else:
                model.addConstr(q_plus[i, t]  == 0.0)
                model.addConstr(q_minus[i, t] == 0.0)

    for t in T:
        model.addConstr(y0[t] <= m_max)
        for i in N:
            model.addConstr(y[i, t] <= y0[t])

    for t in T:
        for (i, j) in A:
            model.addConstr(l[i, j, t] <= Q * x[i, j, t])

    for t in T:
        for i in N:
            model.addConstr(
                quicksum(l[j, i, t] for j in N_all if j != i) +
                q_minus[i, t] - q_plus[i, t] -
                quicksum(l[i, j, t] for j in N_all if j != i) == 0
            )

    for t in T:
        model.addConstr(
            quicksum(l[depot, j, t] for j in N) <= I[depot, t - 1]
        )

    # MTZ subtour elimination
    for t in T:
        for i in N:
            model.addConstr(u[i, t] <= n_cust * y[i, t])
        for i in N:
            for j in N:
                if i == j:
                    continue
                model.addConstr(
                    u[i, t] - u[j, t] + n_cust * x[i, j, t] <= n_cust - 1
                )

    model._x       = x
    model._y       = y
    model._q_plus  = q_plus
    model._q_minus = q_minus
    model._I       = I
    model._A       = A
    return model


# ============================================================
# 5) DESTRUCTION OPERATORS (bound updates only)
# ============================================================
# DESTROY KISMI
def set_destruction(model, data,
                    x_cur, y_cur, qp_cur, qm_cur,
                    destroy_periods=None, destroy_customers=None,
                    destroy_pairs=None,
                    fix_quantities_only=False):
    """
    Reconfigures variable bounds in-place to implement the chosen
    neighborhood. The current incumbent is supplied as MIP-Start.

    Three destruction modes (can be combined; their union is freed):
      destroy_periods   : iterable of t -> free EVERY (i,t) in those periods
      destroy_customers : iterable of i -> free (i,t) for all t
      destroy_pairs     : iterable of (i,t) tuples -> free exactly those pairs
                          (used by surgical, period-restricted operators)
    """
    N     = data["N"]
    T     = data["T"]
    depot = data["depot"]
    N_all = [depot] + list(N)
    A     = [(i, j) for i in N_all for j in N_all if i != j]

    x  = model._x
    y  = model._y
    qp = model._q_plus
    qm = model._q_minus

    if fix_quantities_only:
        for (i, j) in A:
            for t in T:
                v = x_cur[(i, j, t)]
                x[i, j, t].LB = v
                x[i, j, t].UB = v
        for i in N:
            for t in T:
                v = y_cur[(i, t)]
                y[i, t].LB = v
                y[i, t].UB = v
        for i in N:
            for t in T:
                qp[i, t].Start = qp_cur.get((i, t), 0.0)
                qm[i, t].Start = qm_cur.get((i, t), 0.0)
        model.update()
        return

    # ---- Build the union of freed (customer, period) pairs ----
    freed = set()
    if destroy_pairs:
        for p in destroy_pairs:
            freed.add(tuple(p))
    if destroy_periods:
        for t in destroy_periods:
            for i in N:
                freed.add((i, t))
    if destroy_customers:
        for i in destroy_customers:
            for t in T:
                freed.add((i, t))

    # ---- Apply bounds ----
    # An arc (i,j,t) is free if either endpoint's visit (i,t) or (j,t) is free.
    # Depot has no y-variable; arcs touching depot inherit freedom from the
    # other endpoint, which is the natural rule.
    for (i, j) in A:
        for t in T:
            v = x_cur[(i, j, t)]
            if (i, t) in freed or (j, t) in freed:
                x[i, j, t].LB = 0
                x[i, j, t].UB = 1
                x[i, j, t].Start = v
            else:
                x[i, j, t].LB = v
                x[i, j, t].UB = v

    for i in N:
        for t in T:
            v = y_cur[(i, t)]
            if (i, t) in freed:
                y[i, t].LB = 0
                y[i, t].UB = 1
                y[i, t].Start = v
            else:
                y[i, t].LB = v
                y[i, t].UB = v

    for i in N:
        for t in T:
            qp[i, t].Start = qp_cur.get((i, t), 0.0)
            qm[i, t].Start = qm_cur.get((i, t), 0.0)

    model.update()
# Destroy Bitiş

def extract_solution(model, data):
    N     = data["N"]
    T     = data["T"]
    depot = data["depot"]
    N_all = [depot] + list(N)
    A     = [(i, j) for i in N_all for j in N_all if i != j]
    x  = model._x
    y  = model._y
    qp = model._q_plus
    qm = model._q_minus
    x_new  = {(i, j, t): int(round(x[i, j, t].X)) for (i, j) in A for t in T}
    y_new  = {(i, t):    int(round(y[i, t].X))    for i in N      for t in T}
    qp_new = {(i, t): max(0.0, qp[i, t].X)        for i in N      for t in T}
    qm_new = {(i, t): max(0.0, qm[i, t].X)        for i in N      for t in T}
    return x_new, y_new, qp_new, qm_new


# ============================================================
# 6) ROUTE EXTRACTION + OBJECTIVE COMPUTATION
# ============================================================

def extract_routes_from_x(x_dict, N_all, t, depot=0):
    succ = {}
    for i in N_all:
        if i == depot:
            continue
        for j in N_all:
            if i == j:
                continue
            if x_dict.get((i, j, t), 0) >= 1:
                succ[i] = j
                break

    starts = [j for j in N_all
              if j != depot and x_dict.get((depot, j, t), 0) >= 1]

    routes = []
    for first in starts:
        route = [depot, first]
        cur   = first
        seen  = {depot, first}
        while True:
            nxt = succ.get(cur)
            if nxt is None:
                break
            route.append(nxt)
            if nxt == depot:
                break
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
        routes.append(route)
    return routes


# ============================================================
#  NEW DESTROY-OPERATOR HELPERS
# ============================================================

def _shaw_cluster(data, k, seed):
    """
    Shaw / related-customer removal (Shaw, 1998).
    Given a SEED customer, returns the seed plus the K-1 customers closest
    to it by routing cost  c[seed,j].  The freed cluster is geographically
    coherent, giving Gurobi a contiguous region to re-route.
    """
    N = data["N"]
    c = data["c"]
    cands = []
    for j in N:
        if j == seed:
            continue
        d = 0.5 * (c.get((seed, j), 1e18) + c.get((j, seed), 1e18))
        cands.append((j, d))
    cands.sort(key=lambda t: t[1])
    return [seed] + [j for j, _ in cands[: max(0, k - 1)]]


def _worst_detour_customers(data, x_dict, k):
    """
    Greedy 'worst removal' (Pisinger & Ropke).
    For each customer  i  visited in period  t  with predecessor  p  and
    successor  s ,
        detour(i,t) = c[p,i] + c[i,s] - c[p,s]
    This is the cost saved if  i  is bypassed in that period.  Customers
    are ranked by their TOTAL detour across all periods; the worst K are
    returned (they are the best candidates for re-insertion elsewhere).
    """
    N     = data["N"]
    T     = data["T"]
    depot = data["depot"]
    c     = data["c"]
    N_all = [depot] + list(N)

    detour = {i: 0.0 for i in N}
    for t in T:
        routes = extract_routes_from_x(x_dict, N_all, t, depot=depot)
        for route in routes:
            for pos in range(1, len(route) - 1):
                p = route[pos - 1]
                i = route[pos]
                s = route[pos + 1]
                detour[i] += (c.get((p, i), 0.0)
                              + c.get((i, s), 0.0)
                              - c.get((p, s), 0.0))

    ranked = sorted(detour.items(), key=lambda kv: -kv[1])
    return [i for i, dval in ranked if dval > 1e-6][: k]


def _longest_arc_pairs(data, x_dict, k_arcs):
    """
    Surgical 'longest-arc' destroy.
    Locates the K active arcs (i,j,t) with greatest cost  c[i,j]  and
    returns the destroy_pairs set { (i,t), (j,t) }  for each of them
    (depot itself is excluded, since it has no y variable).

    The freed sub-MIP is much smaller than a full period destroy; only
    the most expensive parts of the routing are exposed for re-routing.
    """
    T     = data["T"]
    c     = data["c"]
    depot = data["depot"]
    N_all = [depot] + list(data["N"])

    active = []
    for t in T:
        for i in N_all:
            for j in N_all:
                if i == j:
                    continue
                if x_dict.get((i, j, t), 0) >= 1:
                    active.append((c[(i, j)], i, j, t))
    active.sort(key=lambda x: -x[0])

    pairs = set()
    for _, i, j, t in active[: k_arcs]:
        if i != depot:
            pairs.add((i, t))
        if j != depot:
            pairs.add((j, t))
    return pairs


# ============================================================


def compute_obj(data, x_dict, qp_dict, qm_dict):
    N     = data["N"]
    T     = data["T"]
    depot = data["depot"]
    c     = data["c"]
    h     = data["h"]
    d     = data["d"]
    I0    = data["I0"]
    N_all = [depot] + list(N)
    A     = [(i, j) for i in N_all for j in N_all if i != j]

    routing = sum(c[(i, j)] * x_dict.get((i, j, t), 0)
                  for (i, j) in A for t in T)

    I = {(i, 0): I0[i] for i in N_all}
    holding = 0.0
    for t in T:
        for i in N:
            qp = qp_dict.get((i, t), 0.0)
            qm = qm_dict.get((i, t), 0.0)
            I[(i, t)] = I[(i, t - 1)] + qp - qm - d[(i, t)]
        depot_qp = sum(qp_dict.get((i, t), 0.0) for i in N)
        depot_qm = sum(qm_dict.get((i, t), 0.0) for i in N)
        I[(depot, t)] = I[(depot, t - 1)] - depot_qp + depot_qm
        for i in N_all:
            holding += h[i] * I[(i, t)]
    return routing + holding


# ============================================================
# 7) MAIN HEURISTIC LOOP
# ============================================================

def _solve_phase(model, data, x_cur, y_cur, qp_cur, qm_cur,
                 best_obj, label,
                 sub_time_limit, mip_gap=1e-4, verbose=1, **destroy_kwargs):
    """
    Apply destruction + solve + accept-if-better.
    Returns the (possibly updated) tuple (best_obj, x_cur, y_cur, qp_cur, qm_cur, improved_flag).
    """
    set_destruction(model, data, x_cur, y_cur, qp_cur, qm_cur, **destroy_kwargs)
    model.Params.TimeLimit = sub_time_limit
    model.Params.MIPGap    = mip_gap
    model.optimize()

    improved = False
    if model.SolCount > 0 and model.ObjVal < best_obj - 1e-6:
        x_cur, y_cur, qp_cur, qm_cur = extract_solution(model, data)
        delta    = best_obj - model.ObjVal
        new_obj  = model.ObjVal
        if verbose:
            pct = 100 * delta / best_obj if best_obj > 0 else 0.0
            print(f"   [{label}] IMPROVED  {best_obj:.4f}  ->  {new_obj:.4f}   "
                  f"(Δ={delta:.4f},  {pct:.3f}%)")
        best_obj = new_obj
        improved = True
    else:
        if verbose:
            print(f"   [{label}] no improvement")
    return best_obj, x_cur, y_cur, qp_cur, qm_cur, improved


def heuristic(data, x_init, y_init, qp_init, qm_init,
              time_limit=300, seed=42, verbose=1,
              sub_time_limit=60):
    rng = random.Random(seed)
    N   = data["N"]
    T   = data["T"]

    init_obj = compute_obj(data, x_init, qp_init, qm_init)
    if verbose:
        print(f"\n[Heuristic] Initial objective parsed from input = {init_obj:.6f}")
        print(f"[Heuristic] Building master model...")

    model = build_master_model(data, verbose=0)
    model.Params.MIPFocus = 1   # focus on improving feasibles

    best_obj, best_x, best_y, best_qp, best_qm = (
        init_obj, dict(x_init), dict(y_init), dict(qp_init), dict(qm_init)
    )

    start = time.time()
    def remaining():
        return max(1, int(time_limit - (time.time() - start)))

    # ---- Phase A : Quantity-only LP ----
    if verbose: print("\n[Phase A] Quantity-only re-optimization (fix all routes)")
    best_obj, best_x, best_y, best_qp, best_qm, _ = _solve_phase(
        model, data, best_x, best_y, best_qp, best_qm,
        best_obj, "Phase A",
        sub_time_limit=min(60, remaining()), verbose=verbose,
        fix_quantities_only=True,
    )
    if remaining() <= 5: return _final(best_obj, best_x, best_y, best_qp, best_qm)

    # ---- Phase B : Single-period destroy ----
    if verbose: print("\n[Phase B] Single-period destroy & MIP repair")
    for t in T:
        if remaining() <= 5: break
        label = f"Phase B  t={t}"
        best_obj, best_x, best_y, best_qp, best_qm, _ = _solve_phase(
            model, data, best_x, best_y, best_qp, best_qm,
            best_obj, label,
            sub_time_limit=min(sub_time_limit, remaining()), verbose=verbose,
            destroy_periods=[t],
        )

    # ---- Phase C : Pair-of-periods destroy ----
    if verbose: print("\n[Phase C] Pair-of-periods destroy & MIP repair")
    pairs = [(a, b) for a in T for b in T if a < b]
    rng.shuffle(pairs)
    for (t1, t2) in pairs:
        if remaining() <= 5: break
        label = f"Phase C  ({t1},{t2})"
        best_obj, best_x, best_y, best_qp, best_qm, _ = _solve_phase(
            model, data, best_x, best_y, best_qp, best_qm,
            best_obj, label,
            sub_time_limit=min(sub_time_limit + 30, remaining()), verbose=verbose,
            destroy_periods=[t1, t2],
        )

    n_cust = len(N)

    # ---- Phase E : Shaw / related-customer destroy (spatial cluster) ----
    if verbose: print("\n[Phase E] Shaw / related-customer destroy & MIP repair")
    k_shaw   = max(3, n_cust // 5)
    n_iters_E = 5
    for it in range(1, n_iters_E + 1):
        if remaining() <= 5: break
        seed = rng.choice(N)
        cluster = _shaw_cluster(data, k=k_shaw, seed=seed)
        label = f"Phase E  iter={it}  seed={seed}  |S|={len(cluster)}"
        best_obj, best_x, best_y, best_qp, best_qm, _ = _solve_phase(
            model, data, best_x, best_y, best_qp, best_qm,
            best_obj, label,
            sub_time_limit=min(sub_time_limit, remaining()), verbose=verbose,
            destroy_customers=cluster,
        )#16-25 destroy

    # ---- Phase F : Worst-detour customer destroy (greedy targeted) ----
    if verbose: print("\n[Phase F] Worst-detour customer destroy & MIP repair")
    k_worst   = max(4, n_cust // 5)
    n_iters_F = 4
    for it in range(1, n_iters_F + 1):
        if remaining() <= 5: break
        worst = _worst_detour_customers(data, best_x, k=k_worst)
        if not worst:
            if verbose: print("   [Phase F] no positive-detour customers found, skip")
            break
        top3 = worst[:3]
        label = f"Phase F  iter={it}  |S|={len(worst)}  top-detour={top3}"
        best_obj, best_x, best_y, best_qp, best_qm, _ = _solve_phase(
            model, data, best_x, best_y, best_qp, best_qm,
            best_obj, label,
            sub_time_limit=min(sub_time_limit, remaining()), verbose=verbose,
            destroy_customers=worst,
        )

    return _final(best_obj, best_x, best_y, best_qp, best_qm)


def _final(best_obj, best_x, best_y, best_qp, best_qm):
    return best_x, best_y, best_qp, best_qm, best_obj


# ============================================================
# 8) MIP-STYLE PRINTING
# ============================================================

def print_solution_routes(data, x_dict, qp_dict, qm_dict, header=None):
    N     = data["N"]
    T     = data["T"]
    depot = data["depot"]
    N_all = [depot] + list(N)
    internal_to_orig = data.get("internal_to_orig", {})

    def fmt(i):
        if i == depot: return "0"
        return f"{i}(orig={internal_to_orig.get(i,'?')})"

    title = header or "IMPROVED ROUTES (by period)"
    print(f"\n================= {title} =================")

    for t in T:
        routes = extract_routes_from_x(x_dict, N_all, t, depot=depot)
        print(f"\nPeriod t={t} | #routes={len(routes)}")
        if not routes:
            print("  (no routes)")
            continue
        for k, rt in enumerate(routes, 1):
            print(f"  Route {k}: " + " -> ".join(fmt(n) for n in rt))

        served = []
        for i in N:
            qp = qp_dict.get((i, t), 0.0)
            qm = qm_dict.get((i, t), 0.0)
            if qp > 1e-6 or qm > 1e-6:
                served.append((i, qp, qm))
        if served:
            print("  service amounts (q_plus=delivery, q_minus=pickup):")
            for i, qp, qm in served:
                print(f"    i={fmt(i)}: q_plus={qp:.2f}, q_minus={qm:.2f}")

    print("\n" + "=" * 60 + "\n")


# ============================================================
# 9) MAIN
# ============================================================

def _autodetect_solution_txt():
    """
    If --solution is not given, scan the current directory for a .txt file
    that contains the MIP ROUTES section. Return its path or None.
    """
    import glob, os
    candidates = []
    for path in sorted(glob.glob("*.txt")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
            if "Period t" in head and ("Route" in head or "ROUTES" in head):
                candidates.append(path)
        except Exception:
            continue
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="MIP-LNS improvement heuristic for IRP-PD"
    )
    parser.add_argument("--excel",      default="instance_expanded.xlsx",
                        help="path to the original problem .xlsx")
    parser.add_argument("--solution",   default=None,
                        help="path to .txt file containing the MIP ROUTES output "
                             "(optional; if omitted, the script auto-detects a "
                             "single .txt file in the current folder)")
    parser.add_argument("--time_limit", type=int, default=900,
                        help="overall heuristic time limit (seconds)")
    parser.add_argument("--sub_time",   type=int, default=60,
                        help="per-subproblem time limit (seconds)")
    parser.add_argument("--seed",       type=int, default=42)
    # parse_known_args -> tolerates extra args coming from IPython/Spyder kernels
    args, _unknown = parser.parse_known_args()

    # ---- Resolve --solution if not given ----
    if not args.solution:
        cands = _autodetect_solution_txt()
        if len(cands) == 0:
            print("ERROR: --solution not given and no candidate .txt with a "
                  "ROUTES section was found in the current folder.\n"
                  "Either run with:\n"
                  "    python improvement_heuristic.py --solution YOUR_FILE.txt\n"
                  "or place exactly one MIP-output .txt next to this script.")
            sys.exit(2)
        if len(cands) > 1:
            print("ERROR: --solution not given and multiple candidate .txt files "
                  "were found:\n  " + "\n  ".join(cands) +
                  "\nPlease pass --solution explicitly.")
            sys.exit(2)
        args.solution = cands[0]
        print(f"[Auto-detect] using solution file: {args.solution}")

    print("=" * 64)
    print("  IRP-PD MIP-LNS IMPROVEMENT HEURISTIC")
    print("=" * 64)
    print(f"  Excel    : {args.excel}")
    print(f"  Solution : {args.solution}")
    print(f"  Time lim : {args.time_limit} s   (sub: {args.sub_time} s)")
    print(f"  Seed     : {args.seed}")
    print("=" * 64)

    data = load_irppd_from_excel(args.excel)
    routes_by_t, service_by_t = parse_solution_txt(args.solution)

    if not routes_by_t:
        print("ERROR: could not parse any routes from the solution file.")
        sys.exit(1)

    # ---- Sanity checks: txt vs xlsx ----
    txt_internal_ids = set()
    for t, rs in routes_by_t.items():
        for r in rs:
            for n in r:
                if n != 0:
                    txt_internal_ids.add(n)
    for t, svc in service_by_t.items():
        txt_internal_ids.update(svc.keys())

    excel_N = set(data["N"])
    txt_max = max(txt_internal_ids) if txt_internal_ids else 0
    excel_max = max(excel_N) if excel_N else 0
    extra = txt_internal_ids - excel_N
    if extra:
        print(
            "ERROR: customer-ID mismatch between solution.txt and instance_expanded.xlsx\n"
            f"  Excel has {len(excel_N)} customers (internal ids 1..{excel_max})\n"
            f"  TXT references {len(txt_internal_ids)} ids up to {txt_max}\n"
            f"  Unknown ids found in TXT (not in Excel): {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}\n"
            "Most likely cause: instance_expanded.xlsx was not regenerated for the\n"
            "instance you ran the MIP on. Re-run 'data generation.py' with the same\n"
            "DAT file you used for the MIP, then retry."
        )
        sys.exit(1)

    txt_periods = set(routes_by_t.keys())
    excel_T = set(data["T"])
    if txt_periods - excel_T:
        print(
            f"ERROR: TXT references periods {sorted(txt_periods)} but Excel only has "
            f"T={sorted(excel_T)}. Re-generate the Excel for this instance."
        )
        sys.exit(1)

    print(f"\n[Parser] periods parsed: {sorted(routes_by_t.keys())}")
    n_routes_total = sum(len(rs) for rs in routes_by_t.values())
    print(f"[Parser] total routes parsed: {n_routes_total}")
    print(f"[Parser] {len(txt_internal_ids)} distinct customers referenced "
          f"(matches Excel: {txt_internal_ids.issubset(excel_N)})")

    x_init, y_init = routes_to_xy(data, routes_by_t)
    qp_init = {(i, t): 0.0 for i in data["N"] for t in data["T"]}
    qm_init = {(i, t): 0.0 for i in data["N"] for t in data["T"]}
    for t, svc in service_by_t.items():
        for i, (qp, qm) in svc.items():
            qp_init[(i, t)] = qp
            qm_init[(i, t)] = qm

    init_obj = compute_obj(data, x_init, qp_init, qm_init)
    print(f"[Parser] initial objective recomputed = {init_obj:.6f}")

    t0 = time.time()
    best_x, best_y, best_qp, best_qm, best_obj = heuristic(
        data, x_init, y_init, qp_init, qm_init,
        time_limit=args.time_limit,
        sub_time_limit=args.sub_time,
        seed=args.seed,
        verbose=1,
    )
    elapsed = time.time() - t0

    delta = init_obj - best_obj
    pct   = 100 * delta / init_obj if init_obj > 0 else 0.0

    print("\n" + "=" * 64)
    print("  HEURISTIC SUMMARY")
    print("=" * 64)
    print(f"  Initial objective : {init_obj:.6f}")
    print(f"  Improved objective: {best_obj:.6f}")
    print(f"  Absolute change   : {delta:.6f}")
    print(f"  Percent change    : {pct:.4f} %")
    print(f"  Total time        : {elapsed:.2f} s")
    print("=" * 64)

    print_solution_routes(data, best_x, best_qp, best_qm,
                          header="IMPROVED ROUTES (by period)")

    print(f"\nFinal improved objective value: {best_obj:.6f}")


if __name__ == "__main__":
    main()