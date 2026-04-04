from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
from typing import List

# ============================================================
# 0) ROUTE PRINTING HELPERS
# ============================================================

def extract_routes_from_x(x, N_all: List[int], t: int, depot: int = 0, eps: float = 0.5) -> List[List[int]]:
    """Period t için x[i,j,t] değerlerinden depot->...->depot rotalarını çıkarır."""
    succ = {}
    for i in N_all:
        if i == depot:
            continue
        for j in N_all:
            if i == j:
                continue
            if (i, j, t) in x and x[i, j, t].X > eps:
                succ[i] = j
                break

    starts = []
    for j in N_all:
        if j == depot:
            continue
        if (depot, j, t) in x and x[depot, j, t].X > eps:
            starts.append(j)

    routes = []
    for first in starts:
        route = [depot, first]
        cur = first
        seen = {depot, first}

        while True:
            nxt = succ.get(cur, None)
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


def print_solution_routes(model, depot: int = 0, eps: float = 0.5, show_q: bool = True) -> None:
    if model.SolCount == 0:
        print("No solution available.")
        return

    x = model._x
    q_plus = model._q_plus
    q_minus = model._q_minus
    N = model._N
    N_all = model._N_all
    T = model._T
    internal_to_orig = getattr(model, "_internal_to_orig", None)

    def fmt_node(i: int) -> str:
        if internal_to_orig is None:
            return str(i)
        if i == depot:
            return "0"
        return f"{i}(orig={internal_to_orig.get(i,'?')})"

    print("\n================= ROUTES (by period) =================")
    for t in T:
        routes = extract_routes_from_x(x, N_all, t, depot=depot, eps=eps)
        print(f"\nPeriod t={t} | #routes={len(routes)}")
        if not routes:
            print("  (no routes)")
            continue

        for k, rt in enumerate(routes, 1):
            print(f"  Route {k}: " + " -> ".join(fmt_node(n) for n in rt))

        if show_q:
            served = []
            for i in N:
                dp = q_plus[i, t].X
                pm = q_minus[i, t].X
                if dp > 1e-6 or pm > 1e-6:
                    served.append((i, dp, pm))
            if served:
                print("  service amounts (q_plus=delivery, q_minus=pickup):")
                for i, dp, pm in served:
                    print(f"    i={fmt_node(i)}: q_plus={dp:.2f}, q_minus={pm:.2f}")

    print("\n======================================================\n")


# ============================================================
# 1) DATA LOADER: READ FROM instance_expanded.xlsx
# ============================================================

def load_irppd_from_excel(
    xlsx_path: str,
    depot_inventory_bigM: float = 1e9,
    depot_holding_cost: float = 0.0,
    print_demands: bool = True,
    max_rows: int = 50,
):
    expanded = pd.read_excel(xlsx_path, sheet_name="ExpandedData", engine="openpyxl")
    meta_df = pd.read_excel(xlsx_path, sheet_name="Meta", engine="openpyxl")
    dist = pd.read_excel(xlsx_path, sheet_name="DistanceMatrix", header=None, engine="openpyxl").values

    if meta_df.empty:
        raise ValueError("Meta sheet is empty.")
    meta = meta_df.iloc[0].to_dict()

    T_count = int(meta["T"])
    Q = float(meta["Q"])
    K_max = int(meta.get("K", len(expanded)))
    n_nodes = int(meta["n_nodes"])

    demand_cols = [f"d_{t}" for t in range(1, T_count + 1)]
    missing = [c for c in demand_cols if c not in expanded.columns]
    if missing:
        raise ValueError(f"ExpandedData is missing demand columns: {missing}")

    if dist.shape[0] != n_nodes or dist.shape[1] != n_nodes:
        raise ValueError(f"DistanceMatrix shape {dist.shape} does not match n_nodes={n_nodes}.")

    expanded = expanded.sort_values("Customer").reset_index(drop=True)
    orig_ids = expanded["Customer"].astype(int).tolist()
    n_customers = len(orig_ids)

    if n_customers != n_nodes - 1:
        raise ValueError(
            f"ExpandedData has {n_customers} customers but Meta says n_nodes={n_nodes} (expected {n_nodes-1} customers)."
        )

    depot = 0
    N = list(range(1, n_customers + 1))
    N_all = [depot] + N
    T = list(range(1, T_count + 1))

    internal_to_orig = {i: orig_ids[i - 1] for i in N}

    I0 = {depot: float(depot_inventory_bigM)}
    L = {depot: 0.0}
    U = {depot: float("inf")}
    h = {depot: float(depot_holding_cost)}
    r = {}

    for idx, row in expanded.iterrows():
        i = idx + 1
        r[i] = float(row["ri"])
        L[i] = float(row["LB"])
        U[i] = float(row["UB"])
        I0[i] = float(row["I0"])
        h[i] = float(row["h"]) if not pd.isna(row["h"]) else 0.0

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

    if print_demands:
        print("\n================= EXCEL DEMANDS (sample) =================")
        header = ["node(internal)", "orig_id", "ri"] + [f"d_{t}" for t in T]
        print(" | ".join(f"{h_:>12}" for h_ in header))
        print("-" * (15 * len(header)))
        for i in N[:max_rows]:
            row_vals = [i, internal_to_orig[i], int(round(r[i]))] + [d[(i, t)] for t in T]
            print(" | ".join(f"{x:>12}" for x in row_vals))
        print("==========================================================\n")

    return {
        "N": N,
        "T": T,
        "Q": Q,
        "m": K_max,
        "d": d,
        "h": h,
        "L": L,
        "U": U,
        "I0": I0,
        "c": c,
        "r": r,
        "internal_to_orig": internal_to_orig,
        "depot": depot,
    }


# ============================================================
# 2) IRP-PD MODEL – LAZY CONSTRAINT READY
#    Updated for per-period sign(d) => delivery/pickup/none
# ============================================================

def build_irp_pd_lazy_from_excel(data):
    N  = data["N"]
    T  = data["T"]
    Q  = data["Q"]
    m  = data["m"]

    d  = data["d"]
    h  = data["h"]
    L  = data["L"]
    U  = data["U"]
    I0 = data["I0"]
    c  = data["c"]

    depot = data.get("depot", 0)
    N_all = [depot] + list(N)
    A = [(i, j) for i in N_all for j in N_all if i != j]

    Kcap = {i: min(Q, U[i] - L[i]) for i in N}

    model = Model("IRP_PD_LAZY_EXCEL")

    T0 = [0] + T
    I = model.addVars(N_all, T0, vtype=GRB.CONTINUOUS, name="I")

    # split service
    q_plus  = model.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0.0, name="q_plus")   # delivery
    q_minus = model.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0.0, name="q_minus") # pickup

    y  = model.addVars(N, T, vtype=GRB.BINARY, name="y")
    y0 = model.addVars(T,   vtype=GRB.INTEGER, name="y0")

    x = model.addVars(A, T, vtype=GRB.BINARY, name="x")
    l = model.addVars(A, T, vtype=GRB.CONTINUOUS, name="l")

    # Objective
    model.setObjective(
        quicksum(c[i, j] * x[i, j, t] for (i, j) in A for t in T) +
        quicksum(h[i] * I[i, t]       for i in N_all for t in T),
        GRB.MINIMIZE
    )

    # (1b) depot outflow = y0_t
    for t in T:
        model.addConstr(quicksum(x[depot, j, t] for j in N) == y0[t], name=f"dep_out_{t}")

    # (1c) flow conservation
    for t in T:
        for i in N_all:
            model.addConstr(
                quicksum(x[i, j, t] for j in N_all if i != j) -
                quicksum(x[j, i, t] for j in N_all if i != j) == 0,
                name=f"flow_{i}_{t}"
            )

    # (1d) degree = y_it
    for t in T:
        for i in N:
            model.addConstr(quicksum(x[i, j, t] for j in N_all if i != j) == y[i, t], name=f"visit_{i}_{t}")

    # Customer inventory balance (works for any sign)
    # I[i,t] = I[i,t-1] + q_plus - q_minus - d
    for t in T:
        for i in N:
            model.addConstr(
                I[i, t] - I[i, t-1] - q_plus[i, t] + q_minus[i, t] + d[(i, t)] == 0,
                name=f"inv_{i}_{t}"
            )

    # Depot inventory balance
    for t in T:
        model.addConstr(
            I[depot, t] - I[depot, t-1] +
            quicksum(q_plus[i, t] for i in N) -
            quicksum(q_minus[i, t] for i in N) == 0,
            name=f"inv_depot_{t}"
        )

    # Initial inventories
    for i in N_all:
        model.addConstr(I[i, 0] == I0[i], name=f"init_inv_{i}")

    # Inventory bounds (customers)
    for t in T:
        for i in N:
            model.addConstr(I[i, t] >= L[i], name=f"lb_{i}_{t}")
            model.addConstr(I[i, t] <= U[i], name=f"ub_{i}_{t}")

    # Service capacity + link to visit
    for t in T:
        for i in N:
            model.addConstr(q_plus[i, t] <= Kcap[i] * y[i, t], name=f"serv_plus_{i}_{t}")
            model.addConstr(q_minus[i, t] <= Kcap[i] * y[i, t], name=f"serv_minus_{i}_{t}")

    # FIX: enforce type by sign of d(i,t)
    for t in T:
        for i in N:
            dit = d[(i, t)]
            if dit > 0:
                model.addConstr(q_minus[i, t] == 0.0, name=f"type_del_{i}_{t}")
            elif dit < 0:
                model.addConstr(q_plus[i, t] == 0.0, name=f"type_pick_{i}_{t}")
            else:
                model.addConstr(q_plus[i, t] == 0.0, name=f"type_zero_p_{i}_{t}")
                model.addConstr(q_minus[i, t] == 0.0, name=f"type_zero_m_{i}_{t}")

    # Fleet limit
    for t in T:
        model.addConstr(y0[t] <= m, name=f"fleet_cap_{t}")
        for i in N:
            model.addConstr(y[i, t] <= y0[t], name=f"link_y_y0_{i}_{t}")

    # Arc capacity
    for t in T:
        for (i, j) in A:
            model.addConstr(l[i, j, t] <= Q * x[i, j, t], name=f"cap_arc_{i}_{j}_{t}")

    # Load balance at customers:
    # sum_in l + q_minus - q_plus - sum_out l = 0
    for t in T:
        for i in N:
            model.addConstr(
                quicksum(l[j, i, t] for j in N_all if j != i) +
                q_minus[i, t] -
                q_plus[i, t] -
                quicksum(l[i, j, t] for j in N_all if j != i) == 0,
                name=f"load_{i}_{t}"
            )

    # Depot outload <= depot inventory at t-1
    for t in T:
        model.addConstr(
            quicksum(l[depot, j, t] for j in N) <= I[depot, t-1],
            name=f"depot_load_{t}"
        )

    # Save for callback + printing
    model._N = N
    model._N_all = N_all
    model._T = T
    model._A = A
    model._x = x
    model._y = y
    model._q_plus = q_plus
    model._q_minus = q_minus
    model._I = I
    model._internal_to_orig = data.get("internal_to_orig", {})

    model.Params.LazyConstraints = 1
    return model


# ============================================================
# 3) LAZY SUBTOUR CALLBACK – Archetti (1e)-style cut
# ============================================================

def irp_pd_subtour_callback(model, where):
    """
    Lazy cut:
      for any component S not containing depot:
        sum_{i in S} sum_{j in S} x_ijt <= sum_{i in S} y_it - y_mt
    """
    if where != GRB.Callback.MIPSOL:
        return

    N      = model._N
    N_all  = model._N_all
    T      = model._T
    A      = model._A
    x_var  = model._x
    y_var  = model._y

    x_val = model.cbGetSolution(x_var)
    y_val = model.cbGetSolution(y_var)

    depot = 0

    for t in T:
        visited = [i for i in N if y_val[i, t] > 0.5]
        if not visited:
            continue

        unvisited = set(visited)

        while unvisited:
            current = unvisited.pop()
            stack = [current]
            component = {current}

            while stack:
                i = stack.pop()
                neighbors = []
                for j in N_all:
                    if (i, j) in A and x_val[i, j, t] > 0.5:
                        neighbors.append(j)
                    if (j, i) in A and x_val[j, i, t] > 0.5:
                        neighbors.append(j)

                for j in neighbors:
                    if j in unvisited:
                        unvisited.remove(j)
                        stack.append(j)
                        component.add(j)

            if depot not in component and len(component) > 1:
                S = list(component)
                m_node = min(S)
                lhs = quicksum(model._x[i, j, t] for i in S for j in S if (i, j) in A)
                rhs = quicksum(model._y[i, t] for i in S) - model._y[m_node, t]
                model.cbLazy(lhs <= rhs)


# ============================================================
# 4) MAIN
# ============================================================

if __name__ == "__main__":

    XLSX_PATH = "instance_expanded.xlsx"

    data = load_irppd_from_excel(
        XLSX_PATH,
        depot_inventory_bigM=1e9,
        depot_holding_cost=0.0,
        print_demands=True
    )

    model = build_irp_pd_lazy_from_excel(data)

    model.Params.OutputFlag = 1
    model.Params.TimeLimit = 600

    model.optimize(irp_pd_subtour_callback)

    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal value (IRP-PD + Lazy, Excel input): {model.ObjVal}\n")
        print("Runtime (sec):", model.Runtime)
        print("BB nodes    :", model.NodeCount)
        print_solution_routes(model, depot=0, eps=0.5, show_q=True)
    else:
        print("\nModel optimal çözülmedi, status:", model.status)
        if model.SolCount > 0:
            print("Best incumbent:", model.ObjVal)
            print_solution_routes(model, depot=0, eps=0.5, show_q=True)
