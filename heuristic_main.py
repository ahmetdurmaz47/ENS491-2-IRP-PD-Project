import re
from typing import Dict, List, Tuple

import pandas as pd


def load_instance(xlsx_path: str):
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

    expanded = expanded.sort_values("Customer").reset_index(drop=True)
    n_customers = len(expanded)
    if n_customers != n_nodes - 1:
        raise ValueError(f"ExpandedData has {n_customers} customers but Meta says n_nodes={n_nodes}.")

    depot = 0
    T = list(range(1, T_count + 1))
    N = list(range(1, n_customers + 1))

    internal_to_orig = {i: int(expanded.loc[i - 1, "Customer"]) for i in N}

    I0 = {depot: 10**9}
    L = {depot: 0}
    U = {depot: 10**9}
    h = {depot: 0.0}
    d = {}

    for idx, row in expanded.iterrows():
        i = idx + 1
        I0[i] = int(row["I0"])
        L[i] = int(row["LB"])
        U[i] = int(row["UB"])
        h[i] = float(row["h"]) if not pd.isna(row["h"]) else 0.0
        for t in T:
            d[(i, t)] = int(row[f"d_{t}"])

    return {
        "expanded": expanded,
        "meta": meta,
        "dist": dist,
        "T": T,
        "N": N,
        "Q": Q,
        "m": K_max,
        "depot": depot,
        "I0": I0,
        "L": L,
        "U": U,
        "h": h,
        "d": d,
        "internal_to_orig": internal_to_orig,
    }


def parse_mip_output(txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    best_obj = None
    m = re.search(r"Best incumbent:\s*([0-9eE+\-.]+)", txt)
    if m:
        best_obj = float(m.group(1))

    periods = {}
    cur_t = None
    for raw_line in txt.splitlines():
        line = raw_line.strip()

        m = re.match(r"Period t=(\d+) \| #routes=(\d+)", line)
        if m:
            cur_t = int(m.group(1))
            periods[cur_t] = {"routes": [], "service": {}}
            continue

        m = re.match(r"Route \d+: (.*)", line)
        if m and cur_t is not None:
            route = [int(x.split("(")[0].strip()) for x in m.group(1).split("->")]
            periods[cur_t]["routes"].append(route)
            continue

        m = re.match(r"i=(\d+)(?:\(orig=.*\))?: q_plus=([0-9.]+), q_minus=([0-9.]+)", line)
        if m and cur_t is not None:
            i = int(m.group(1))
            q_plus = float(m.group(2))
            q_minus = float(m.group(3))
            periods[cur_t]["service"][i] = q_plus - q_minus

    return best_obj, periods


def route_cost(route: List[int], dist) -> float:
    return sum(float(dist[route[k], route[k + 1]]) for k in range(len(route) - 1))


def inventory_path(i: int, q: Dict[Tuple[int, int], int], data) -> List[int]:
    cur = data["I0"][i]
    path = []
    for t in data["T"]:
        cur = cur + q[(i, t)] - data["d"][(i, t)]
        path.append(cur)
    return path


def check_customer_feasibility(i: int, q: Dict[Tuple[int, int], int], data) -> bool:
    inv = inventory_path(i, q, data)
    for idx, t in enumerate(data["T"]):
        if inv[idx] < data["L"][i] - 1e-9 or inv[idx] > data["U"][i] + 1e-9:
            return False
        qt = q[(i, t)]
        dt = data["d"][(i, t)]
        if dt > 0 and qt < -1e-9:
            return False
        if dt < 0 and qt > 1e-9:
            return False
        if dt == 0 and abs(qt) > 1e-9:
            return False
    return True


def feasible_initial_interval(route: List[int], q_t: Dict[int, int], Q: float):
    prefix = 0.0
    min_pref = 0.0
    max_pref = 0.0
    for node in route[1:-1]:
        delta = -q_t.get(node, 0.0)
        prefix += delta
        min_pref = min(min_pref, prefix)
        max_pref = max(max_pref, prefix)
    low = max(0.0, -min_pref)
    high = min(Q, Q - max_pref)
    return low, high


def locate_customer(routes: List[List[int]], customer: int):
    hits = []
    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route) - 1):
            if route[pos] == customer:
                hits.append((r_idx, pos))
    return hits


def purge_customer_from_routes(routes: List[List[int]], customer: int) -> List[List[int]]:
    new_routes = []
    for route in routes:
        cleaned = [route[0]] + [node for node in route[1:-1] if node != customer] + [route[-1]]
        if len(cleaned) > 2:
            new_routes.append(cleaned)
    return new_routes


def route_structure_ok(route: List[int], data) -> bool:
    if len(route) < 2 or route[0] != 0 or route[-1] != 0:
        return False
    seen = set()
    prev = None
    for node in route[1:-1]:
        if node == 0:
            return False
        if node == prev:
            return False
        if node in seen:
            return False
        if node not in data["N"]:
            return False
        seen.add(node)
        prev = node
    return True


def sanitize_routes_by_period(routes_by_t, q, data):
    clean = {t: [] for t in data["T"]}
    for t in data["T"]:
        required = {i for i in data["N"] if abs(q[(i, t)]) > 1e-9}
        assigned = set()

        for route in routes_by_t[t]:
            new_route = [0]
            seen = set()
            for node in route[1:-1]:
                if node not in required:
                    continue
                if node in seen or node in assigned:
                    continue
                new_route.append(node)
                seen.add(node)
                assigned.add(node)
            new_route.append(0)
            if len(new_route) > 2:
                clean[t].append(new_route)

        missing = sorted(required - assigned)
        for node in missing:
            clean[t].append([0, node, 0])

    return clean


def check_route_customer_consistency(routes_by_t, q, data) -> bool:
    for t in data["T"]:
        served = {i for i in data["N"] if abs(q[(i, t)]) > 1e-9}
        counted = []
        for route in routes_by_t[t]:
            if not route_structure_ok(route, data):
                return False
            counted.extend(route[1:-1])
        if len(counted) != len(set(counted)):
            return False
        if set(counted) != served:
            return False
    return True


def check_routes_feasible(routes_by_t, q, data) -> bool:
    if not check_route_customer_consistency(routes_by_t, q, data):
        return False
    for t in data["T"]:
        q_t = {i: q[(i, t)] for i in data["N"] if abs(q[(i, t)]) > 1e-9}
        for route in routes_by_t[t]:
            low, high = feasible_initial_interval(route, q_t, data["Q"])
            if low > high + 1e-9:
                return False
    return True


def holding_cost(q, data) -> float:
    total = 0.0
    for i in data["N"]:
        total += sum(data["h"][i] * x for x in inventory_path(i, q, data))
    return total


def travel_cost(routes_by_t, data) -> float:
    total = 0.0
    for t in data["T"]:
        for route in routes_by_t[t]:
            total += route_cost(route, data["dist"])
    return total


def total_objective(q, routes_by_t, data) -> float:
    return holding_cost(q, data) + travel_cost(routes_by_t, data)


def build_solution_from_output(periods, data):
    q = {(i, t): 0 for i in data["N"] for t in data["T"]}
    routes_by_t = {t: [] for t in data["T"]}

    for t in data["T"]:
        if t in periods:
            for route in periods[t]["routes"]:
                routes_by_t[t].append(route[:])
            for i, val in periods[t]["service"].items():
                q[(i, t)] = int(round(val))

    routes_by_t = sanitize_routes_by_period(routes_by_t, q, data)

    for i in data["N"]:
        if not check_customer_feasibility(i, q, data):
            raise ValueError(f"Incumbent schedule infeasible for customer {i}.")
    if not check_routes_feasible(routes_by_t, q, data):
        raise ValueError("Incumbent route load sequence or visit structure infeasible.")

    return q, routes_by_t


def removal_delta_for_period(routes: List[List[int]], customer: int, dist):
    hits = locate_customer(routes, customer)
    if len(hits) != 1:
        return None
    r_idx, pos = hits[0]
    route = routes[r_idx]
    a, b, c = route[pos - 1], route[pos], route[pos + 1]
    delta = float(dist[a, c] - dist[a, b] - dist[b, c])
    return delta, r_idx


def insertion_delta_for_period(routes: List[List[int]], customer: int, q, t: int, data):
    dist = data["dist"]
    best = None

    for r_idx, route in enumerate(routes):
        if customer in route[1:-1]:
            continue
        for pos in range(len(route) - 1):
            a, b = route[pos], route[pos + 1]
            delta = float(dist[a, customer] + dist[customer, b] - dist[a, b])
            new_route = route[:pos + 1] + [customer] + route[pos + 1:]
            q_t = {i: q[(i, t)] for i in data["N"] if abs(q[(i, t)]) > 1e-9}
            low, high = feasible_initial_interval(new_route, q_t, data["Q"])
            if low <= high + 1e-9:
                move = (delta, r_idx, pos + 1, False)
                if best is None or move[0] < best[0]:
                    best = move

    if len(routes) < data["m"]:
        delta = float(dist[0, customer] + dist[customer, 0])
        move = (delta, -1, -1, True)
        if best is None or move[0] < best[0]:
            best = move

    return best


def best_schedule_for_customer(i: int, q_cur, routes_cur, data):
    old_q = {t: q_cur[(i, t)] for t in data["T"]}
    old_hold = sum(data["h"][i] * x for x in inventory_path(i, q_cur, data))

    route_effect = {}
    for t in data["T"]:
        old_vis = abs(old_q[t]) > 1e-9
        rem = removal_delta_for_period(routes_cur[t], i, data["dist"]) if old_vis else None
        ins = insertion_delta_for_period(routes_cur[t], i, q_cur, t, data) if not old_vis else None
        route_effect[t] = {"old_vis": old_vis, "remove": rem, "insert": ins}

        if old_vis and rem is None:
            return None

    states = {data["I0"][i]: (0.0, [])}

    for t in data["T"]:
        dt = data["d"][(i, t)]
        new_states = {}
        for prev_I, (acc_cost, path) in states.items():
            for I_t in range(data["L"][i], data["U"][i] + 1):
                q_t = I_t - prev_I + dt

                if dt > 0 and q_t < -1e-9:
                    continue
                if dt < 0 and q_t > 1e-9:
                    continue
                if dt == 0 and abs(q_t) > 1e-9:
                    continue

                q_t = int(round(q_t))
                new_vis = abs(q_t) > 1e-9
                old_vis = route_effect[t]["old_vis"]

                route_delta = 0.0
                if old_vis and not new_vis:
                    route_delta += route_effect[t]["remove"][0]
                elif (not old_vis) and new_vis:
                    ins = route_effect[t]["insert"]
                    if ins is None:
                        continue
                    route_delta += ins[0]

                total = acc_cost + data["h"][i] * I_t + route_delta
                prev_best = new_states.get(I_t)
                if prev_best is None or total < prev_best[0] - 1e-9:
                    new_states[I_t] = (total, path + [q_t])

        states = new_states
        if not states:
            return None

    _, (best_cost, best_path) = min(states.items(), key=lambda kv: kv[1][0])
    new_sched = {t: best_path[idx] for idx, t in enumerate(data["T"])}
    approx_delta = best_cost - old_hold
    return approx_delta, new_sched


def apply_customer_schedule(i: int, new_sched, q_cur, routes_cur, data):
    q_new = dict(q_cur)
    routes_new = {t: [r[:] for r in routes_cur[t]] for t in data["T"]}

    for t in data["T"]:
        q_new[(i, t)] = int(new_sched[t])

    for t in data["T"]:
        old_vis = abs(q_cur[(i, t)]) > 1e-9
        new_vis = abs(new_sched[t]) > 1e-9

        routes_new[t] = purge_customer_from_routes(routes_new[t], i)

        if new_vis:
            best = insertion_delta_for_period(routes_new[t], i, q_new, t, data)
            if best is None:
                return None
            _, r_idx, pos, create_new = best
            if create_new:
                routes_new[t].append([0, i, 0])
            else:
                routes_new[t][r_idx].insert(pos, i)

    routes_new = sanitize_routes_by_period(routes_new, q_new, data)

    if not check_customer_feasibility(i, q_new, data):
        return None
    if not check_routes_feasible(routes_new, q_new, data):
        return None

    return q_new, routes_new


def two_opt_route(route: List[int], q_t: Dict[int, int], data) -> List[int]:
    best = route[:]
    best_cost = route_cost(best, data["dist"])
    improved = True

    while improved:
        improved = False
        m = len(best)
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not route_structure_ok(cand, data):
                    continue
                low, high = feasible_initial_interval(cand, q_t, data["Q"])
                if low > high + 1e-9:
                    continue
                cand_cost = route_cost(cand, data["dist"])
                if cand_cost < best_cost - 1e-9:
                    best = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break

    return best


def improve_all_routes_2opt(routes_cur, q_cur, data):
    routes_new = {t: [r[:] for r in routes_cur[t]] for t in data["T"]}
    for t in data["T"]:
        q_t = {i: q_cur[(i, t)] for i in data["N"] if abs(q_cur[(i, t)]) > 1e-9}
        for ridx in range(len(routes_new[t])):
            routes_new[t][ridx] = two_opt_route(routes_new[t][ridx], q_t, data)
    routes_new = sanitize_routes_by_period(routes_new, q_cur, data)
    return routes_new


def improvement_heuristic(q_start, routes_start, data, max_iters: int = 200, verbose: bool = True):
    q_cur = dict(q_start)
    routes_cur = sanitize_routes_by_period({t: [r[:] for r in routes_start[t]] for t in data["T"]}, q_start, data)
    best_obj = total_objective(q_cur, routes_cur, data)

    if verbose:
        print(f"Initial objective = {best_obj:.6f}")

    iteration = 0
    while iteration < max_iters:
        best_move = None

        for i in data["N"]:
            candidate = best_schedule_for_customer(i, q_cur, routes_cur, data)
            if candidate is None:
                continue
            _, new_sched = candidate

            if all(int(new_sched[t]) == int(q_cur[(i, t)]) for t in data["T"]):
                continue

            applied = apply_customer_schedule(i, new_sched, q_cur, routes_cur, data)
            if applied is None:
                continue

            q_tmp, routes_tmp = applied
            routes_tmp = improve_all_routes_2opt(routes_tmp, q_tmp, data)

            if not check_routes_feasible(routes_tmp, q_tmp, data):
                continue

            obj_tmp = total_objective(q_tmp, routes_tmp, data)
            real_delta = obj_tmp - best_obj

            if real_delta < -1e-9:
                if best_move is None or real_delta < best_move[0]:
                    best_move = (real_delta, i, new_sched, q_tmp, routes_tmp, obj_tmp)

        if best_move is None:
            break

        _, i_star, _, q_cur, routes_cur, best_obj = best_move
        iteration += 1
        if verbose:
            print(f"iter {iteration:3d} | customer {i_star:3d} accepted | new obj = {best_obj:.6f}")

    return q_cur, routes_cur, best_obj


def print_solution(q, routes_by_t, data, title: str = "Improved solution"):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Travel cost : {travel_cost(routes_by_t, data):.6f}")
    print(f"Holding cost: {holding_cost(q, data):.6f}")
    print(f"Total obj   : {total_objective(q, routes_by_t, data):.6f}")

    for t in data["T"]:
        print(f"\nPeriod t={t} | #routes={len(routes_by_t[t])}")
        for k, route in enumerate(routes_by_t[t], start=1):
            print(f"  Route {k}: " + " -> ".join(map(str, route)))
        print("  service amounts:")
        for i in data["N"]:
            qt = q[(i, t)]
            if abs(qt) > 1e-9:
                if qt > 0:
                    print(f"    i={i}: q_plus={qt:.0f}, q_minus=0")
                else:
                    print(f"    i={i}: q_plus=0, q_minus={-qt:.0f}")


if __name__ == "__main__":
    XLSX_PATH = "instance_expanded.xlsx"
    INCUMBENT_TXT = "mip_solution.txt"  # MTZ output. Change to Lazy txt if needed.

    data = load_instance(XLSX_PATH)
    incumbent_obj_from_txt, periods = parse_mip_output(INCUMBENT_TXT)
    q0, routes0 = build_solution_from_output(periods, data)

    exact_initial_obj = total_objective(q0, routes0, data)
    print(f"Incumbent objective from txt : {incumbent_obj_from_txt}")
    print(f"Recomputed incumbent objective: {exact_initial_obj:.6f}")

    q_best, routes_best, obj_best = improvement_heuristic(q0, routes0, data, max_iters=200, verbose=True)

    print_solution(q_best, routes_best, data, title="Improved solution")
    print(f"Improvement = {exact_initial_obj - obj_best:.6f}")
