import os
import re
import pandas as pd
import numpy as np

# ============================================================
# DATA GENERATION (ONLY d_it) for your pipeline
# Input : base_instance.dat  (e.g., 3abs10.dat)
# Output: instance_expanded.xlsx  (ExpandedData + Meta + DistanceMatrix)
#
# What this version does (per your request):
# - Each customer can be pickup OR delivery OR zero in EACH period (fully random).
# - Demand is generated as:
#       d_it = round(ri_i * u_it)
#   where u_it ~ Uniform(low, high) is CONTINUOUS (not integer).
#   Default: low=-2.0, high=2.0
# - K is inferred from filename prefix before "abs" (e.g., 3abs10.dat -> K=3)
# ============================================================


# ----------------------------
# Infer K from filename
# ----------------------------
def infer_K_from_filename(dat_path: str, default_K: int = 1) -> int:
    fname = os.path.basename(dat_path)
    m = re.search(r"(\d+)\s*abs", fname, flags=re.IGNORECASE)
    return int(m.group(1)) if m else default_K


# ----------------------------
# Read .dat (your instance format)
# Assumption:
#   first line = n_nodes  T  Q
#   next n_nodes lines:
#       depot (id=1): id x y [extra fields ignored]
#       customers (id>=2): id x y  ri UB LB I0 h   (h optional)
# ----------------------------
def read_instance_dat(dat_path: str):
    with open(dat_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    first = lines[0].split()
    if len(first) < 3:
        raise ValueError("First line must be: n_nodes T Q (e.g., '11 6 712').")

    n_nodes = int(float(first[0]))
    T = int(float(first[1]))      # period count
    Q = float(first[2])

    node_lines = lines[1:1 + n_nodes]
    if len(node_lines) != n_nodes:
        raise ValueError(f"Expected {n_nodes} node lines after header, got {len(node_lines)}.")

    records = []
    depot = None

    for ln in node_lines:
        parts = ln.split()
        if len(parts) < 3:
            raise ValueError(f"Bad node line (needs at least id x y): {ln}")

        node_id = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        extras = parts[3:]  # may contain depot/customer extra fields

        if node_id == 1:
            # accept any extra fields but only store coords + id
            depot = {"DepotID": node_id, "DepotX": x, "DepotY": y}
            continue

        # expected at least: ri UB LB I0  (h optional)
        if len(extras) < 4:
            raise ValueError(
                f"Customer line for node {node_id} has too few fields after x,y. "
                f"Expected at least 4 (ri UB LB I0). Got {len(extras)}. Line: {ln}"
            )

        ri = float(extras[0])
        UB = float(extras[1])
        LB = float(extras[2])
        I0 = float(extras[3])
        h = float(extras[4]) if len(extras) >= 5 else np.nan

        if LB > UB:
            raise ValueError(f"LB>UB for customer {node_id} (LB={LB}, UB={UB}).")

        records.append({
            "Customer": node_id,
            "X": x,
            "Y": y,
            "ri": ri,
            "LB": LB,
            "UB": UB,
            "I0": I0,
            "h": h
        })

    if depot is None:
        # fallback if depot missing
        depot = {"DepotID": 1, "DepotX": 0.0, "DepotY": 0.0}

    base_df = pd.DataFrame(records).sort_values("Customer").reset_index(drop=True)
    meta = {"n_nodes": n_nodes, "T": T, "Q": Q, **depot}
    return base_df, meta


# ----------------------------
# Generate ONLY d_1..d_T using CONTINUOUS multipliers
# d_it = round(ri_i * U(low, high))
# - fully random each period (pickup/delivery/zero possible)
# ----------------------------
def generate_only_d_continuous(
    base_df: pd.DataFrame,
    T: int,
    seed: int = 42,
    low: float = -2.0,
    high: float = 2.0
):
    rng = np.random.default_rng(seed)
    df = base_df.copy()
    ri_vals = df["ri"].to_numpy(dtype=float)

    for t in range(1, T + 1):
        mult = rng.uniform(low, high, size=len(df))  # CONTINUOUS
        df[f"d_{t}"] = np.rint(ri_vals * mult).astype(int)

    return df


# ----------------------------
# Distance matrix (depot + customers) from X,Y
# Node order in matrix:
#   0: depot
#   1..n_customers: customers in df order
# ----------------------------
def build_distance_matrix(meta: dict, df: pd.DataFrame):
    depx = float(meta.get("DepotX", 0.0))
    depy = float(meta.get("DepotY", 0.0))
    X_all = np.concatenate([[depx], df["X"].to_numpy(dtype=float)])
    Y_all = np.concatenate([[depy], df["Y"].to_numpy(dtype=float)])
    n_all = len(X_all)

    dist = np.zeros((n_all, n_all), dtype=float)
    for i in range(n_all):
        for j in range(n_all):
            dist[i, j] = float(np.hypot(X_all[i] - X_all[j], Y_all[i] - Y_all[j]))

    return pd.DataFrame(dist)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DAT_INPUT_PATH = "5abs50.dat"
    OUT_EXCEL_PATH = "instance_expanded.xlsx"
    SEED = 42

    # Continuous multiplier range (you wanted [-2,2] continuous)
    MULT_LOW = -2.0
    MULT_HIGH = 2.0

    base_df, meta = read_instance_dat(DAT_INPUT_PATH)

    # infer K from filename "<K>abs..."
    K = infer_K_from_filename(DAT_INPUT_PATH, default_K=1)
    meta["K"] = int(K)

    T = int(meta["T"])
    expanded_df = generate_only_d_continuous(
        base_df,
        T=T,
        seed=SEED,
        low=MULT_LOW,
        high=MULT_HIGH
    )

    dist_df = build_distance_matrix(meta, expanded_df)

    with pd.ExcelWriter(OUT_EXCEL_PATH, engine="openpyxl") as writer:
        expanded_df.to_excel(writer, sheet_name="ExpandedData", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="Meta", index=False)
        dist_df.to_excel(writer, sheet_name="DistanceMatrix", index=False, header=False)

    print("DONE")
    print(f"Input : {DAT_INPUT_PATH}")
    print(f"Output: {OUT_EXCEL_PATH}")
    print(f"Inferred K from filename: {K}")
    print(f"Periods (T): {T} -> generated d_1..d_{T}")
    print(f"Multiplier: U({MULT_LOW}, {MULT_HIGH}) continuous")
    print("Sheets: ExpandedData, Meta, DistanceMatrix")
