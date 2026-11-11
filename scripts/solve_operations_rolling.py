"""
Step 2 of the LWT resilience study: operate a system designed with perfect foresight,
using rolling horizon (limited foresight), and measure unserved energy.

Usage (from the pypsa-eur folder, inside `pixi shell`):
    python scripts/solve_operations_rolling.py <solved_network.nc> <output_op.nc> [horizon_snapshots] [overlap_snapshots]

Example (3h resolution, 2 weeks of foresight = 14*8 = 112 snapshots):
    python scripts/solve_operations_rolling.py \
        results/lwt_resilience/with_lwt/networks/base_s_40__Co2L0-3h-lv1.25-wcf0.7_2050.nc \
        results/lwt_resilience/with_lwt/networks/base_s_40__Co2L0-3h-lv1.25-wcf0.7_2050_op.nc \
        112
"""

import sys

import pypsa

VOLL = 100_000  # value of lost load, EUR/MWh

network_in = sys.argv[1]
network_out = sys.argv[2]
horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 112
# Overlap: the last `overlap` snapshots of each window are only a look-ahead and get
# re-solved in the next window. Each window commits (horizon - overlap) snapshots.
# Example at 24h: horizon=21, overlap=7 -> commit 2 weeks, with 1 extra week of look-ahead.
overlap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
assert 0 <= overlap < horizon, "overlap must be smaller than horizon"

n = pypsa.Network(network_in)

# 1. Freeze the capacities chosen in the design run.
n.optimize.fix_optimal_capacities()

# 2. Remove investment-related global constraints (nothing can be built any more).
#    Keep CO2 limits, which still constrain operation.
investment_constraints = n.global_constraints.index[
    n.global_constraints.type.str.contains("expansion|volume|cost", case=False)
]
n.remove("GlobalConstraint", investment_constraints)

# 3. Storage: no cyclic condition in operation. Start from the state at the end of
#    the design run, so the first window doesn't start with empty storage.
if not n.stores.empty:
    n.stores["e_initial"] = n.stores_t.e.iloc[-1]
    n.stores["e_cyclic"] = False
if not n.storage_units.empty:
    n.storage_units["state_of_charge_initial"] = n.storage_units_t.state_of_charge.iloc[-1]
    n.storage_units["cyclic_state_of_charge"] = False

# 4. Load shedding on every bus with a load, so shortages show up as unserved energy
#    instead of making the problem infeasible.
load_buses = n.loads.bus.unique()
if "load" not in n.carriers.index:
    n.add("Carrier", "load")
n.add(
    "Generator",
    load_buses,
    suffix=" load shedding",
    bus=load_buses,
    carrier="load",
    p_nom=1e6,
    marginal_cost=VOLL,
)

# 5. Rolling-horizon operation.
n.optimize.optimize_with_rolling_horizon(
    horizon=horizon,
    overlap=overlap,
    solver_name="gurobi",
    solver_options={"threads": 32, "method": 2, "crossover": 0, "BarConvTol": 1e-5},
)

n.export_to_netcdf(network_out)

# 6. Quick summary.
shed = n.generators_t.p.filter(like="load shedding")
weights = n.snapshot_weightings.generators
unserved_mwh = shed.multiply(weights, axis=0).sum().sum()
hours_with_shedding = weights[(shed.sum(axis=1) > 1e-3)].sum()
print(f"Unserved energy: {unserved_mwh / 1e6:.3f} TWh")
print(f"Hours with unserved energy: {hours_with_shedding:.0f} h")
