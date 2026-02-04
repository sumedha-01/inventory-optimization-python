# inventory_optimization_demo.py
"""
Inventory Optimization Demo (Python)
-----------------------------------
A simple, self-contained Python script that:
  1) Simulates daily demand for a few SKUs
  2) Forecasts demand (moving average + simple exponential smoothing)
  3) Computes EOQ, Safety Stock, and Reorder Point per SKU
  4) Runs a basic Monte Carlo stockout simulation to assess service level
  5) Plots demand and inventory position over time

No external data files or non-standard packages required.

How to run:
  $ python3 inventory_optimization_demo.py

What this demonstrates (resume-ready):
  - Demand forecasting (basic time-series smoothing)
  - Inventory control policies (EOQ, safety stock, reorder point)
  - Service level simulation under demand and lead time variability
  - Data analysis and plotting with pandas/matplotlib

Author: You (replace with your name)
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Configuration ---------- #
np.random.seed(42)
random.seed(42)

DAYS = 180                     # length of simulation
LEAD_TIME_MEAN = 7            # days
LEAD_TIME_STD = 2             # days (variability in supplier lead time)
SERVICE_LEVEL_Z = 1.65        # ~95% cycle service level
HOLDING_COST_RATE = 0.2       # 20% of unit cost per year (approx)
ORDER_COST = 50.0             # $/order (fixed ordering cost)
WORKING_DAYS_PER_YEAR = 365   # for EOQ annualization

SKUS = [
    {"sku": "A100", "unit_cost": 15.0, "baseline_daily_mean": 20, "baseline_daily_std": 5},
    {"sku": "B200", "unit_cost": 8.0,  "baseline_daily_mean": 35, "baseline_daily_std": 8},
    {"sku": "C300", "unit_cost": 25.0, "baseline_daily_mean": 12, "baseline_daily_std": 3},
]

# ---------- Helper functions ---------- #

def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def simple_exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    ses = []
    prev = series.iloc[0]
    for x in series:
        prev = alpha * x + (1 - alpha) * prev
        ses.append(prev)
    return pd.Series(ses, index=series.index)


def eoq(annual_demand_units: float, order_cost: float, holding_cost_per_unit_per_year: float) -> float:
    if annual_demand_units <= 0 or holding_cost_per_unit_per_year <= 0:
        return 0.0
    return math.sqrt((2 * annual_demand_units * order_cost) / holding_cost_per_unit_per_year)


def safety_stock(daily_demand_std: float, lead_time_days: float, z: float) -> float:
    # Assuming independent daily demand; std over lead time ~ sqrt(lead_time_days) * daily std
    return z * daily_demand_std * math.sqrt(max(lead_time_days, 0))


def reorder_point(avg_daily_demand: float, avg_lead_time_days: float, safety_stock_units: float) -> float:
    return avg_daily_demand * max(avg_lead_time_days, 0) + safety_stock_units


@dataclass
class Policy:
    sku: str
    eoq_qty: float
    safety_stock: float
    reorder_point: float
    avg_daily_demand: float
    demand_std: float


# ---------- 1) Simulate historical daily demand ---------- #

def simulate_demand(days: int, mean: float, std: float) -> np.ndarray:
    # Ensure demand is non-negative integers; use normal + clamp at 0
    raw = np.random.normal(loc=mean, scale=std, size=days)
    return np.clip(np.round(raw), 0, None).astype(int)


def build_history_df() -> pd.DataFrame:
    records = []
    for sku_cfg in SKUS:
        demand = simulate_demand(DAYS, sku_cfg["baseline_daily_mean"], sku_cfg["baseline_daily_std"])
        for t, qty in enumerate(demand, start=1):
            records.append({
                "day": t,
                "sku": sku_cfg["sku"],
                "demand": qty,
                "unit_cost": sku_cfg["unit_cost"],
            })
    df = pd.DataFrame.from_records(records)
    return df


# ---------- 2) Forecast + 3) Compute EOQ / SS / ROP ---------- #

def design_policies(history: pd.DataFrame) -> Dict[str, Policy]:
    policies: Dict[str, Policy] = {}

    for sku, g in history.groupby("sku"):
        g = g.sort_values("day")
        demand_series = g["demand"].astype(float)

        # Forecast next-day demand using 7-day MA + SES ensemble (simple average)
        ma_forecast = moving_average(demand_series, window=7)
        ses_forecast = simple_exponential_smoothing(demand_series, alpha=0.3)
        combined_forecast = 0.5 * ma_forecast + 0.5 * ses_forecast

        avg_daily_demand = float(combined_forecast.iloc[-1])
        demand_std = float(demand_series.std(ddof=1)) if len(demand_series) > 1 else 0.0

        unit_cost = float(g["unit_cost"].iloc[0])
        annual_demand = avg_daily_demand * WORKING_DAYS_PER_YEAR
        holding_cost_per_unit_per_year = HOLDING_COST_RATE * unit_cost

        eoq_qty = eoq(annual_demand, ORDER_COST, holding_cost_per_unit_per_year)
        ss = safety_stock(demand_std, LEAD_TIME_MEAN, SERVICE_LEVEL_Z)
        rop = reorder_point(avg_daily_demand, LEAD_TIME_MEAN, ss)

        policies[sku] = Policy(
            sku=sku,
            eoq_qty=eoq_qty,
            safety_stock=ss,
            reorder_point=rop,
            avg_daily_demand=avg_daily_demand,
            demand_std=demand_std,
        )

    return policies


# ---------- 4) Monte Carlo simulation of inventory position & service level ---------- #

def simulate_inventory(sku: str, history: pd.DataFrame, policy: Policy, horizon_days: int = 60):
    """
    Simulate future horizon using policy (Q, ROP) with stochastic demand and lead time.
    - When inventory position <= ROP, place an order of size EOQ.
    - Lead time is Normal with mean/std; negative values clipped to 0.
    - Track fill rate (1 - backordered units / total demand).
    """
    # Initialize with on-hand equal to 1.5x ROP to avoid immediate stockout
    on_hand = max(int(round(1.5 * policy.reorder_point)), 0)
    pipeline: List[Tuple[int, int]] = []  # (arrival_day, qty)

    rows = []
    total_demand = 0
    total_backordered = 0

    # Use last known demand stats for sampling
    # Sample daily demand ~ Normal(avg, std); clamp at 0 and round
    mu = policy.avg_daily_demand
    sigma = max(policy.demand_std, 1.0)

    for day in range(1, horizon_days + 1):
        # Receive arrivals
        arrivals_today = 0
        still_pipeline = []
        for arr_day, qty in pipeline:
            if arr_day == day:
                arrivals_today += qty
            else:
                still_pipeline.append((arr_day, qty))
        pipeline = still_pipeline
        on_hand += arrivals_today

        # Demand realization
        d = max(int(round(np.random.normal(mu, sigma))), 0)
        total_demand += d

        # Fulfill demand
        shipped = min(on_hand, d)
        backordered = d - shipped
        on_hand -= shipped
        total_backordered += backordered

        # Check reorder (inventory position = on-hand + pipeline - backorders)
        pipeline_qty = sum(q for _, q in pipeline)
        inv_position = on_hand + pipeline_qty - 0  # assuming backorders not kept as negative IP here
        if inv_position <= policy.reorder_point:
            lot = int(max(round(policy.eoq_qty), 1))
            lt = max(int(round(np.random.normal(LEAD_TIME_MEAN, LEAD_TIME_STD))), 0)
            arrival = day + lt
            pipeline.append((arrival, lot))

        rows.append({
            "day": day,
            "sku": sku,
            "demand": d,
            "shipped": shipped,
            "backordered": backordered,
            "on_hand": on_hand,
            "arrivals": arrivals_today,
            "pipeline": sum(q for _, q in pipeline),
            "inv_position": inv_position,
        })

    df = pd.DataFrame(rows)
    fill_rate = 1.0 - (total_backordered / total_demand if total_demand > 0 else 0.0)
    return df, fill_rate


# ---------- 5) Visualization ---------- #

def plot_history_and_policy(history: pd.DataFrame, policies: Dict[str, Policy]):
    n = len(policies)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (sku, g) in zip(axes, history.groupby("sku")):
        g = g.sort_values("day")
        ax.plot(g["day"], g["demand"], label=f"Demand ({sku})", alpha=0.7)
        pol = policies[sku]
        ax.axhline(pol.avg_daily_demand, color="tab:green", linestyle="--", label="Avg Daily Demand (forecast)")
        ax.set_title(f"SKU {sku} - Historical Demand & Forecast")
        ax.set_ylabel("Units/day")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    plt.show()


def plot_inventory_sim(sim_df: pd.DataFrame, policy: Policy):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(sim_df["day"], sim_df["on_hand"], label="On-hand", color="tab:blue")
    ax1.plot(sim_df["day"], sim_df["inv_position"], label="Inv Position", color="tab:orange")
    ax1.axhline(policy.reorder_point, color="tab:red", linestyle=":", label="Reorder Point")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Units")

    ax2 = ax1.twinx()
    ax2.bar(sim_df["day"], sim_df["demand"], alpha=0.2, color="tab:gray", label="Demand")
    ax2.set_ylabel("Demand")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"SKU {policy.sku} - Inventory Simulation")
    plt.tight_layout()
    plt.show()


# ---------- Main ---------- #

def main():
    print("Simulating historical demand...")
    history = build_history_df()

    print("Designing inventory policies (EOQ / Safety Stock / ROP)...")
    policies = design_policies(history)

    # Show policy table
    pol_rows = []
    for p in policies.values():
        pol_rows.append({
            "sku": p.sku,
            "avg_daily_demand": round(p.avg_daily_demand, 2),
            "demand_std": round(p.demand_std, 2),
            "EOQ": int(round(p.eoq_qty)),
            "SafetyStock": int(round(p.safety_stock)),
            "ReorderPoint": int(round(p.reorder_point)),
        })
    pol_df = pd.DataFrame(pol_rows).set_index("sku")
    print("\nDesigned Policies:\n", pol_df)

    # Visualize history and forecast level
    plot_history_and_policy(history, policies)

    # Simulate one SKU to illustrate
    sku_to_demo = list(policies.keys())[0]
    sim_df, fill_rate = simulate_inventory(sku_to_demo, history, policies[sku_to_demo], horizon_days=90)
    print(f"\nSimulation Fill Rate for {sku_to_demo}: {fill_rate:.2%}")

    plot_inventory_sim(sim_df, policies[sku_to_demo])


if __name__ == "__main__":
    main()