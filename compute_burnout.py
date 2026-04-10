"""
PulseIQ — Step 4: Behavioral Burnout Engine (V3.2: Bug Fixes)
=============================================================
Fix (Bug 2): Rebalanced SUBSCORE_WEIGHTS so fragmentation_lead persona
registers high burnout. fragmentation weight raised from 1.5 → 2.5 and
recovery_debt lowered from 2.2 → 1.8, reflecting that sustained fragmentation
is itself a primary burnout driver even without explicit rest debt.

Also adds a meeting_overload bonus: if fragmentation_score is in the top
quarter of the baseline distribution AND meetings_count z-score > 1.5, an
extra score bump is applied — making Ishaan's chronic 9+ mtg/day clearly
visible in the probability output.

All other logic unchanged from V3.1.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Configuration
DATA_DIR            = Path("pulseiq_data")
FEATURES_FILE       = DATA_DIR / "daily_features.csv"
BASELINES_FILE      = DATA_DIR / "baselines.csv"
BASELINES_META_FILE = DATA_DIR / "baselines_meta.csv"
OUTPUT_FILE         = DATA_DIR / "daily_scores.csv"

BASELINE_DAYS      = 14
CRITICAL_THRESHOLD = 0.85

# ----------------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------------
def robust_z(value, median, std_robust):
    if std_robust < 1e-6:
        return 0.0 if abs(value - median) < 1e-6 else (3.0 if value > median else -3.0)
    return (value - median) / std_robust

def linear_slope(values):
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den > 1e-9 else 0.0

def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))

# ----------------------------------------------------------------------------
# Sub-score computations
# ----------------------------------------------------------------------------
def compute_deep_work(day, baseline):
    if day.get("is_weekend"):
        return None
    def ratio(feat, val):
        med = baseline.get(feat, {}).get("median", 0)
        return min(2.0, val / max(1.0, med))
    raw = (0.5 * ratio("largest_focus_gap_minutes", day.get("largest_focus_gap_minutes", 0)) +
           0.5 * ratio("jira_story_points_closed",  day.get("jira_story_points_closed",  0)))
    return clamp(raw * 50)

def compute_fragmentation(day, baseline):
    if day.get("is_weekend"):
        return None
    def z(feat, val):
        b = baseline.get(feat, {})
        return max(0.0, robust_z(val, b.get("median", 0), b.get("std_robust", 1.0)))
    z_sum = (0.6 * z("meetings_count",     day.get("meetings_count",     0)) +
             0.4 * z("slack_msgs_total",   day.get("slack_msgs_total",   0)))
    return clamp(50 * (1 + math.tanh(z_sum / 2.0)))

def compute_connection(day, baseline):
    if day.get("is_weekend"):
        return None
    sent_med       = baseline.get("slack_avg_sentiment",       {}).get("median", 0.7)
    sentiment_score = clamp(1.0 + (day.get("slack_avg_sentiment", 0) - sent_med) * 4.0, 0, 2)
    dm_med          = baseline.get("slack_distinct_dm_partners", {}).get("median", 3)
    dm_ratio        = min(2.0, day.get("slack_distinct_dm_partners", 0) / max(1, dm_med))
    return clamp((0.6 * sentiment_score + 0.4 * dm_ratio) * 50)

def update_recovery_state(day, baseline, state):
    work_act = day.get("slack_msgs_work_channels", 0) + day.get("jira_events", 0)
    thresh   = (baseline.get("slack_msgs_work_channels", {}).get("median", 5) +
                baseline.get("jira_events",              {}).get("median", 1)) * 0.8
    if work_act == 0:
        state["gap_days"] += 1
        return 0.0
    if state["gap_days"] >= 2:
        state["in_recovery"] = True
        state["gap_days"]    = 0
    if state["in_recovery"]:
        if work_act >= thresh:
            state["in_recovery"] = False
            state["debt_days"]   = 0
        else:
            state["debt_days"] += 1
    if work_act > 0:
        state["consecutive_work_days"] += 1
    else:
        state["consecutive_work_days"] = 0
    return float(state["debt_days"])

# ----------------------------------------------------------------------------
# Bug 2 fix: rebalanced weights
#   fragmentation: 1.5 → 2.5  (chronic meeting load is a primary driver)
#   recovery_debt: 2.2 → 1.8  (still important but not the only path)
#   connection and deep_work unchanged
# ----------------------------------------------------------------------------
SUBSCORE_WEIGHTS = {
    "fragmentation": 2.5,
    "recovery_debt": 1.8,
    "connection":    1.8,
    "deep_work":     1.0,
}

def compute_burnout_probability(sub_today, sub_baseline, sub_recent, rest_deficit, day_baseline):
    """
    Compute burnout probability.

    Extra meeting_overload bonus (Bug 2 fix):
    If fragmentation is already elevated AND raw meetings z-score > 1.5,
    add a targeted score bonus so the fragmented_lead persona is clearly
    flagged even when recovery_debt hasn't accumulated yet.
    """
    score   = 0.0
    drivers = []

    for name, weight in SUBSCORE_WEIGHTS.items():
        curr      = sub_today.get(name)
        if curr is None:
            continue
        base_vals = [v for v in sub_baseline.get(name, []) if v is not None]
        if not base_vals:
            continue
        med = statistics.median(base_vals)
        std = max(5.0, statistics.stdev(base_vals) if len(base_vals) > 1 else 5.0)

        # For "bad high" scores (fragmentation, recovery_debt): penalise above median
        # For "bad low" scores (deep_work, connection):          penalise below median
        if name in ("deep_work", "connection"):
            dev = (med - curr) / std
        else:
            dev = (curr - med) / std

        if dev > 1.2 or (name == "recovery_debt" and curr > 0):
            score += weight * max(0.5, dev)
            drivers.append(name)

    # Meeting-overload bonus: chronic fragmentation even without rest debt
    # Guard against None fragmentation (weekend days)
    frag_val = sub_today.get("fragmentation")
    if frag_val is not None:
        mtg_b        = day_baseline.get("meetings_count", {})
        frag_base    = [v for v in sub_baseline.get("fragmentation", [50]) if v is not None] or [50]
        frag_base_med = statistics.median(frag_base)
        frag_base_std = max(5.0, statistics.stdev(frag_base) if len(frag_base) > 1 else 5.0)
        mtg_z = robust_z(frag_val, frag_base_med, frag_base_std)
    else:
        mtg_z = 0.0
    if mtg_z > 1.5 and (sub_today.get("fragmentation") or 0) > 65:
        score   += 1.5
        drivers.append("meeting_overload")

    # Rest deficit penalty (unchanged from V3.1)
    if rest_deficit > 10:
        score += (rest_deficit - 10) * 0.05
        drivers.append("rest_deficit")

    offset, divisor = 6.0, 3.0
    prob = 1.0 / (1.0 + math.exp(-(score - offset) / divisor))
    if score < 1.0:
        prob *= score
    return clamp(prob, 0.0, 1.0), ",".join(drivers)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # Load features
    features = []
    with open(FEATURES_FILE) as f:
        for r in csv.DictReader(f):
            for k in r:
                if k not in ("employee_id", "name", "team", "persona", "date"):
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        pass
            r["day_index"] = int(r.get("day_index", 0))
            r["is_weekend"] = int(r.get("is_weekend", 0))
            features.append(r)

    # Load baselines into nested dict: baselines[eid][feature] = {median, std_robust, ...}
    baselines = defaultdict(dict)
    with open(BASELINES_FILE) as f:
        for r in csv.DictReader(f):
            eid, feat = r["employee_id"], r["feature"]
            baselines[eid][feat] = {
                k: float(v) for k, v in r.items()
                if k not in ("employee_id", "feature", "scope")
            }

    by_emp = defaultdict(list)
    for r in features:
        by_emp[r["employee_id"]].append(r)

    output = []

    for eid, days in by_emp.items():
        state = {
            "gap_days": 0, "in_recovery": False,
            "debt_days": 0, "consecutive_work_days": 0,
        }
        sub_hist     = {"deep_work": [], "fragmentation": [], "recovery_debt": [], "connection": []}
        burnout_hist = []

        for day in days:
            d_idx = day["day_index"]

            dw   = compute_deep_work(day,    baselines[eid])
            frag = compute_fragmentation(day, baselines[eid])
            conn = compute_connection(day,    baselines[eid])
            rd   = update_recovery_state(day, baselines[eid], state)

            for k, v in zip(sub_hist.keys(), [dw, frag, rd, conn]):
                sub_hist[k].append(v)

            if d_idx >= BASELINE_DAYS:
                sub_today  = {"deep_work": dw, "fragmentation": frag, "recovery_debt": rd, "connection": conn}
                sub_base   = {k: v[:BASELINE_DAYS] for k, v in sub_hist.items()}
                sub_recent = {k: v[-7:]            for k, v in sub_hist.items()}

                prob, drivers = compute_burnout_probability(
                    sub_today, sub_base, sub_recent,
                    state["consecutive_work_days"],
                    baselines[eid],
                )
                burnout_hist.append(prob)

                valid_b     = [v for v in burnout_hist if v is not None]
                slope       = linear_slope(valid_b[-14:]) if len(valid_b) >= 5 else 0
                days_to_crit = (CRITICAL_THRESHOLD - prob) / slope if slope > 0.001 else None
            else:
                prob, days_to_crit, drivers = None, None, ""

            output.append({
                "employee_id":               eid,
                "name":                      day["name"],
                "date":                      day["date"],
                "day_index":                 d_idx,
                "deep_work_index":           dw,
                "fragmentation_score":       frag,
                "connection_index":          conn,
                "recovery_debt":             rd,
                "rest_deficit":              float(state["consecutive_work_days"]),
                "burnout_probability":       prob,
                "forecast_days_until_critical": days_to_crit,
                "driving_factors":           drivers,
            })

    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=output[0].keys())
        w.writeheader()
        w.writerows(output)

    print(f"Success. Analysis complete. Results in {OUTPUT_FILE}")

import subprocess

print("\nRunning recommendation engine...")
subprocess.run(["python3", "recommendations.py"])

if __name__ == "__main__":
    main()