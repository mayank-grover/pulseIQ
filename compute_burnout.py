"""
PulseIQ — Step 3: Sub-Scores + Burnout Probability + Forecast Date
====================================================================
MODIFIED VERSION: Recovery Debt now triggers on 2-day breaks (weekends)
and ignores social noise to properly track Monday re-entry friction.
"""

import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

DATA_DIR = Path("pulseiq_data")
FEATURES_FILE = DATA_DIR / "daily_features.csv"
BASELINES_FILE = DATA_DIR / "baselines.csv"
BASELINES_META_FILE = DATA_DIR / "baselines_meta.csv"
OUTPUT_FILE = DATA_DIR / "daily_scores.csv"

# Configuration
BASELINE_DAYS = 14
TREND_WINDOW = 7
FORECAST_WINDOW = 14
CRITICAL_THRESHOLD = 0.85
WARNING_THRESHOLD = 0.50

# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def robust_z(value, median, std_robust):
    if std_robust < 1e-6:
        return 0.0 if abs(value - median) < 1e-6 else (1.0 if value > median else -1.0) * 3.0
    return (value - median) / std_robust

def linear_slope(values):
    n = len(values)
    if n < 2: return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den > 1e-9 else 0.0

def linear_regression(values):
    n = len(values)
    if n < 2: return 0.0, values[0] if values else 0.0, 0.0
    xs = list(range(n))
    mean_x, mean_y = sum(xs)/n, sum(values)/n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den < 1e-9: return 0.0, mean_y, 0.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    ss_res = sum((values[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
    ss_tot = sum((values[i] - mean_y) ** 2 for i in range(n))
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    return slope, intercept, max(0.0, r_sq)

def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_features():
    rows = []
    if not FEATURES_FILE.exists(): return []
    with open(FEATURES_FILE) as f:
        for r in csv.DictReader(f):
            for k in r:
                if k in ("employee_id", "name", "team", "persona", "date"): continue
                try: r[k] = float(r[k])
                except: pass
            r["day_index"] = int(r["day_index"])
            r["is_weekend"] = int(r["is_weekend"])
            rows.append(r)
    return rows

def load_baselines():
    out = defaultdict(dict)
    if not BASELINES_FILE.exists(): return out
    with open(BASELINES_FILE) as f:
        for r in csv.DictReader(f):
            out[r["employee_id"]][r["feature"]] = {
                "median": float(r["median"]),
                "mad": float(r["mad"]),
                "std_robust": float(r["std_robust"]),
                "p25": float(r["p25"]),
                "p75": float(r["p75"]),
            }
    return out

def load_baselines_meta():
    out = {}
    if not BASELINES_META_FILE.exists(): return out
    with open(BASELINES_META_FILE) as f:
        for r in csv.DictReader(f):
            out[r["employee_id"]] = {
                "chronotype": r["chronotype"],
                "data_sufficiency": r["data_sufficiency"],
                "active_days_in_baseline": int(r["active_days_in_baseline"]),
            }
    return out

# ----------------------------------------------------------------------------
# Sub-Scores (Deep Work, Fragmentation, Connection)
# ----------------------------------------------------------------------------
def compute_deep_work(day, baseline):
    if day["is_weekend"]: return None
    def ratio(feat, val):
        med = baseline.get(feat, {}).get("median", 0)
        return min(2.0, val / max(1.0, med))

    focus_gap = ratio("largest_focus_gap_minutes", day["largest_focus_gap_minutes"])
    story_pts = ratio("jira_story_points_closed", day["jira_story_points_closed"])

    mtg_med = baseline.get("meetings_count", {}).get("median", 3.0)
    mtg_score = min(2.0, mtg_med / max(0.5, day["meetings_count"])) if mtg_med > 0.5 else 1.0

    msg_med = baseline.get("slack_msgs_total", {}).get("median", 10.0)
    focus_quality = 0.5 if day["largest_focus_gap_minutes"] >= 120 and day["slack_msgs_total"] > msg_med * 1.5 else 1.0

    raw = (0.35 * focus_gap + 0.30 * story_pts + 0.20 * mtg_score + 0.15 * focus_quality)
    return clamp(raw * 50)

def compute_fragmentation(day, baseline):
    if day["is_weekend"]: return None
    def z(feat, val):
        b = baseline.get(feat, {})
        return max(0.0, robust_z(val, b.get("median", 0), b.get("std_robust", 1.0)))

    z_sum = (0.25 * z("meetings_count", day["meetings_count"]) +
             0.20 * z("meetings_back_to_back", day["meetings_back_to_back"]) +
             0.15 * z("slack_distinct_channels", day["slack_distinct_channels"]) +
             0.15 * z("jira_distinct_tickets", day["jira_distinct_tickets"]))

    return clamp(50 * (1 + math.tanh(z_sum / 2.0)))

def compute_connection(day, baseline):
    if day["is_weekend"]: return None
    def ratio(feat, val):
        med = baseline.get(feat, {}).get("median", 0)
        return min(2.0, val / max(1.0, med))

    dm_p = ratio("slack_distinct_dm_partners", day["slack_distinct_dm_partners"])
    soc = ratio("slack_msgs_social_channels", day["slack_msgs_social_channels"])

    sent_med = baseline.get("slack_avg_sentiment", {}).get("median", 0.7)
    sentiment_score = clamp(1.0 + (day["slack_avg_sentiment"] - sent_med) * 3.0, 0, 2) if day["slack_msgs_total"] > 0 else 0.5

    raw = (0.30 * dm_p + 0.25 * soc + 0.30 * sentiment_score + 0.15 * ratio("slack_msgs_total", day["slack_msgs_total"]))
    return clamp(raw * 50)

# ----------------------------------------------------------------------------
# MODIFIED Recovery Debt: Weekend-Aware Friction
# ----------------------------------------------------------------------------
def compute_recovery_debt_step(day, baseline, state):
    """
    Counts days taken to hit 80% baseline activity after a 2-day gap (weekend).
    """
    # IGNORE SOCIAL MESSAGES: Only count work activity
    work_activity = day["slack_msgs_work_channels"] + day["jira_events"]

    # Calculate target activity
    # $$ \text{Threshold} = (\text{Slack}_{\text{med}} + \text{Jira}_{\text{med}}) \times 0.8 $$
    slack_med = baseline.get("slack_msgs_work_channels", {}).get("median", 5.0)
    jira_med = baseline.get("jira_events", {}).get("median", 1.0)
    baseline_threshold = (slack_med + jira_med) * 0.8

    # 1. Check for absence
    if work_activity == 0:
        state["consecutive_inactive_days"] += 1
        state["is_in_recovery_period"] = False
        state["days_in_recovery"] = 0
        return 0.0

    # 2. TRIGGER ON WEEKENDS (2 days)
    if state["consecutive_inactive_days"] >= 2:
        state["is_in_recovery_period"] = True
        state["consecutive_inactive_days"] = 0

    # 3. Handle the ramp-up phase
    if state["is_in_recovery_period"]:
        if work_activity >= baseline_threshold:
            state["is_in_recovery_period"] = False
            state["days_in_recovery"] = 0
        else:
            # Each day they miss the target, debt increases
            state["days_in_recovery"] += 1

    return float(min(10, state["days_in_recovery"]))

# ----------------------------------------------------------------------------
# Burnout Engine Logic
# ----------------------------------------------------------------------------
SUBSCORE_WEIGHTS = {"recovery_debt": 2.0, "connection": 1.3, "deep_work": 1.0, "fragmentation": 0.8}

def is_degraded(name, current, baseline_hist, recent_hist):
    valid_baseline = [v for v in baseline_hist if v is not None]
    valid_recent = [v for v in recent_hist if v is not None]

    if not valid_baseline or current is None: return False, 0.0

    med = statistics.median(valid_baseline)
    mad = statistics.median([abs(v - med) for v in valid_baseline])
    std_r = max(8.0, mad * 1.4826)

    if valid_recent:
        recent_slice = valid_recent[-5:]
        smoothed = sum(recent_slice) / len(recent_slice)
    else:
        smoothed = current

    if name == "recovery_debt":
        # Any 'ramp-up' day is technically degraded re-entry
        return current > 0, current / 1.5

    if name in ("deep_work", "connection"):
        dev = (med - smoothed) / std_r
    else:
        dev = (smoothed - med) / std_r
    return dev > 2.0, dev

def compute_burnout_probability(sub_today, sub_baseline, sub_recent):
    score = 0.0
    driving_factors = []
    degraded_count = 0

    for name, weight in SUBSCORE_WEIGHTS.items():
        curr = sub_today.get(name)
        if curr is None: continue

        degraded, dev = is_degraded(name, curr, sub_baseline.get(name, []), sub_recent.get(name, []))
        if degraded:
            recent = [v for v in sub_recent.get(name, []) if v is not None]
            slope = linear_slope(recent) if len(recent) >= 3 else 0.0
            trend_bad = max(0.0, -slope) if name in ("deep_work", "connection") else max(0.0, slope)

            contribution = weight * dev * (1.0 + trend_bad / 5.0)
            score += contribution
            degraded_count += 1
            driving_factors.append({"name": name, "contribution": contribution})

    if degraded_count >= 2: score *= 1.5
    burnout_prob = sigmoid((score - 4.0) / 2.0)
    driving_factors.sort(key=lambda f: -f["contribution"])
    return burnout_prob, driving_factors, degraded_count

def compute_forecast(history):
    valid = [v for v in history if v is not None]
    if len(valid) < 5: return None, 0.0
    slope, intercept, r_sq = linear_regression(valid[-FORECAST_WINDOW:])
    if slope <= 0.001: return None, r_sq
    if valid[-1] >= CRITICAL_THRESHOLD: return 0, r_sq
    days = (CRITICAL_THRESHOLD - valid[-1]) / slope
    return int(round(days)) if days < 60 else None, r_sq

# ----------------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------------
def main():
    features = load_features()
    baselines = load_baselines()
    meta = load_baselines_meta()

    if not features:
        print("No features found. Run aggregate_features.py first.")
        return

    by_emp = defaultdict(list)
    for r in features: by_emp[r["employee_id"]].append(r)

    output_rows = []

    for eid, days in by_emp.items():
        baseline = baselines[eid]
        emp_meta = meta.get(eid, {"data_sufficiency": "low"})

        recovery_state = {"consecutive_inactive_days": 0, "is_in_recovery_period": False, "days_in_recovery": 0}
        sub_hist = {"deep_work": [], "fragmentation": [], "recovery_debt": [], "connection": []}
        burnout_hist = []
        first_flag_date = None

        for day in days:
            d_idx = day["day_index"]
            dw, frag, conn = compute_deep_work(day, baseline), compute_fragmentation(day, baseline), compute_connection(day, baseline)
            rd = compute_recovery_debt_step(day, baseline, recovery_state)

            for k, v in zip(sub_hist.keys(), [dw, frag, rd, conn]):
                sub_hist[k].append(v)

            if d_idx < BASELINE_DAYS:
                b_prob, forecast, conf, deg_count, drivers = None, None, "learning", 0, ""
            else:
                sub_today = {"deep_work": dw, "fragmentation": frag, "recovery_debt": rd, "connection": conn}
                sub_base = {k: v[:BASELINE_DAYS] for k, v in sub_hist.items()}
                sub_recent = {k: v[-10:] for k, v in sub_hist.items()}

                b_prob, dr_list, deg_count = compute_burnout_probability(sub_today, sub_base, sub_recent)
                burnout_hist.append(b_prob)
                forecast, r_sq = compute_forecast(burnout_hist)

                if emp_meta["data_sufficiency"] == "low": conf = "low"
                elif r_sq > 0.6: conf = "high"
                else: conf = "medium"

                drivers = ",".join(f["name"] for f in dr_list)
                if first_flag_date is None and b_prob >= WARNING_THRESHOLD:
                    first_flag_date = day["date"]

            output_rows.append({
                "employee_id": eid, "name": day["name"], "team": day["team"], "persona": day["persona"],
                "date": day["date"], "day_index": d_idx, "deep_work_index": dw, "fragmentation_score": frag,
                "recovery_debt": rd, "connection_index": conn, "burnout_probability": b_prob,
                "forecast_days_until_critical": forecast, "confidence": conf, "degraded_subscores_count": deg_count,
                "driving_factors": drivers, "first_flagged_date": first_flag_date or ""
            })

    OUTPUT_DIR = DATA_DIR
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        w.writeheader()
        w.writerows(output_rows)
    print(f"Success. Analysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()