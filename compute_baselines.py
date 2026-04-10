"""
PulseIQ — Step 2: Per-Person Baselines
=======================================
Reads daily_features.csv and computes a "normal" profile for each employee
using their first 14 days of data.

For each employee, for each numeric feature, we compute:
  - median       : their typical value (robust to spike days)
  - mad          : median absolute deviation (robust spread)
  - p25, p75     : quartiles for richer downstream comparisons
  - workday_only : whether the median was computed only on weekdays

Why median + MAD instead of mean + std:
  - Means get pulled around by one bad/big day
  - MAD ignores outliers, gives a stable "this is what normal looks like"
  - Std-equivalent: MAD * 1.4826 (we store both)

Why 14 days:
  - Long enough to capture two full work-weeks (Mon-Fri x 2)
  - Short enough to leave 46 days for actual detection on a 60-day dataset
  - In production, you'd extend this to 21-28 days

Weekend handling:
  - Most features (meetings, slack work msgs) are computed weekday-only
  - After-hours and weekend-msg features are computed across all days
  - This avoids "Saturday is normal because it's quiet" polluting the baseline
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("pulseiq_data")
INPUT_FILE = DATA_DIR / "daily_features.csv"
OUTPUT_FILE = DATA_DIR / "baselines.csv"

BASELINE_DAYS = 14  # use first 14 days as the learning window

# Features that should only be measured on workdays (Mon-Fri).
# Anything that's "active work" — measuring it on weekends would pollute the baseline.
WORKDAY_ONLY_FEATURES = {
    "slack_msgs_total",
    "slack_msgs_work_channels",
    "slack_msgs_social_channels",
    "slack_dms_sent",
    "slack_distinct_channels",
    "slack_distinct_dm_partners",
    "slack_avg_sentiment",
    "jira_events",
    "jira_distinct_tickets",
    "jira_tickets_closed",
    "jira_story_points_closed",
    "jira_comments",
    "meetings_count",
    "meetings_total_minutes",
    "meetings_attendees_total",
    "meetings_back_to_back",
    "largest_focus_gap_minutes",
    "after_hours_ratio",
    "social_msg_ratio",
}

# Features that should be measured across ALL days (incl. weekends), because
# the very point of measuring them is to catch out-of-hours work.
ALL_DAYS_FEATURES = {
    "slack_msgs_after_hours",
    "slack_msgs_weekend",
}

ALL_FEATURES = sorted(WORKDAY_ONLY_FEATURES | ALL_DAYS_FEATURES)

# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def median_absolute_deviation(values, med):
    """Robust spread metric. ~equivalent to std for normal data when *1.4826."""
    if not values:
        return 0.0
    return statistics.median([abs(v - med) for v in values])

def safe_quantile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac

def compute_feature_baseline(values):
    """Returns dict of summary stats for one feature, one person."""
    if not values:
        return {
            "median": 0.0,
            "mad": 0.0,
            "std_robust": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "n_observations": 0,
        }
    med = statistics.median(values)
    mad = median_absolute_deviation(values, med)
    return {
        "median": round(med, 4),
        "mad": round(mad, 4),
        "std_robust": round(mad * 1.4826, 4),  # std-equivalent for normal data
        "p25": round(safe_quantile(values, 0.25), 4),
        "p75": round(safe_quantile(values, 0.75), 4),
        "n_observations": len(values),
    }

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print(f"Loading {INPUT_FILE}...")
    rows = list(csv.DictReader(open(INPUT_FILE)))
    print(f"  {len(rows):,} rows loaded")

    # Group rows by employee, then take their first BASELINE_DAYS days
    by_emp = defaultdict(list)
    for r in rows:
        by_emp[r["employee_id"]].append(r)

    # Sort each employee's rows by day_index to be safe
    for eid in by_emp:
        by_emp[eid].sort(key=lambda r: int(r["day_index"]))

    print(f"\nComputing baselines from first {BASELINE_DAYS} days per employee...")

    # Output: one row per (employee, feature) — long format is easier to query
    output_rows = []
    employee_meta = {}

    for eid, emp_rows in by_emp.items():
        baseline_window = emp_rows[:BASELINE_DAYS]
        employee_meta[eid] = {
            "name": baseline_window[0]["name"],
            "team": baseline_window[0]["team"],
            "persona": baseline_window[0]["persona"],
        }

        # Detect chronotype: are they a "late starter" or "early bird"?
        # Approximated by comparing after-hours ratio in baseline. Most people
        # have very little after-hours activity in their healthy baseline; we
        # call anyone with >15% baseline after-hours a natural "night owl"
        # so we don't penalize them later.
        after_hours_baseline_vals = [
            int(r["slack_msgs_after_hours"]) for r in baseline_window
        ]
        msgs_total_baseline = sum(int(r["slack_msgs_total"]) for r in baseline_window)
        baseline_after_hours_ratio = (
            sum(after_hours_baseline_vals) / msgs_total_baseline
            if msgs_total_baseline else 0.0
        )
        chronotype = "night_owl" if baseline_after_hours_ratio > 0.15 else "standard"

        # Detect a "data sufficiency" flag — if the person had nearly zero
        # activity during baseline (e.g. on PTO during it), confidence is low.
        active_days_in_baseline = sum(
            1 for r in baseline_window
            if int(r["slack_msgs_total"]) > 0 or int(r["jira_events"]) > 0
        )
        data_sufficiency = (
            "high" if active_days_in_baseline >= 8 else
            "medium" if active_days_in_baseline >= 5 else
            "low"
        )

        employee_meta[eid]["chronotype"] = chronotype
        employee_meta[eid]["baseline_after_hours_ratio"] = round(baseline_after_hours_ratio, 3)
        employee_meta[eid]["active_days_in_baseline"] = active_days_in_baseline
        employee_meta[eid]["data_sufficiency"] = data_sufficiency

        # For each feature, gather the right slice of values and compute stats
        for feat in ALL_FEATURES:
            if feat in WORKDAY_ONLY_FEATURES:
                vals = [
                    float(r[feat]) for r in baseline_window
                    if int(r["is_weekend"]) == 0
                ]
            else:
                vals = [float(r[feat]) for r in baseline_window]

            stats = compute_feature_baseline(vals)
            output_rows.append({
                "employee_id": eid,
                "feature": feat,
                "scope": "workday_only" if feat in WORKDAY_ONLY_FEATURES else "all_days",
                **stats,
            })

    # Write the long-format baselines.csv
    fieldnames = [
        "employee_id", "feature", "scope",
        "median", "mad", "std_robust", "p25", "p75", "n_observations",
    ]
    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(output_rows)

    # Also write a meta file with chronotype + data sufficiency per person
    meta_path = DATA_DIR / "baselines_meta.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "employee_id", "name", "team", "persona",
            "chronotype", "baseline_after_hours_ratio",
            "active_days_in_baseline", "data_sufficiency",
        ])
        for eid, m in employee_meta.items():
            w.writerow([
                eid, m["name"], m["team"], m["persona"],
                m["chronotype"], m["baseline_after_hours_ratio"],
                m["active_days_in_baseline"], m["data_sufficiency"],
            ])

    print(f"\nWrote {len(output_rows):,} baseline rows to {OUTPUT_FILE}")
    print(f"  ({len(by_emp)} employees × {len(ALL_FEATURES)} features)")
    print(f"Wrote employee meta to {meta_path}")

    # Quick sanity print — show baselines for the planted personas
    print("\n--- Baselines for planted personas (key features only) ---")
    print(f"{'Persona':10s} {'Name':18s} {'AftHr%med':>10s} {'Mtg/d med':>10s} {'StPts med':>10s} {'Sent med':>10s} {'Chrono':>12s}")
    print("-" * 90)

    # Re-organize for printing
    by_emp_feat = defaultdict(dict)
    for r in output_rows:
        by_emp_feat[r["employee_id"]][r["feature"]] = r

    for eid, m in employee_meta.items():
        if m["persona"] == "healthy":
            continue
        feats = by_emp_feat[eid]
        aft = feats["after_hours_ratio"]["median"] * 100
        mtg = feats["meetings_count"]["median"]
        sp = feats["jira_story_points_closed"]["median"]
        sent = feats["slack_avg_sentiment"]["median"]
        print(f"{m['persona']:10s} {m['name']:18s} {aft:>9.1f}% {mtg:>10.1f} {sp:>10.1f} {sent:>10.2f} {m['chronotype']:>12s}")

if __name__ == "__main__":
    main()
