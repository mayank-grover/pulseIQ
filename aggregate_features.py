"""
PulseIQ — Step 1: Daily Feature Aggregation
============================================
Reads the 4 raw CSVs (employees, slack_messages, jira_events, calendar_events)
and produces daily_features.csv with one row per (employee, day).

Each row contains ~22 numeric features that the rest of the engine builds on:
  - Slack activity (volume, after-hours, weekend, social graph, sentiment)
  - Jira activity (tickets touched, closed, story points)
  - Calendar load (meeting count, total minutes, fragmentation, longest gap)

The output drives baselines, sub-scores, and the burnout probability.
Run AFTER generate_data.py.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path

DATA_DIR = Path("pulseiq_data")
OUTPUT_FILE = DATA_DIR / "daily_features.csv"

# After-hours window (local time)
AFTER_HOURS_START = 21  # 9pm
AFTER_HOURS_END = 7     # 7am
SOCIAL_CHANNELS = {"#random", "#general"}

# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_employees():
    employees = {}
    with open(DATA_DIR / "employees.csv") as f:
        for row in csv.DictReader(f):
            employees[row["employee_id"]] = row
    return employees

def load_slack():
    rows = []
    with open(DATA_DIR / "slack_messages.csv") as f:
        for r in csv.DictReader(f):
            r["timestamp"] = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            r["message_length"] = int(r["message_length"])
            r["sentiment_score"] = float(r["sentiment_score"])
            r["is_dm"] = r["is_dm"] == "true"
            rows.append(r)
    return rows

def load_jira():
    rows = []
    with open(DATA_DIR / "jira_events.csv") as f:
        for r in csv.DictReader(f):
            r["timestamp"] = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            r["story_points"] = int(r["story_points"])
            rows.append(r)
    return rows

def load_calendar():
    rows = []
    with open(DATA_DIR / "calendar_events.csv") as f:
        for r in csv.DictReader(f):
            r["date"] = datetime.strptime(r["date"], "%Y-%m-%d").date()
            r["start_min"] = int(r["start_time"].split(":")[0]) * 60 + int(r["start_time"].split(":")[1])
            r["end_min"] = int(r["end_time"].split(":")[0]) * 60 + int(r["end_time"].split(":")[1])
            r["attendee_count"] = int(r["attendee_count"])
            rows.append(r)
    return rows

# ----------------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------------
def is_after_hours(dt: datetime) -> bool:
    return dt.hour >= AFTER_HOURS_START or dt.hour < AFTER_HOURS_END

def is_weekend(d) -> bool:
    if isinstance(d, datetime):
        d = d.date()
    return d.weekday() >= 5

def compute_slack_features(slack_rows):
    """
    Returns dict keyed by (employee_id, date) with slack-derived features.
    """
    out = defaultdict(lambda: {
        "slack_msgs_total": 0,
        "slack_msgs_after_hours": 0,
        "slack_msgs_weekend": 0,
        "slack_msgs_work_channels": 0,
        "slack_msgs_social_channels": 0,
        "slack_dms_sent": 0,
        "slack_distinct_channels": set(),
        "slack_distinct_dm_partners": set(),
        "sentiment_sum": 0.0,
        "sentiment_count": 0,
    })

    for r in slack_rows:
        key = (r["employee_id"], r["timestamp"].date())
        f = out[key]
        f["slack_msgs_total"] += 1

        if is_after_hours(r["timestamp"]):
            f["slack_msgs_after_hours"] += 1
        if is_weekend(r["timestamp"]):
            f["slack_msgs_weekend"] += 1

        if r["is_dm"]:
            f["slack_dms_sent"] += 1
            if r["recipient_id"]:
                f["slack_distinct_dm_partners"].add(r["recipient_id"])
        else:
            f["slack_distinct_channels"].add(r["channel"])
            if r["channel"] in SOCIAL_CHANNELS:
                f["slack_msgs_social_channels"] += 1
            else:
                f["slack_msgs_work_channels"] += 1

        f["sentiment_sum"] += r["sentiment_score"]
        f["sentiment_count"] += 1

    # Finalize sets into counts and compute avg sentiment
    finalized = {}
    for key, f in out.items():
        finalized[key] = {
            "slack_msgs_total": f["slack_msgs_total"],
            "slack_msgs_after_hours": f["slack_msgs_after_hours"],
            "slack_msgs_weekend": f["slack_msgs_weekend"],
            "slack_msgs_work_channels": f["slack_msgs_work_channels"],
            "slack_msgs_social_channels": f["slack_msgs_social_channels"],
            "slack_dms_sent": f["slack_dms_sent"],
            "slack_distinct_channels": len(f["slack_distinct_channels"]),
            "slack_distinct_dm_partners": len(f["slack_distinct_dm_partners"]),
            "slack_avg_sentiment": (
                round(f["sentiment_sum"] / f["sentiment_count"], 3)
                if f["sentiment_count"] else 0.0
            ),
        }
    return finalized

def compute_jira_features(jira_rows):
    out = defaultdict(lambda: {
        "jira_events": 0,
        "jira_distinct_tickets": set(),
        "jira_tickets_closed": 0,
        "jira_story_points_closed": 0,
        "jira_comments": 0,
    })
    for r in jira_rows:
        key = (r["employee_id"], r["timestamp"].date())
        f = out[key]
        f["jira_events"] += 1
        f["jira_distinct_tickets"].add(r["ticket_id"])
        if r["event_type"] == "comment":
            f["jira_comments"] += 1
        if r["to_status"] == "Done":
            f["jira_tickets_closed"] += 1
            f["jira_story_points_closed"] += r["story_points"]

    finalized = {}
    for key, f in out.items():
        finalized[key] = {
            "jira_events": f["jira_events"],
            "jira_distinct_tickets": len(f["jira_distinct_tickets"]),
            "jira_tickets_closed": f["jira_tickets_closed"],
            "jira_story_points_closed": f["jira_story_points_closed"],
            "jira_comments": f["jira_comments"],
        }
    return finalized

def compute_calendar_features(cal_rows):
    """
    For each (employee, day), compute meeting load + fragmentation signals:
      - meetings_count
      - meetings_total_minutes
      - meetings_attendees_total (sum of attendees across all their meetings — proxy for context switching)
      - meetings_back_to_back (pairs of meetings with <15 min gap)
      - largest_focus_gap_minutes (largest gap between meetings during 9am-6pm)
    """
    by_key = defaultdict(list)
    for r in cal_rows:
        by_key[(r["employee_id"], r["date"])].append(r)

    out = {}
    WORK_START_MIN = 9 * 60   # 9am
    WORK_END_MIN = 18 * 60    # 6pm

    for key, meetings in by_key.items():
        meetings.sort(key=lambda m: m["start_min"])
        total_min = sum(m["end_min"] - m["start_min"] for m in meetings)
        attendees = sum(m["attendee_count"] for m in meetings)

        # Count back-to-back pairs
        b2b = 0
        for i in range(1, len(meetings)):
            gap = meetings[i]["start_min"] - meetings[i - 1]["end_min"]
            if 0 <= gap < 15:
                b2b += 1

        # Largest focus gap inside the workday
        # Build a sorted list of (start, end) within work hours
        intervals = []
        for m in meetings:
            s = max(m["start_min"], WORK_START_MIN)
            e = min(m["end_min"], WORK_END_MIN)
            if s < e:
                intervals.append((s, e))
        intervals.sort()

        # Sweep to find largest open gap
        cursor = WORK_START_MIN
        largest_gap = 0
        for s, e in intervals:
            if s > cursor:
                largest_gap = max(largest_gap, s - cursor)
            cursor = max(cursor, e)
        if cursor < WORK_END_MIN:
            largest_gap = max(largest_gap, WORK_END_MIN - cursor)

        # If no meetings, the whole workday is one big gap
        if not meetings:
            largest_gap = WORK_END_MIN - WORK_START_MIN

        out[key] = {
            "meetings_count": len(meetings),
            "meetings_total_minutes": total_min,
            "meetings_attendees_total": attendees,
            "meetings_back_to_back": b2b,
            "largest_focus_gap_minutes": largest_gap,
        }
    return out

# ----------------------------------------------------------------------------
# Main aggregation
# ----------------------------------------------------------------------------
def main():
    print("Loading raw CSVs...")
    employees = load_employees()
    slack = load_slack()
    jira = load_jira()
    calendar = load_calendar()
    print(f"  {len(employees)} employees, {len(slack):,} slack msgs, "
          f"{len(jira):,} jira events, {len(calendar):,} calendar events")

    print("Computing per-day features...")
    slack_feats = compute_slack_features(slack)
    jira_feats = compute_jira_features(jira)
    cal_feats = compute_calendar_features(calendar)

    # Determine the full date range from the slack data
    all_dates = sorted({r["timestamp"].date() for r in slack})
    start_date = all_dates[0]
    end_date = all_dates[-1]
    num_days = (end_date - start_date).days + 1
    print(f"  Date range: {start_date} -> {end_date} ({num_days} days)")

    # Build the dense matrix: every employee × every day, even zero-activity days.
    # Zero-activity days matter — they're how we detect rest (or its absence).
    feature_columns = [
        # identifiers
        "employee_id", "name", "team", "persona", "date", "day_index", "weekday", "is_weekend",
        # slack
        "slack_msgs_total", "slack_msgs_after_hours", "slack_msgs_weekend",
        "slack_msgs_work_channels", "slack_msgs_social_channels",
        "slack_dms_sent", "slack_distinct_channels", "slack_distinct_dm_partners",
        "slack_avg_sentiment",
        # jira
        "jira_events", "jira_distinct_tickets", "jira_tickets_closed",
        "jira_story_points_closed", "jira_comments",
        # calendar
        "meetings_count", "meetings_total_minutes", "meetings_attendees_total",
        "meetings_back_to_back", "largest_focus_gap_minutes",
        # derived ratios (helpful downstream)
        "after_hours_ratio", "social_msg_ratio",
    ]

    rows_written = 0
    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(feature_columns)

        for eid, emp in employees.items():
            for i in range(num_days):
                d = start_date + timedelta(days=i)
                key = (eid, d)

                slack_f = slack_feats.get(key, {})
                jira_f = jira_feats.get(key, {})
                cal_f = cal_feats.get(key, {})

                msgs_total = slack_f.get("slack_msgs_total", 0)
                msgs_after = slack_f.get("slack_msgs_after_hours", 0)
                msgs_social = slack_f.get("slack_msgs_social_channels", 0)

                after_hours_ratio = round(msgs_after / msgs_total, 3) if msgs_total else 0.0
                social_msg_ratio = round(msgs_social / msgs_total, 3) if msgs_total else 0.0

                # Default calendar largest gap = full workday if no meetings
                largest_gap = cal_f.get("largest_focus_gap_minutes", 9 * 60 if not is_weekend(d) else 0)

                w.writerow([
                    eid, emp["name"], emp["team"], emp["persona"],
                    d.isoformat(), i, d.weekday(), 1 if is_weekend(d) else 0,
                    msgs_total, msgs_after, slack_f.get("slack_msgs_weekend", 0),
                    slack_f.get("slack_msgs_work_channels", 0), msgs_social,
                    slack_f.get("slack_dms_sent", 0),
                    slack_f.get("slack_distinct_channels", 0),
                    slack_f.get("slack_distinct_dm_partners", 0),
                    slack_f.get("slack_avg_sentiment", 0.0),
                    jira_f.get("jira_events", 0),
                    jira_f.get("jira_distinct_tickets", 0),
                    jira_f.get("jira_tickets_closed", 0),
                    jira_f.get("jira_story_points_closed", 0),
                    jira_f.get("jira_comments", 0),
                    cal_f.get("meetings_count", 0),
                    cal_f.get("meetings_total_minutes", 0),
                    cal_f.get("meetings_attendees_total", 0),
                    cal_f.get("meetings_back_to_back", 0),
                    largest_gap,
                    after_hours_ratio, social_msg_ratio,
                ])
                rows_written += 1

    print(f"\nWrote {rows_written:,} rows to {OUTPUT_FILE}")
    print(f"  ({len(employees)} employees × {num_days} days)")

if __name__ == "__main__":
    main()
