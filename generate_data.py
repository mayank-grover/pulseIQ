"""
PulseIQ Synthetic Data Generator (V3.2: Bug Fixes)
===================================================
Fixes from V3.1:
  - Bug 1: Jira events now have varied event_type and realistic to_status
            transitions so jira_tickets_closed / story_points_closed > 0.
  - Bug 3: DM recipient_id is now picked randomly from the employee pool
            (not hardcoded to E002) so slack_distinct_dm_partners is meaningful.
  - Bug 4: weekend_warrior messages are now timestamped in the late-evening
            window (21:00-23:30) so is_after_hours() fires correctly for them.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

START_DATE = datetime(2026, 2, 9)   # Monday
NUM_DAYS = 60
REORG_DAY = 30
OUTPUT_DIR = Path("pulseiq_data")
OUTPUT_DIR.mkdir(exist_ok=True)

TEAMS = {
    "Engineering": {"channels": ["#eng", "#eng-help", "#dev"], "prefix": "ENG", "size": 11},
    "Design":      {"channels": ["#design", "#critique"],       "prefix": "DES", "size": 9},
    "Platform":    {"channels": ["#infra", "#incidents"],       "prefix": "PLAT", "size": 10},
}

# Jira realistic ticket pool
TICKET_IDS  = [f"TKT-{n}" for n in range(100, 140)]
STORY_POINT_OPTIONS = [1, 2, 3, 5, 8]

# Jira event-type distribution (weighted): most activity is comments/updates,
# with a realistic ~25% chance of actually closing a ticket on a given touch.
JIRA_EVENT_TYPES = ["comment", "status_change", "status_change", "status_change"]
JIRA_TRANSITIONS = {
    # from_status -> (to_status, is_close)
    "To Do":        [("In Progress", False), ("In Progress", False), ("Done", True)],
    "In Progress":  [("In Progress", False), ("Done", True), ("Done", True), ("In Review", False)],
    "In Review":    [("Done", True), ("In Progress", False)],
    "Done":         [("Done", False)],
}

def pick_jira_event(rng):
    """Returns (event_type, ticket_id, story_points, from_status, to_status)."""
    ticket = rng.choice(TICKET_IDS)
    event_type = rng.choice(JIRA_EVENT_TYPES)
    sp = rng.choice(STORY_POINT_OPTIONS)
    if event_type == "comment":
        return event_type, ticket, sp, "In Progress", "In Progress"
    from_status = rng.choice(list(JIRA_TRANSITIONS.keys()))
    to_status, _ = rng.choice(JIRA_TRANSITIONS[from_status])
    return event_type, ticket, sp, from_status, to_status

# ----------------------------------------------------------------------------
# Helpers & Templates
# ----------------------------------------------------------------------------
def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5

def jitter_minutes(base_hour: int, base_minute: int = 0, spread: int = 30) -> tuple:
    """Return (hour, minute) jittered around base_hour:base_minute."""
    minutes_offset = random.randint(-spread, spread)
    total = base_hour * 60 + base_minute + minutes_offset
    total = max(0, min(24 * 60 - 1, total))
    return divmod(total, 60)

def after_hours_ts(dt: datetime) -> datetime:
    """
    Bug 4 fix: weekend_warrior messages get a timestamp in the 21:00–23:30
    window so is_after_hours() (>= 21) fires correctly.
    """
    h = random.randint(21, 23)
    m = random.randint(0, 59) if h < 23 else random.randint(0, 30)
    return dt.replace(hour=h, minute=m, second=random.randint(0, 59))

MESSAGE_TEMPLATES = {
    "positive":  {"work": ["Merged!", "Great catch.", "Looks good."],          "social": ["Coffee?", "Match tonight?"]},
    "neutral":   {"work": ["Checking.", "Updated ticket.", "Sync at 3?"],      "social": ["Lunch?", "Machine broken."]},
    "negative":  {"work": ["Stuck.", "Tooling sucks.", "Pushing deadline."],   "social": ["Too busy.", "Maybe later."]},
    "exhausted": {"work": ["Still on it.", "K.", "Not today."],                "social": ["No.", "Cant."]},
}

def pick_message_text(sentiment: float, channel: str) -> str:
    bucket = ("positive" if sentiment >= 0.7 else
              "neutral"  if sentiment >= 0.5 else
              "negative" if sentiment >= 0.3 else "exhausted")
    pool = "social" if channel in ("#random", "#general") else "work"
    return random.choice(MESSAGE_TEMPLATES[bucket][pool])

# ----------------------------------------------------------------------------
# Persona Definitions
# ----------------------------------------------------------------------------
def healthy_profile(day_idx, dt, emp, ctx):
    if is_weekend(dt):
        return dict(msg_count_work=0, msg_count_social=0, msg_count_dms=0,
                    after_hours=False, tickets_touched=0, meetings_count=0, sentiment=0.7)
    return dict(
        msg_count_work=random.randint(5, 10),
        msg_count_social=random.randint(1, 3),
        msg_count_dms=random.randint(2, 5),
        after_hours=False,
        tickets_touched=random.randint(1, 3),
        meetings_count=random.randint(2, 4),
        sentiment=random.uniform(0.6, 0.8),
    )

def priya_profile(day_idx, dt, emp, ctx):
    if day_idx < REORG_DAY:
        return healthy_profile(day_idx, dt, emp, ctx)
    intensity = min(1.0, (day_idx - REORG_DAY) / 30)
    return dict(
        msg_count_work=random.randint(3, 6),
        msg_count_social=0,
        msg_count_dms=2,
        after_hours=random.random() < (0.1 + 0.5 * intensity),   # prob of after-hours day
        tickets_touched=1,
        meetings_count=6,
        sentiment=max(0.2, 0.7 - 0.5 * intensity),
    )

def fragmented_lead_profile(day_idx, dt, emp, ctx):
    return dict(
        msg_count_work=25,
        msg_count_social=2,
        msg_count_dms=10,
        after_hours=random.random() < 0.3,
        tickets_touched=0,
        meetings_count=random.randint(8, 11),
        sentiment=0.5,
    )

def weekend_warrior_profile(day_idx, dt, emp, ctx):
    if is_weekend(dt):
        # Bug 4 fix: flag after_hours=True so timestamps get pushed into 21:00+
        return dict(msg_count_work=8, msg_count_social=0, msg_count_dms=4,
                    after_hours=True, tickets_touched=2, meetings_count=0, sentiment=0.5)
    return healthy_profile(day_idx, dt, emp, ctx)

def silent_struggler_profile(day_idx, dt, emp, ctx):
    intensity = min(1.0, day_idx / 60)
    return dict(
        msg_count_work=3,
        msg_count_social=0,
        msg_count_dms=1,
        after_hours=False,
        tickets_touched=5,
        meetings_count=1,
        sentiment=max(0.25, 0.75 - 0.5 * intensity),
    )

PERSONA_FUNCS = {
    "healthy":           healthy_profile,
    "priya":             priya_profile,
    "fragmented_lead":   fragmented_lead_profile,
    "weekend_warrior":   weekend_warrior_profile,
    "silent_struggler":  silent_struggler_profile,
    "rohan":             healthy_profile,
    "sam":               healthy_profile,
    "arjun":             healthy_profile,
    "karthik":           healthy_profile,
}

# ----------------------------------------------------------------------------
# Build and Run
# ----------------------------------------------------------------------------
def main():
    employees = []
    planted = [
        ("Priya Sharma",  "Platform",    "priya"),
        ("Ishaan Verma",  "Engineering", "fragmented_lead"),
        ("Vihaan Gupta",  "Engineering", "weekend_warrior"),
        ("Saanvi Reddy",  "Design",      "silent_struggler"),
    ]
    for i, (name, team, persona) in enumerate(planted):
        employees.append({"employee_id": f"E{i+1:03d}", "name": name, "team": team, "persona": persona})
    for i in range(len(employees) + 1, 31):
        employees.append({"employee_id": f"E{i:03d}", "name": f"Employee {i}",
                          "team": "Engineering", "persona": "healthy"})

    employee_ids = [e["employee_id"] for e in employees]

    with open(OUTPUT_DIR / "employees.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["employee_id", "name", "team", "persona"])
        w.writeheader()
        w.writerows(employees)

    slack_f = open(OUTPUT_DIR / "slack_messages.csv",  "w", newline="")
    jira_f  = open(OUTPUT_DIR / "jira_events.csv",     "w", newline="")
    cal_f   = open(OUTPUT_DIR / "calendar_events.csv", "w", newline="")

    slack_w = csv.writer(slack_f)
    jira_w  = csv.writer(jira_f)
    cal_w   = csv.writer(cal_f)

    slack_w.writerow(["message_id", "employee_id", "timestamp", "channel",
                      "message_length", "is_dm", "recipient_id", "sentiment_score", "message_text"])
    jira_w.writerow(["employee_id", "timestamp", "ticket_id", "event_type",
                     "story_points", "from_status", "to_status"])
    cal_w.writerow(["meeting_id", "employee_id", "date", "start_time", "end_time", "title", "attendee_count"])

    msg_idx = 0
    rng = random.Random(SEED)   # separate RNG so persona calls don't shift it

    for day_idx in range(NUM_DAYS):
        dt = START_DATE + timedelta(days=day_idx)
        for emp in employees:
            profile = PERSONA_FUNCS.get(emp["persona"], healthy_profile)(day_idx, dt, emp, {})
            eid = emp["employee_id"]

            # ----------------------------------------------------------------
            # Slack messages
            # ----------------------------------------------------------------
            total_msgs = profile["msg_count_work"] + profile["msg_count_social"] + profile["msg_count_dms"]
            for i in range(total_msgs):
                msg_idx += 1
                is_dm_msg = i >= (total_msgs - profile["msg_count_dms"])
                channel   = "#work" if i < profile["msg_count_work"] else "#random"

                # Bug 4 fix: after_hours flag drives timestamp window
                if profile["after_hours"]:
                    ts = after_hours_ts(dt)
                else:
                    h, m = jitter_minutes(14, 0, 180)   # normal work hours ±3h around 2pm
                    ts = dt.replace(hour=h, minute=m, second=rng.randint(0, 59))

                # Bug 3 fix: pick a random recipient (not always E002)
                if is_dm_msg:
                    others = [e for e in employee_ids if e != eid]
                    recipient = rng.choice(others)
                else:
                    recipient = ""

                text = pick_message_text(profile["sentiment"], channel)
                slack_w.writerow([
                    f"M{msg_idx:07d}", eid, fmt_ts(ts), channel,
                    len(text), "true" if is_dm_msg else "false",
                    recipient, profile["sentiment"], text,
                ])

            # ----------------------------------------------------------------
            # Jira events
            # Bug 1 fix: varied event_type + real Done transitions
            # ----------------------------------------------------------------
            for _ in range(profile["tickets_touched"]):
                event_type, ticket, sp, from_s, to_s = pick_jira_event(rng)
                jira_w.writerow([eid, fmt_ts(dt), ticket, event_type, sp, from_s, to_s])

            # ----------------------------------------------------------------
            # Calendar
            # ----------------------------------------------------------------
            for i in range(profile["meetings_count"]):
                cal_w.writerow([
                    f"MTG-{msg_idx}-{i}", eid, dt.strftime("%Y-%m-%d"),
                    f"{10+i}:00", f"{11+i}:00", "Sync", 5,
                ])

    slack_f.close()
    jira_f.close()
    cal_f.close()
    print(f"Success: Data with full headers generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()