"""
PulseIQ Synthetic Data Generator
=================================
Generates 60 days of activity for 30 employees across 3 teams.
Outputs 4 CSVs: employees.csv, slack_messages.csv, jira_events.csv, calendar_events.csv

Plants 6 specific personas the burnout engine must catch:
  - Priya  : gradual burnout (Platform)        -> degrades from day 30
  - Rohan  : productivity theater (Platform)    -> high noise, low output, all 60 days
  - Sam    : quiet star (Platform)              -> low noise, high output, all 60 days
  - Arjun  : boreout / underutilized (Platform) -> low everything, all 60 days
  - Ananya : healthy baseline (Design)          -> stable normal, all 60 days
  - Karthik: sudden crisis (Engineering)        -> healthy then crashes day 50
The other 24 employees are healthy with mild natural variation.

A "reorg event" is planted on day 30 that affects the Platform team (this is
what kicks off Priya's degradation and dampens the rest of Platform).
"""

import csv
import random
from datetime import datetime, timedelta, time
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

START_DATE = datetime(2026, 2, 9)   # Monday
NUM_DAYS = 60
REORG_DAY = 30                       # day index when Platform reorg hits
OUTPUT_DIR = Path("pulseiq_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Teams and roles
# ----------------------------------------------------------------------------
TEAMS = {
    "Engineering": {
        "channels": ["#engineering", "#eng-help", "#code-review", "#random", "#general"],
        "ticket_prefix": "ENG",
        "size": 11,
    },
    "Design": {
        "channels": ["#design", "#design-critique", "#design-systems", "#random", "#general"],
        "ticket_prefix": "DES",
        "size": 9,
    },
    "Platform": {
        "channels": ["#platform-eng", "#incidents", "#infra", "#random", "#general"],
        "ticket_prefix": "PLAT",
        "size": 10,
    },
}

FIRST_NAMES = [
    "Aarav", "Vihaan", "Reyansh", "Kabir", "Ishaan", "Ayaan", "Krishna", "Rohan",
    "Arjun", "Karthik", "Aditya", "Dhruv", "Vivaan", "Sai", "Rudra",
    "Ananya", "Diya", "Saanvi", "Aanya", "Pari", "Kiara", "Myra", "Anika",
    "Priya", "Riya", "Aarohi", "Navya", "Sara", "Ira", "Tara", "Sam",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Kumar", "Singh", "Mehta", "Iyer",
    "Reddy", "Nair", "Rao", "Joshi", "Desai", "Kapoor", "Malhotra", "Bose",
    "Chopra", "Bhat", "Pillai", "Menon", "Okafor", "Khan", "Ahuja", "Sethi",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5

def jitter_minutes(base_hour: int, base_minute: int = 0, spread: int = 30) -> tuple:
    minutes_offset = random.randint(-spread, spread)
    total = base_hour * 60 + base_minute + minutes_offset
    total = max(0, min(24 * 60 - 1, total))
    return divmod(total, 60)

def random_text_length(short_bias: bool = False) -> int:
    if short_bias:
        return random.choice([8, 12, 15, 19, 22, 28, 34])
    return random.choice([19, 23, 34, 47, 56, 67, 78, 89, 102, 134, 156, 201, 234])

# ----------------------------------------------------------------------------
# Message text templates by sentiment bucket
# ----------------------------------------------------------------------------
# Each bucket has work-channel and social-channel variants. We pick from the
# bucket matching the persona's current sentiment_baseline so DistilBERT/RoBERTa
# will actually score them in the expected direction at inference time.
#
# Buckets:
#   positive  : sentiment >= 0.70  (warm, engaged, encouraging)
#   neutral   : 0.50 <= s < 0.70   (functional, professional, fine)
#   negative  : 0.30 <= s < 0.50   (clipped, frustrated, tired)
#   exhausted : s < 0.30           (depleted, hopeless, terse)

MESSAGE_TEMPLATES = {
    "positive": {
        "work": [
            "Awesome work on the deploy yesterday, the rollout was super smooth!",
            "Just merged the fix, thanks for the great review comments!",
            "Love this approach, much cleaner than what I had in mind",
            "Great catch on that edge case, would have missed it otherwise",
            "Nice, this unblocks me completely. Thanks for jumping on it!",
            "Just shipped PLAT-4392, feeling really good about the test coverage",
            "Excellent point in standup today, totally agree with the direction",
            "This is exactly what we needed, brilliant work everyone",
            "Happy to pair on this tomorrow if it would help",
            "Just finished the migration, all green and looking solid",
            "Really enjoyed the architecture discussion, learned a lot",
            "Big thanks to the team for pushing through on this one",
            "The new dashboard looks amazing, props to whoever designed it",
            "Glad we caught this early, good instinct on flagging it",
            "Loving the momentum this sprint, we're crushing it",
        ],
        "social": [
            "Anyone else watching the match tonight? Should be a good one",
            "Just tried that new ramen place downtown, ten out of ten",
            "Coffee in the kitchen if anyone wants some, freshly brewed",
            "Happy Friday everyone, hope you all have great weekends",
            "My dog learned a new trick this weekend, I'm a proud parent",
            "Great hike yesterday, anyone else into trails around here?",
            "Just finished that book everyone was recommending, loved it",
            "Anyone got good podcast recommendations for a long drive?",
            "Office plants are thriving this week, very pleased",
            "Just saw the funniest thing on the way in, made my morning",
        ],
    },
    "neutral": {
        "work": [
            "Pushed the changes to staging, can someone take a look when free",
            "Updated the ticket with the latest status, still in progress",
            "Will need another day on this, hitting some unexpected complexity",
            "Joining the call in 5, just wrapping up something",
            "Can we move the sync to 3pm instead, conflict with another meeting",
            "Reviewed the PR, left a few comments, nothing blocking",
            "Looking into the bug now, will update once I have more info",
            "Need to confirm this with the data team before proceeding",
            "Updated the doc with the latest decisions from yesterday",
            "Standup notes are in the channel if anyone missed it",
            "Will pick this up after lunch, currently context switching",
            "Need to think about this one a bit more, will respond later",
            "Not sure about that approach, lets discuss in the meeting",
            "Have a question about the spec, will DM you in a minute",
            "Working on it, should have something to show by EOD",
        ],
        "social": [
            "Anyone know a good lunch spot near the office",
            "Coffee machine is acting up again, fyi",
            "Reminder that the offsite is next month, need to book travel",
            "Did anyone leave a black hoodie in the conference room",
            "Heads up, fire drill tomorrow at 11",
        ],
    },
    "negative": {
        "work": [
            "This is taking way longer than expected, frustrated with the tooling",
            "Cant get the build to pass, been stuck on this for hours",
            "Need to push the deadline, theres no way this lands by Friday",
            "Another meeting got added to my calendar, when am I supposed to code",
            "Honestly this requirement keeps changing, hard to make progress",
            "Tried three approaches and none of them work, running out of ideas",
            "Will need to skip the sync today, drowning in tickets",
            "Dont have bandwidth for another review this week, sorry",
            "Cant make the meeting, will catch up async later",
            "Pushing PLAT-4421 to next sprint, no realistic way to finish",
            "Blocked again on the same issue from last week, getting old",
            "Late on this, sorry. Will try to get to it tonight",
            "Not going to make standup, too much on my plate this morning",
            "Need to step away from this for a bit, head is fried",
            "Quick reply, in the middle of something urgent right now",
        ],
        "social": [
            "Skipping coffee chat today, too much going on",
            "Cant make happy hour, will try next week",
            "Maybe later, drowning right now",
        ],
    },
    "exhausted": {
        "work": [
            "Still working on it",
            "Will look tomorrow",
            "Cant tonight, sorry",
            "K",
            "Noted",
            "Pushing to next week, no choice",
            "Need to drop off, will catch up later",
            "Not today, too much already",
            "Sorry for late reply, swamped",
            "Will try, no promises",
            "Cant deal with this right now",
            "Just merged it, going to log off",
            "Done for the day, will handle tomorrow",
            "Sorry, missed this. Looking now",
            "Late again, apologies",
        ],
        "social": [
            "Cant",
            "Maybe",
            "Sorry no",
        ],
    },
}

def sentiment_to_bucket(score: float) -> str:
    if score >= 0.70: return "positive"
    if score >= 0.50: return "neutral"
    if score >= 0.30: return "negative"
    return "exhausted"

def pick_message_text(sentiment: float, channel: str) -> str:
    bucket = sentiment_to_bucket(sentiment)
    is_social = channel in ("#random", "#general")
    pool_key = "social" if is_social else "work"
    pool = MESSAGE_TEMPLATES[bucket][pool_key]
    if not pool:
        pool = MESSAGE_TEMPLATES[bucket]["work"]
    return random.choice(pool)

# ----------------------------------------------------------------------------
# Persona definitions
# ----------------------------------------------------------------------------
# Each persona is a function that takes (day_index, day_date, employee, context)
# and returns counts/parameters that drive the daily generation.
#
# Core daily knobs:
#   msg_count_work, msg_count_social, msg_count_dms
#   after_hours_msg_ratio (0..1), weekend_msg_count
#   distinct_dm_partners
#   meetings_count, meeting_chain_chance
#   tickets_touched, tickets_closed_today, story_points_closed
#   sentiment_baseline (0..1, higher = more positive)
# ----------------------------------------------------------------------------

def healthy_profile(day_idx, dt, emp, ctx):
    """Default healthy worker."""
    weekday = dt.weekday()
    if weekday >= 5:  # weekend
        return dict(
            msg_count_work=0, msg_count_social=0, msg_count_dms=0,
            after_hours_msg_ratio=0.0, weekend_msg_count=0,
            distinct_dm_partners=0,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=0, tickets_closed_today=0, story_points_closed=0,
            sentiment_baseline=0.7,
        )
    return dict(
        msg_count_work=random.randint(4, 8),
        msg_count_social=random.randint(1, 3),
        msg_count_dms=random.randint(2, 5),
        after_hours_msg_ratio=random.uniform(0.0, 0.08),
        weekend_msg_count=0,
        distinct_dm_partners=random.randint(2, 4),
        meetings_count=random.randint(2, 4),
        meeting_chain_chance=0.2,
        tickets_touched=random.randint(1, 3),
        tickets_closed_today=1 if random.random() < 0.4 else 0,
        story_points_closed=random.choice([0, 0, 3, 5]),
        sentiment_baseline=random.uniform(0.6, 0.8),
    )

def priya_profile(day_idx, dt, emp, ctx):
    """Gradual burnout starting at REORG_DAY."""
    if is_weekend(dt):
        # Weekend work creeps in after the reorg
        if day_idx < REORG_DAY:
            return healthy_profile(day_idx, dt, emp, ctx)
        weekend_intensity = min(1.0, (day_idx - REORG_DAY) / 25)
        return dict(
            msg_count_work=int(3 * weekend_intensity),
            msg_count_social=0,
            msg_count_dms=int(2 * weekend_intensity),
            after_hours_msg_ratio=0.5,
            weekend_msg_count=int(5 * weekend_intensity),
            distinct_dm_partners=1,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=int(2 * weekend_intensity),
            tickets_closed_today=0,
            story_points_closed=0,
            sentiment_baseline=0.4,
        )

    # Pre-reorg: healthy
    if day_idx < REORG_DAY:
        base = healthy_profile(day_idx, dt, emp, ctx)
        base["sentiment_baseline"] = random.uniform(0.65, 0.8)
        return base

    # Post-reorg: gradual decline over 30 days
    intensity = min(1.0, (day_idx - REORG_DAY) / 30)
    return dict(
        msg_count_work=random.randint(3, 6),  # similar work msg count
        msg_count_social=max(0, int(2 * (1 - intensity))),  # social drops fast
        msg_count_dms=max(1, int(4 * (1 - intensity * 0.7))),
        after_hours_msg_ratio=0.1 + 0.5 * intensity,  # ramps to 60%
        weekend_msg_count=0,
        distinct_dm_partners=max(1, int(4 * (1 - intensity * 0.75))),  # graph shrinks
        meetings_count=random.randint(4, 7),  # meetings stay high
        meeting_chain_chance=0.5 + 0.3 * intensity,
        tickets_touched=random.randint(1, 3),
        tickets_closed_today=1 if random.random() < 0.3 - 0.2 * intensity else 0,
        story_points_closed=random.choice([0, 0, 3]) if random.random() < 0.5 else 0,
        sentiment_baseline=max(0.2, 0.75 - 0.5 * intensity),
    )

def rohan_profile(day_idx, dt, emp, ctx):
    """Productivity theater. Manager-coded. High noise, low output, no overtime."""
    if is_weekend(dt):
        return dict(
            msg_count_work=0, msg_count_social=0, msg_count_dms=0,
            after_hours_msg_ratio=0.0, weekend_msg_count=0,
            distinct_dm_partners=0,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=0, tickets_closed_today=0, story_points_closed=0,
            sentiment_baseline=0.7,
        )
    return dict(
        msg_count_work=random.randint(15, 22),
        msg_count_social=random.randint(3, 6),
        msg_count_dms=random.randint(8, 12),
        after_hours_msg_ratio=0.0,  # he goes home at 6
        weekend_msg_count=0,
        distinct_dm_partners=random.randint(6, 9),
        meetings_count=random.randint(8, 11),  # always in meetings
        meeting_chain_chance=0.85,
        tickets_touched=1 if random.random() < 0.3 else 0,
        tickets_closed_today=1 if random.random() < 0.1 else 0,
        story_points_closed=random.choice([0, 0, 0, 0, 3]),
        sentiment_baseline=0.7,
    )

def sam_profile(day_idx, dt, emp, ctx):
    """Quiet star. Low noise, high output."""
    if is_weekend(dt):
        return dict(
            msg_count_work=0, msg_count_social=0, msg_count_dms=0,
            after_hours_msg_ratio=0.0, weekend_msg_count=0,
            distinct_dm_partners=0,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=0, tickets_closed_today=0, story_points_closed=0,
            sentiment_baseline=0.8,
        )
    return dict(
        msg_count_work=random.randint(2, 4),
        msg_count_social=random.randint(0, 2),
        msg_count_dms=random.randint(1, 3),
        after_hours_msg_ratio=random.uniform(0.0, 0.05),
        weekend_msg_count=0,
        distinct_dm_partners=random.randint(2, 4),
        meetings_count=random.randint(1, 3),  # protected calendar
        meeting_chain_chance=0.1,
        tickets_touched=random.randint(2, 4),
        tickets_closed_today=1 if random.random() < 0.7 else 0,
        story_points_closed=random.choice([3, 5, 5, 8]),
        sentiment_baseline=random.uniform(0.7, 0.85),
    )

def arjun_profile(day_idx, dt, emp, ctx):
    """Boreout. Underutilized. Low everything."""
    if is_weekend(dt):
        return dict(
            msg_count_work=0, msg_count_social=0, msg_count_dms=0,
            after_hours_msg_ratio=0.0, weekend_msg_count=0,
            distinct_dm_partners=0,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=0, tickets_closed_today=0, story_points_closed=0,
            sentiment_baseline=0.5,
        )
    return dict(
        msg_count_work=random.randint(1, 2),
        msg_count_social=0,
        msg_count_dms=1 if random.random() < 0.3 else 0,
        after_hours_msg_ratio=0.0,
        weekend_msg_count=0,
        distinct_dm_partners=1,  # only DMs manager
        meetings_count=random.randint(1, 2),
        meeting_chain_chance=0.0,
        tickets_touched=1 if random.random() < 0.4 else 0,
        tickets_closed_today=1 if random.random() < 0.15 else 0,
        story_points_closed=random.choice([0, 0, 0, 2]),
        sentiment_baseline=random.uniform(0.4, 0.55),
    )

def ananya_profile(day_idx, dt, emp, ctx):
    """Healthy baseline reference. Stable forever."""
    if is_weekend(dt):
        return dict(
            msg_count_work=0, msg_count_social=0, msg_count_dms=0,
            after_hours_msg_ratio=0.0, weekend_msg_count=0,
            distinct_dm_partners=0,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=0, tickets_closed_today=0, story_points_closed=0,
            sentiment_baseline=0.8,
        )
    return dict(
        msg_count_work=random.randint(5, 8),
        msg_count_social=random.randint(1, 3),
        msg_count_dms=random.randint(3, 5),
        after_hours_msg_ratio=random.uniform(0.0, 0.05),
        weekend_msg_count=0,
        distinct_dm_partners=random.randint(3, 5),
        meetings_count=random.randint(2, 3),
        meeting_chain_chance=0.15,
        tickets_touched=random.randint(1, 3),
        tickets_closed_today=1 if random.random() < 0.5 else 0,
        story_points_closed=random.choice([0, 3, 5]),
        sentiment_baseline=random.uniform(0.7, 0.85),
    )

def karthik_profile(day_idx, dt, emp, ctx):
    """Sudden crisis. Healthy until day 50, then crashes hard."""
    if day_idx < 50:
        return healthy_profile(day_idx, dt, emp, ctx)
    if is_weekend(dt):
        return dict(
            msg_count_work=2, msg_count_social=0, msg_count_dms=1,
            after_hours_msg_ratio=0.7, weekend_msg_count=4,
            distinct_dm_partners=1,
            meetings_count=0, meeting_chain_chance=0.0,
            tickets_touched=1, tickets_closed_today=0, story_points_closed=0,
            sentiment_baseline=0.2,
        )
    return dict(
        msg_count_work=random.randint(2, 4),
        msg_count_social=0,
        msg_count_dms=1,
        after_hours_msg_ratio=0.6,
        weekend_msg_count=0,
        distinct_dm_partners=1,
        meetings_count=random.randint(3, 5),
        meeting_chain_chance=0.7,
        tickets_touched=random.randint(0, 2),
        tickets_closed_today=0,
        story_points_closed=0,
        sentiment_baseline=random.uniform(0.15, 0.3),
    )

PERSONA_FUNCS = {
    "priya": priya_profile,
    "rohan": rohan_profile,
    "sam": sam_profile,
    "arjun": arjun_profile,
    "ananya": ananya_profile,
    "karthik": karthik_profile,
    "healthy": healthy_profile,
}

# ----------------------------------------------------------------------------
# Build the employee roster
# ----------------------------------------------------------------------------
def build_employees():
    employees = []
    eid = 1

    # Planted personas (fixed names + teams)
    planted = [
        ("Priya",   "Sharma",  "Platform",    "Senior Backend Engineer", "priya"),
        ("Rohan",   "Mehta",   "Platform",    "Engineering Manager",     "rohan"),
        ("Sam",     "Okafor",  "Platform",    "Backend Engineer",        "sam"),
        ("Arjun",   "Patel",   "Platform",    "Backend Engineer",        "arjun"),
        ("Ananya",  "Iyer",    "Design",      "Senior Designer",         "ananya"),
        ("Karthik", "Reddy",   "Engineering", "Software Engineer",       "karthik"),
    ]
    for first, last, team, role, persona in planted:
        employees.append({
            "employee_id": f"E{eid:03d}",
            "name": f"{first} {last}",
            "team": team,
            "role": role,
            "persona": persona,
        })
        eid += 1

    # Fill in the rest with healthy folks
    used_names = {(e["name"]) for e in employees}
    team_counts = {t: sum(1 for e in employees if e["team"] == t) for t in TEAMS}

    for team, info in TEAMS.items():
        needed = info["size"] - team_counts[team]
        for _ in range(needed):
            while True:
                name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            role = "Engineer" if team != "Design" else "Designer"
            employees.append({
                "employee_id": f"E{eid:03d}",
                "name": name,
                "team": team,
                "role": role,
                "persona": "healthy",
            })
            eid += 1

    return employees

# ----------------------------------------------------------------------------
# Day-by-day event generation
# ----------------------------------------------------------------------------
def generate_slack_events(emp, dt, profile, ctx, writer, msg_id_counter):
    team_channels = TEAMS[emp["team"]]["channels"]
    work_channels = [c for c in team_channels if c not in ("#random", "#general")]
    social_channels = ["#random", "#general"]

    # Pick DM partners deterministically per day for the social graph
    dm_partners = ctx["dm_pool"][emp["employee_id"]][:profile["distinct_dm_partners"]]

    def emit(channel, is_dm, recipient, ts):
        msg_id_counter[0] += 1
        # Sample sentiment around the persona's daily baseline
        sentiment = max(0.0, min(1.0, profile["sentiment_baseline"] + random.uniform(-0.1, 0.1)))
        # DMs to a single trusted person tend slightly warmer; weekend pings tend cooler
        if is_dm and ts.hour < 21:
            sentiment = min(1.0, sentiment + 0.05)
        if ts.weekday() >= 5 or ts.hour >= 22:
            sentiment = max(0.0, sentiment - 0.1)
        text = pick_message_text(sentiment, "dm" if is_dm else channel)
        writer.writerow([
            f"M{msg_id_counter[0]:07d}",
            emp["employee_id"],
            fmt_ts(ts),
            channel,
            len(text),
            "true" if is_dm else "false",
            recipient or "",
            round(sentiment, 2),
            text,
        ])

    # Work channel messages
    for _ in range(profile["msg_count_work"]):
        is_after_hours = random.random() < profile["after_hours_msg_ratio"]
        if is_after_hours:
            h, m = jitter_minutes(random.choice([21, 22, 23]), 0, 25)
        else:
            h, m = jitter_minutes(random.choice([9, 10, 11, 13, 14, 15, 16]), 0, 25)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        emit(random.choice(work_channels), False, None, ts)

    # Social channel messages
    for _ in range(profile["msg_count_social"]):
        h, m = jitter_minutes(random.choice([10, 12, 15, 16]), 0, 30)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        emit(random.choice(social_channels), False, None, ts)

    # DMs
    for _ in range(profile["msg_count_dms"]):
        if not dm_partners:
            break
        recipient = random.choice(dm_partners)
        is_after_hours = random.random() < profile["after_hours_msg_ratio"]
        if is_after_hours:
            h, m = jitter_minutes(random.choice([21, 22, 23]), 0, 25)
        else:
            h, m = jitter_minutes(random.choice([9, 11, 14, 16]), 0, 30)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        emit("dm", True, recipient, ts)

    # Weekend work
    for _ in range(profile["weekend_msg_count"]):
        h, m = jitter_minutes(random.choice([10, 14, 20, 22]), 0, 60)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        emit(random.choice(work_channels), False, None, ts)

def generate_jira_events(emp, dt, profile, ctx, writer, ticket_counter):
    prefix = TEAMS[emp["team"]]["ticket_prefix"]

    # Pick or create a ticket the person is working on
    active = ctx["active_tickets"].setdefault(emp["employee_id"], [])

    # Maybe start a new ticket
    if profile["tickets_touched"] > 0 and (not active or random.random() < 0.3):
        ticket_counter[0] += 1
        new_ticket = {
            "id": f"{prefix}-{ticket_counter[0]:04d}",
            "story_points": random.choice([2, 3, 5, 5, 8]),
            "status": "In Progress",
        }
        active.append(new_ticket)
        h, m = jitter_minutes(10, 0, 60)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        writer.writerow([
            emp["employee_id"], fmt_ts(ts), new_ticket["id"],
            "status_change", new_ticket["story_points"], "To Do", "In Progress",
        ])

    # Touch existing tickets (comments / movement)
    for _ in range(max(0, profile["tickets_touched"] - 1)):
        if not active:
            break
        t = random.choice(active)
        h, m = jitter_minutes(random.choice([11, 14, 16]), 0, 60)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        writer.writerow([
            emp["employee_id"], fmt_ts(ts), t["id"],
            "comment", t["story_points"], t["status"], t["status"],
        ])

    # Close tickets
    if profile["tickets_closed_today"] > 0 and active:
        t = active.pop(0)
        h, m = jitter_minutes(random.choice([15, 16, 17]), 0, 30)
        ts = dt.replace(hour=h, minute=m, second=random.randint(0, 59))
        writer.writerow([
            emp["employee_id"], fmt_ts(ts), t["id"],
            "status_change", t["story_points"], t["status"], "Done",
        ])

def generate_calendar_events(emp, dt, profile, writer, meeting_id_counter):
    n = profile["meetings_count"]
    if n == 0:
        return

    # Standup at 10:00 always exists if any meetings
    used_slots = []

    def add_meeting(start_h, start_m, duration_min, title, attendees):
        meeting_id_counter[0] += 1
        start = dt.replace(hour=start_h, minute=start_m, second=0)
        end = start + timedelta(minutes=duration_min)
        writer.writerow([
            f"MTG{meeting_id_counter[0]:06d}",
            emp["employee_id"],
            dt.strftime("%Y-%m-%d"),
            start.strftime("%H:%M"),
            end.strftime("%H:%M"),
            title,
            attendees,
        ])
        used_slots.append((start_h * 60 + start_m, start_h * 60 + start_m + duration_min))

    # Standup
    add_meeting(10, 0, 30, "Standup", random.randint(4, 8))
    n -= 1

    # Other meetings
    chain = profile["meeting_chain_chance"]
    candidate_starts = [(11, 0), (11, 30), (13, 0), (14, 0), (14, 30),
                        (15, 0), (15, 30), (16, 0), (16, 30), (17, 0)]
    random.shuffle(candidate_starts)
    titles = ["1:1", "Sync", "Review", "Planning", "Cross-team", "Critique",
              "Postmortem", "Vendor Call", "All-Hands", "Hiring Loop", "Strategy"]

    for h, m in candidate_starts:
        if n <= 0:
            break
        slot_start = h * 60 + m
        duration = random.choice([30, 30, 45, 60, 60])
        slot_end = slot_start + duration
        # Check overlap
        if any(not (slot_end <= s or slot_start >= e) for s, e in used_slots):
            continue
        # Chain bias: if last meeting ended recently, prefer this slot
        if used_slots:
            last_end = max(e for _, e in used_slots)
            if slot_start - last_end < 30 and random.random() > chain:
                continue
        add_meeting(h, m, duration, random.choice(titles), random.randint(2, 12))
        n -= 1

# ----------------------------------------------------------------------------
# Main generation loop
# ----------------------------------------------------------------------------
def main():
    employees = build_employees()
    print(f"Generated {len(employees)} employees")

    # Write employees.csv
    with open(OUTPUT_DIR / "employees.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["employee_id", "name", "team", "role", "persona"])
        for e in employees:
            w.writerow([e["employee_id"], e["name"], e["team"], e["role"], e["persona"]])

    # Pre-build a DM pool for each employee (their potential social graph)
    all_ids = [e["employee_id"] for e in employees]
    dm_pool = {}
    for e in employees:
        same_team = [x["employee_id"] for x in employees
                     if x["team"] == e["team"] and x["employee_id"] != e["employee_id"]]
        random.shuffle(same_team)
        cross_team = [x for x in all_ids if x != e["employee_id"] and x not in same_team]
        random.shuffle(cross_team)
        dm_pool[e["employee_id"]] = same_team[:6] + cross_team[:3]

    ctx = {
        "dm_pool": dm_pool,
        "active_tickets": {},
    }

    msg_id_counter = [0]
    ticket_counter = [4000]
    meeting_id_counter = [0]

    slack_f = open(OUTPUT_DIR / "slack_messages.csv", "w", newline="")
    jira_f = open(OUTPUT_DIR / "jira_events.csv", "w", newline="")
    cal_f = open(OUTPUT_DIR / "calendar_events.csv", "w", newline="")

    slack_w = csv.writer(slack_f)
    jira_w = csv.writer(jira_f)
    cal_w = csv.writer(cal_f)

    slack_w.writerow(["message_id", "employee_id", "timestamp", "channel",
                      "message_length", "is_dm", "recipient_id", "sentiment_score",
                      "message_text"])
    jira_w.writerow(["employee_id", "timestamp", "ticket_id", "event_type",
                     "story_points", "from_status", "to_status"])
    cal_w.writerow(["meeting_id", "employee_id", "date", "start_time", "end_time",
                    "title", "attendee_count"])

    for day_idx in range(NUM_DAYS):
        dt = START_DATE + timedelta(days=day_idx)
        for emp in employees:
            persona_func = PERSONA_FUNCS[emp["persona"]]
            profile = persona_func(day_idx, dt, emp, ctx)
            generate_slack_events(emp, dt, profile, ctx, slack_w, msg_id_counter)
            generate_jira_events(emp, dt, profile, ctx, jira_w, ticket_counter)
            generate_calendar_events(emp, dt, profile, cal_w, meeting_id_counter)

    slack_f.close()
    jira_f.close()
    cal_f.close()

    # Quick stats
    print(f"\nFiles written to {OUTPUT_DIR}/")
    for fname in ["employees.csv", "slack_messages.csv", "jira_events.csv", "calendar_events.csv"]:
        path = OUTPUT_DIR / fname
        line_count = sum(1 for _ in open(path)) - 1
        print(f"  {fname:25s} {line_count:>8,} rows")

if __name__ == "__main__":
    main()
