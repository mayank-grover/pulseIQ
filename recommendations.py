import csv

INPUT_FILE = "pulseiq_data/daily_scores.csv"
OUTPUT_FILE = "pulseiq_data/recommendations.csv"


def generate_recommendations(row):
    recs = []

    # Safe parsing
    deep = float(row["deep_work_index"] or 0)
    frag = float(row["fragmentation_score"] or 0)
    recov = float(row["recovery_debt"] or 0)
    burnout = float(row["burnout_probability"] or 0)

    factors = row.get("driving_factors", "")
    factors = factors.split(",") if factors else []

    # -----------------------------
    # 1. RECOVERY (Fatigue)
    # -----------------------------
    if "recovery_debt" in factors or recov > 5:
        recs.append("⚠️ Recovery debt is high → reduce workload and take longer breaks")
        recs.append("💤 Avoid late-night work and prioritize recovery")

    # -----------------------------
    # 2. FRAGMENTATION (Interruptions)
    # -----------------------------
    if "fragmentation" in factors or frag > 60:
        recs.append("📵 High interruptions → block focus time and mute Slack")
        recs.append("📅 Reduce meetings or create no-meeting windows")

    # -----------------------------
    # 3. DEEP WORK (Focus)
    # -----------------------------
    if "deep_work" in factors:
        recs.append("🧠 Low deep work → schedule 60–90 min uninterrupted sessions")
    elif deep > 70 and frag < 40:
        recs.append("🔥 Strong focus today → schedule high-impact work")

    # -----------------------------
    # 4. SOCIAL / CONNECTION
    # -----------------------------
    if "connection" in factors:
        recs.append("🤝 Low collaboration → check in with team or manager")

    # -----------------------------
    # 5. BURNOUT RISK
    # -----------------------------
    if burnout > 0.7:
        recs.append("🛑 High burnout risk → avoid overtime and reschedule non-critical work")
    elif burnout > 0.5:
        recs.append("⚠️ Rising burnout risk → take breaks and reduce workload intensity")

    # -----------------------------
    # 6. FORECAST (Future Warning)
    # -----------------------------
    forecast = row.get("forecast_days_until_critical")
    if forecast and forecast not in ("", "None"):
        try:
            days = int(float(forecast))
            if days <= 5:
                recs.append(f"⏳ Burnout risk escalating → critical in ~{days} days")
        except:
            pass

    # -----------------------------
    # 7. DEFAULT (No Issues)
    # -----------------------------
    if not recs:
        recs.append("✅ Normal day → maintain current pace and schedule focus work")

    return recs


# -----------------------------
# MAIN FUNCTION (THIS WAS MISSING)
# -----------------------------
def main():
    with open(INPUT_FILE) as f:
        rows = list(csv.DictReader(f))

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["employee_id", "date", "recommendations"])

        for r in rows:
            recs = generate_recommendations(r)

            # Debug print (optional)
            print(f"{r['employee_id']} ({r['date']}):", recs)

            writer.writerow([
                r["employee_id"],
                r["date"],
                " • ".join(recs)   # ✅ Clean formatting
            ])

    print(f"\nRecommendations saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
