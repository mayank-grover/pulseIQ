import csv

INPUT_FILE = "pulseiq_data/daily_scores.csv"
OUTPUT_FILE = "pulseiq_data/recommendations.csv"


def generate_recommendations(row):
    recs = []

    deep = float(row["deep_work_index"] or 0)
    frag = float(row["fragmentation_score"] or 0)
    recov = float(row["recovery_debt"] or 0)
    burnout = float(row["burnout_probability"] or 0)

    # Recovery
    if recov > 5:
        recs.append("⚠️ High recovery debt → reduce workload")

    # Deep work
    if deep > 70 and frag < 40:
        recs.append("🔥 Strong focus → schedule deep work (90 mins)")
    elif deep < 40:
        recs.append("Low focus → keep tasks short")

    # Fragmentation
    if frag > 60:
        recs.append("Too many interruptions → block focus time")

    # Burnout
    if burnout and burnout > 0.7:
        recs.append("🛑 High burnout risk → avoid meetings & overtime")

    return recs


def main():
    with open(INPUT_FILE) as f:
        rows = list(csv.DictReader(f))

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["employee_id", "date", "recommendations"])

        for r in rows:
            recs = generate_recommendations(r)
            writer.writerow([
                r["employee_id"],
                r["date"],
                " | ".join(recs)
            ])

    print(f"\nRecommendations saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()