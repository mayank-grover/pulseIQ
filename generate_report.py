import pandas as pd
from datetime import timedelta
from pathlib import Path

DATA_DIR = Path("pulseiq_data")
INPUT_FILE = DATA_DIR / "daily_scores.csv"
OUTPUT_FILE = DATA_DIR / "pulseiq_summary.csv"

def generate_summary():
    if not INPUT_FILE.exists(): return
    df = pd.read_csv(INPUT_FILE)

    # Get latest snapshot per employee using the newly fixed day_index
    latest_df = df.sort_values('day_index').groupby('employee_id').tail(1).copy()

    def calculate_crash_date(row):
        current_date = pd.to_datetime(row['date'])
        days_rem = row['forecast_days_until_critical']
        if row['burnout_probability'] >= 0.85: return current_date.strftime('%Y-%m-%d')
        if not pd.isna(days_rem) and days_rem > 0:
            return (current_date + timedelta(days=int(days_rem))).strftime('%Y-%m-%d')
        return "N/A"

    latest_df['forecasted_crash_date'] = latest_df.apply(calculate_crash_date, axis=1)

    # Columns matching the "Final Boss" math
    summary_cols = [
        'employee_id', 'name', 'deep_work_index', 'fragmentation_score',
        'connection_index', 'recovery_debt', 'rest_deficit',
        'burnout_probability', 'forecasted_crash_date'
    ]
    summary_df = latest_df[summary_cols].copy()
    summary_df['burnout_probability'] = (summary_df['burnout_probability'] * 100).round(1).astype(str) + '%'

    summary_df.columns = [
        'ID', 'Employee Name', 'Deep Work', 'Fragmentation', 'Connection',
        'Recovery Debt', 'Rest Deficit', 'Burnout Risk', 'Crash Date'
    ]

    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success: Report generated. Top Risk:\n{summary_df.sort_values('Burnout Risk', ascending=False).head(5)}")

if __name__ == "__main__": generate_summary()