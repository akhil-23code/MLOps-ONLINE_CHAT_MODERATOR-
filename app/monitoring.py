import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import TextOverviewPreset, DataDriftPreset, TargetDriftPreset
from pathlib import Path

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
REPORT_PATH = BASE_DIR / "reports"
REPORT_PATH.mkdir(exist_ok=True)

def generate_monitoring_report():
    print("📊 Starting Evidently AI Monitoring Report...")

    # 2. Load Reference Data (The data you trained on)
    # Ensure you have this file in mlops_local/data/reference_data.csv
    try:
        ref_data = pd.read_csv(DATA_PATH / "reference_data.csv")
    except FileNotFoundError:
        print("❌ Reference data not found. Creating dummy data for demonstration...")
        ref_data = pd.DataFrame({
            "text": ["I love this!", "This is bad", "Have a nice day"],
            "label": [2, 1, 2] # 2: Neither, 1: Offensive
        })

    # 3. Simulate "Current" Data (Data coming into your API)
    # In a real app, you would pull this from your database/logs
    curr_data = pd.DataFrame({
        "text": ["You are amazing", "Shut up", "I hate everything"],
        "label": [2, 1, 0] # 0: Hate Speech
    })

    # 4. Create the Report
    # We use 'TextOverviewPreset' because this is an NLP project
    toxic_report = Report(metrics=[
        TextOverviewPreset(column_name="text"),
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    toxic_report.run(reference_data=ref_data, current_data=curr_data)

    # 5. Save to HTML
    report_file = REPORT_PATH / "monitoring_report.html"
    toxic_report.save_html(str(report_file))
    
    print(f"✅ Report generated: {report_file}")

if __name__ == "__main__":
    generate_monitoring_report()