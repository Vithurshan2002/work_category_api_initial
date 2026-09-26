from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]

# ஏற்கெனவே train செய்த model-ஐ load செய்கிறோம்.
model = joblib.load(ROOT / "models/category_model.joblib")

# புதிய test requests-ஐப் படிக்கிறோம்.
df = pd.read_csv(ROOT / "data/new_test_requests.csv")

if not {"text", "category"}.issubset(df.columns):
    raise ValueError("CSV file-இல் text,category columns வேண்டும்")

if df.empty or df[["text", "category"]].isna().any().any():
    raise ValueError("CSV file-இல் empty rows அல்லது values இருக்கக்கூடாது")

# புதிய requests-க்கு model பதில் சொல்கிறது.
df["predicted_category"] = model.predict(df["text"])

# சரியான பதில்களுடன் ஒப்பிடுகிறோம்.
print("Total requests:", len(df))
print("Correct predictions:", (df["category"] == df["predicted_category"]).sum())
print("Accuracy:", accuracy_score(df["category"], df["predicted_category"]))

print(classification_report(
    df["category"],
    df["predicted_category"],
    zero_division=0,
))

# ஒவ்வொரு request-க்கும் கிடைத்த பதிலை file-இல் சேமிக்கிறோம்.
report_dir = ROOT / "reports"
report_dir.mkdir(exist_ok=True)

df.to_csv(report_dir / "new_test_predictions.csv", index=False)

print("Results saved: reports/new_test_predictions.csv")



# சரியான category-யும் model சொன்ன category-யும் வேறாக உள்ள rows.
wrong_predictions = df[
    df["category"] != df["predicted_category"]
]

# தவறான predictions மட்டும் தனி CSV-இல் சேமிக்கிறோம்.
wrong_predictions.to_csv(
    report_dir / "new_test_wrong_predictions.csv",
    index=False,
)

print("Wrong predictions:", len(wrong_predictions))
print("Saved: reports/new_test_wrong_predictions.csv")