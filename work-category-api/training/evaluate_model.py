from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(ROOT / "models/category_model.joblib")
test_df = pd.read_csv(ROOT / "data/processed/test.csv")

predictions = model.predict(test_df["text"])

print("Test accuracy:", accuracy_score(
    test_df["category"],
    predictions,
))

print(classification_report(
    test_df["category"],
    predictions,
    zero_division=0,
))

# தவறான predictions-ஐ ஆய்வு செய்ய save செய்கிறோம்.
results = test_df.copy()
results["predicted_category"] = predictions

wrong_predictions = results[
    results["category"] != results["predicted_category"]
]

report_dir = ROOT / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

wrong_predictions.to_csv(
    report_dir / "wrong_predictions.csv",
    index=False,
)