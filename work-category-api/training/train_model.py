from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]

train_df = pd.read_csv(ROOT / "data/processed/train.csv")
validation_df = pd.read_csv(ROOT / "data/processed/validation.csv")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        ),
    ),
    (
        "classifier",
        LinearSVC(
            C=1.0,
            random_state=42,
        ),
    ),
])

# Training text மற்றும் correct labels மூலம் கற்றுக்கொள்கிறது.
model.fit(
    train_df["text"],
    train_df["category"],
)

# கற்றுக்கொள்ள பயன்படுத்தாத validation requests-ஐ predict செய்கிறது.
predictions = model.predict(validation_df["text"])

print(
    classification_report(
        validation_df["category"],
        predictions,
        zero_division=0,
    )
)

# TF-IDF + SVM இரண்டும் சேர்ந்த pipeline save ஆகிறது.
model_dir = ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "category_model.joblib")

print("Model saved successfully.")