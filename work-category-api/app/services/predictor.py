from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]


class CategoryPredictor:
    def __init__(self):
        model_path = ROOT / "models/category_model.joblib"

        # Saved pipeline memory-இல் load ஆகிறது.
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        # Training cleaning-க்கு ஏற்ப text normalize செய்கிறோம்.
        cleaned_text = " ".join(text.lower().split())

        # predict() ஒரு list of texts எதிர்பார்க்கிறது.
        prediction = self.model.predict([cleaned_text])

        # முதல் request-க்கான category return செய்கிறோம்.
        return str(prediction[0])