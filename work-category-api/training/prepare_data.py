from pathlib import Path
import pandas as pd

# இந்த file இருக்கும் இடத்திலிருந்து project root கண்டுபிடிக்கிறது.
ROOT = Path(__file__).resolve().parents[1]

# Terminal எந்த folder-இல் இருந்தாலும் சரியான CSV path கிடைக்கும்.
data_path = ROOT / "data" / "raw" / "service_requests.csv"

df = pd.read_csv(data_path)

# Text அல்லது category இல்லாத rows நீக்கப்படுகின்றன.
df = df.dropna(subset=["text", "category"])

# Text-இல் தேவையற்ற spaces மற்றும் case வேறுபாடுகள் சரிசெய்யப்படுகின்றன.
df["text"] = (
    df["text"]
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

df["category"] = df["category"].str.strip()

# Empty strings உள்ள rows நீக்கப்படுகின்றன.
df = df[(df["text"] != "") & (df["category"] != "")]

# ஒரே text-க்கு ஒன்றுக்கு மேற்பட்ட labels இருக்கிறதா?
label_counts = df.groupby("text")["category"].nunique()

if (label_counts > 1).any():
    raise ValueError("Same text has conflicting labels. Fix the CSV.")

df = df.drop_duplicates(subset=["text"])

print(df["category"].value_counts())


from sklearn.model_selection import train_test_split

# 70% training; remaining 30% validation + test.
train_df, remaining_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["category"],
)

# Remaining 30%-ஐ இரண்டாகப் பிரிக்கிறது: 15% + 15%.
validation_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    stratify=remaining_df["category"],
)

output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
validation_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)