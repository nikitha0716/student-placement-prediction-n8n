import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("data/data.csv")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================
# 2. FEATURES AND TARGET
# =========================

features = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "workex",
    "etest_p",
    "specialisation",
    "mba_p"
]

X = df[features]

y = df["status"].map({
    "Placed": 1,
    "Not Placed": 0
})


# =========================
# 3. PREPROCESSING
# =========================

numeric_features = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p"
]

categorical_features = [
    "workex",
    "specialisation"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================
# 4. RANDOM FOREST
# =========================

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", random_forest)
])


# =========================
# 5. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 6. TRAIN MODEL
# =========================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")


# =========================
# 7. TEST MODEL
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("       RANDOM FOREST RESULTS")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =========================
# 8. SAVE MODEL
# =========================

joblib.dump(
    model,
    "model/placement_model.pkl"
)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")

print("\nFile: model/placement_model.pkl")