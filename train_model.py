import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_thermoplastic_extrusion_quality_dataset.csv"
MODEL1_PATH = BASE_DIR / "quality_status_model.pkl"
MODEL2_PATH = BASE_DIR / "defect_type_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"


def load_and_preprocess_data() -> pd.DataFrame:
    """Load the dataset and apply the existing preprocessing steps."""
    df = pd.read_csv(DATA_PATH)

    df["Braid_Type_Planned"] = df["Braid_Type_Planned"].fillna("None")
    df["Vacuum_bar"] = df["Vacuum_bar"].fillna(df["Vacuum_bar"].mean())
    df["Melt_Pressure_bar"] = df["Melt_Pressure_bar"].fillna(
        df["Melt_Pressure_bar"].mean()
    )
    df["Cooling_Water_Temp_C"] = df["Cooling_Water_Temp_C"].fillna(
        df["Cooling_Water_Temp_C"].mean()
    )
    df["Material_Moisture_pct"] = df["Material_Moisture_pct"].fillna(
        df["Material_Moisture_pct"].mean()
    )
    df["Defect_Type"] = df["Defect_Type"].fillna("Unknown")

    df["Machine_ID"] = df["Machine_ID"].map(
        {"EXT-01": 0, "EXT-02": 1, "EXT-03": 2}
    )
    df["Shift"] = df["Shift"].map({"A": 0, "B": 1, "C": 2})
    df["Material_Type"] = df["Material_Type"].map(
        {"PU": 0, "PA11": 1, "Hytrel": 2, "PA12": 3, "PVDF": 4}
    )
    df["Cover_Required"] = df["Cover_Required"].map({"No": 0, "Yes": 1})
    df["Braid_Type_Planned"] = df["Braid_Type_Planned"].map(
        {"None": 0, "Aramid": 1, "SS Wire": 2, "Polyester": 3}
    )
    df["Defect_Status"] = df["Defect_Status"].map({"Fail": 0, "Pass": 1})

    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    """Return the feature list used by the models."""
    features = [
        "Braid_Type_Planned",
        "Material_Type",
        "Machine_ID",
        "Target_Tube_OD_mm",
        "Target_Tube_ID_mm",
        "Target_Wall_Thickness_mm",
        "OD_Tolerance_mm",
        "ID_Tolerance_mm",
        "Wall_Tolerance_mm",
        "Vacuum_bar",
        "Screw_RPM",
        "Zone2_Temp_C",
        "Zone3_Temp_C",
        "Melt_Pressure_bar",
        "Puller_Speed_m_min",
        "Material_Moisture_pct",
    ]

    print("\nFeatures used for training:")
    for feature in features:
        print(feature)

    print("\nMissing values in selected features:")
    print(df[features].isnull().sum())

    if df[features].isnull().sum().sum() > 0:
        raise ValueError("Selected features still contain missing values.")

    return features


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20):
    """Split data into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )


def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> dict[str, float | str]:
    """Evaluate a trained classifier and print metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def train_model_1(X: pd.DataFrame, y: pd.Series) -> tuple[object, dict[str, float | str]]:
    """Train and compare Model 1 classifiers for Defect_Status."""
    print("\n=== MODEL 1: Defect_Status ===")
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print(f"{name} trained successfully")

        print(f"\n{name} Metrics")
        metrics = evaluate_classifier(model, X_test, y_test, ["Fail", "Pass"])
        print(
            f"Accuracy: {metrics['accuracy'] * 100:.2f}%"
        )
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        results[name] = metrics

    best_name = max(
        results,
        key=lambda name: (
            results[name]["f1"],
            results[name]["accuracy"],
            -list(models.keys()).index(name),
        ),
    )

    best_model = models[best_name]
    best_model.fit(X, y)

    print(f"\nBest Model 1 selected: {best_name}")
    print(f"Weighted F1: {results[best_name]['f1']:.4f}")
    print(f"Accuracy: {results[best_name]['accuracy'] * 100:.2f}%")

    return best_model, {"best_name": best_name, "results": results}


def train_model_2(X: pd.DataFrame, y: pd.Series) -> tuple[object, dict[str, float | str]]:
    """Train and compare Model 2 classifiers for Defect_Type."""
    print("\n=== MODEL 2: Defect_Type ===")

    failed_frame = X.copy()
    failed_frame["Defect_Type"] = y
    failed_frame = failed_frame[failed_frame["Defect_Type"] != "Unknown"].copy()

    X_model2 = failed_frame.drop(columns=["Defect_Type"])
    y_model2 = failed_frame["Defect_Type"]

    X_train, X_test, y_train, y_test = split_data(X_model2, y_model2)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train_encoded)
        print(f"{name} trained successfully")

        print(f"\n{name} Metrics")
        y_pred_encoded = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision = precision_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            zero_division=0,
        )
        f1 = f1_score(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            zero_division=0,
        )

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test_encoded,
                y_pred_encoded,
                target_names=label_encoder.classes_,
                zero_division=0,
            )
        )
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_encoded, y_pred_encoded))

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "label_encoder": label_encoder,
        }

    best_name = max(
        results,
        key=lambda name: (
            results[name]["f1"],
            results[name]["accuracy"],
            -list(models.keys()).index(name),
        ),
    )

    best_model = models[best_name]
    best_model.fit(X_model2, y_train_encoded)
    best_model.label_encoder = label_encoder

    print(f"\nBest Model 2 selected: {best_name}")
    print(f"Weighted F1: {results[best_name]['f1']:.4f}")
    print(f"Accuracy: {results[best_name]['accuracy'] * 100:.2f}%")

    return best_model, {"best_name": best_name, "results": results, "label_encoder": label_encoder}


def predict_quality(
    sample: pd.DataFrame,
    model1,
    model2,
    features: list[str],
    label_encoder=None,
):
    """Predict quality status and defect type with the selected models."""
    sample = sample[features]

    status = model1.predict(sample)[0]
    status_probabilities = model1.predict_proba(sample)[0]
    fail_index = list(model1.classes_).index(0)
    failure_risk = status_probabilities[fail_index] * 100

    if status == 1:
        return {
            "Quality_Status": "PASS",
            "Failure_Risk": round(failure_risk, 2),
            "Likely_Defect": "None",
            "Defect_Confidence": 0.0,
        }

    defect_encoded = model2.predict(sample)[0]
    defect_probabilities = model2.predict_proba(sample)[0]
    defect_confidence = max(defect_probabilities) * 100

    encoded_labeler = label_encoder or getattr(model2, "label_encoder", None)

    if encoded_labeler is not None:
        defect = encoded_labeler.inverse_transform([defect_encoded])[0]
    else:
        defect = defect_encoded

    return {
        "Quality_Status": "FAIL",
        "Failure_Risk": round(failure_risk, 2),
        "Likely_Defect": defect,
        "Defect_Confidence": round(defect_confidence, 2),
    }


def save_artifacts(model1, model2, features) -> None:
    """Save the trained models and feature list using the expected filenames."""
    joblib.dump(model1, MODEL1_PATH)
    joblib.dump(model2, MODEL2_PATH)
    joblib.dump(features, FEATURES_PATH)


def main() -> None:
    """Run the full training and evaluation workflow."""
    print("Starting training pipeline...")

    df = load_and_preprocess_data()
    features = build_feature_list(df)

    X1 = df[features]
    y1 = df["Defect_Status"]
    model1, model1_info = train_model_1(X1, y1)

    df_failed = df[df["Defect_Status"] == 0].copy()
    df_failed = df_failed[df_failed["Defect_Type"] != "Unknown"].copy()

    X2 = df_failed[features]
    y2 = df_failed["Defect_Type"]
    model2, model2_info = train_model_2(X2, y2)

    sample = X1.iloc[[0]]
    result = predict_quality(
        sample,
        model1,
        model2,
        features,
        label_encoder=model2_info.get("label_encoder"),
    )

    print("\nFINAL SYSTEM TEST")
    print("Quality Status:", result["Quality_Status"])
    print("Failure Risk:", result["Failure_Risk"], "%")
    print("Likely Defect:", result["Likely_Defect"])
    print("Defect Confidence:", result["Defect_Confidence"], "%")

    save_artifacts(model1, model2, features)

    print("\nModels saved successfully")
    print("\nSaved feature list:")
    print(features)


if __name__ == "__main__":
    main()
