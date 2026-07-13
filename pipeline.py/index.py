import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv(
    "synthetic_thermoplastic_extrusion_quality_dataset.csv"
)

df["Braid_Type_Planned"] = (
    df["Braid_Type_Planned"].fillna("None")
)

df["Vacuum_bar"] = df["Vacuum_bar"].fillna(
    df["Vacuum_bar"].mean()
)

df["Melt_Pressure_bar"] = (
    df["Melt_Pressure_bar"].fillna(
        df["Melt_Pressure_bar"].mean()
    )
)

df["Cooling_Water_Temp_C"] = (
    df["Cooling_Water_Temp_C"].fillna(
        df["Cooling_Water_Temp_C"].mean()
    )
)

df["Material_Moisture_pct"] = (
    df["Material_Moisture_pct"].fillna(
        df["Material_Moisture_pct"].mean()
    )
)

df["Defect_Type"] = (
    df["Defect_Type"].fillna("Unknown")
)

df["Machine_ID"] = df["Machine_ID"].map({
    "EXT-01": 0,
    "EXT-02": 1,
    "EXT-03": 2
})

df["Shift"] = df["Shift"].map({
    "A": 0,
    "B": 1,
    "C": 2
})

df["Material_Type"] = df["Material_Type"].map({
    "PU": 0,
    "PA11": 1,
    "Hytrel": 2,
    "PA12": 3,
    "PVDF": 4
})

df["Cover_Required"] = df["Cover_Required"].map({
    "No": 0,
    "Yes": 1
})

df["Braid_Type_Planned"] = (
    df["Braid_Type_Planned"].map({
        "None": 0,
        "Aramid": 1,
        "SS Wire": 2,
        "Polyester": 3
    })
)

df["Defect_Status"] = df["Defect_Status"].map({
    "Fail": 0,
    "Pass": 1
})

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
    "Material_Moisture_pct"
]

print("\nFeatures used for training:")

for feature in features:
    print(feature)

print("\nMissing values in selected features:")
print(df[features].isnull().sum())

if df[features].isnull().sum().sum() > 0:
    raise ValueError(
        "Selected features still contain missing values."
    )

X1 = df[features]
y1 = df["Defect_Status"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1,
    y1,
    test_size=0.20,
    random_state=42,
    stratify=y1
)

model1 = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model1.fit(
    X1_train,
    y1_train
)

y1_pred = model1.predict(
    X1_test
)

print("\nMODEL 1 RESULTS")

print(
    f"Accuracy: "
    f"{accuracy_score(y1_test, y1_pred) * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y1_test,
        y1_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y1_test,
        y1_pred
    )
)

df_failed = df[
    df["Defect_Status"] == 0
].copy()

df_failed = df_failed[
    df_failed["Defect_Type"] != "Unknown"
].copy()

print("\nFailed samples for Model 2:")
print(len(df_failed))

print("\nDefect distribution:")
print(
    df_failed["Defect_Type"].value_counts()
)

X2 = df_failed[features]
y2 = df_failed["Defect_Type"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    test_size=0.20,
    random_state=42,
    stratify=y2
)

model2 = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model2.fit(
    X2_train,
    y2_train
)

y2_pred = model2.predict(
    X2_test
)

print("\nMODEL 2 RESULTS")

print(
    f"Accuracy: "
    f"{accuracy_score(y2_test, y2_pred) * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y2_test,
        y2_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y2_test,
        y2_pred
    )
)

def predict_quality(sample):
    sample = sample[features]

    status = model1.predict(
        sample
    )[0]

    status_probabilities = model1.predict_proba(
        sample
    )[0]

    fail_index = list(
        model1.classes_
    ).index(0)

    failure_risk = (
        status_probabilities[fail_index]
        * 100
    )

    if status == 1:
        return {
            "Quality_Status": "PASS",
            "Failure_Risk": round(
                failure_risk,
                2
            ),
            "Likely_Defect": "None",
            "Defect_Confidence": 0.0
        }

    defect = model2.predict(
        sample
    )[0]

    defect_probabilities = model2.predict_proba(
        sample
    )[0]

    defect_confidence = (
        max(defect_probabilities)
        * 100
    )

    return {
        "Quality_Status": "FAIL",
        "Failure_Risk": round(
            failure_risk,
            2
        ),
        "Likely_Defect": defect,
        "Defect_Confidence": round(
            defect_confidence,
            2
        )
    }

sample = X1_test.iloc[[0]]

result = predict_quality(
    sample
)

print("\nFINAL SYSTEM TEST")

print(
    "Quality Status:",
    result["Quality_Status"]
)

print(
    "Failure Risk:",
    result["Failure_Risk"],
    "%"
)

print(
    "Likely Defect:",
    result["Likely_Defect"]
)

print(
    "Defect Confidence:",
    result["Defect_Confidence"],
    "%"
)

joblib.dump(
    model1,
    "quality_status_model.pkl"
)

joblib.dump(
    model2,
    "defect_type_model.pkl"
)

joblib.dump(
    features,
    "model_features.pkl"
)

print("\nModels saved successfully")

print("\nSaved feature list:")
print(features)