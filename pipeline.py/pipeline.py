# 1. IMPORT LIBRARIES
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 2. LOAD DATASET
df = pd.read_csv(
    "synthetic_thermoplastic_extrusion_quality_dataset.csv"
)
# 3. HANDLE MISSING VALUES
df['Braid_Type_Planned'] = (
    df['Braid_Type_Planned'].fillna('None')
)
df['Vacuum_bar'] = (
    df['Vacuum_bar'].fillna(df['Vacuum_bar'].mean())
)

df['Melt_Pressure_bar'] = (
    df['Melt_Pressure_bar'].fillna(
        df['Melt_Pressure_bar'].mean()
    )
)
df['Cooling_Water_Temp_C'] = (
    df['Cooling_Water_Temp_C'].fillna(
        df['Cooling_Water_Temp_C'].mean()
    )
)
df['Material_Moisture_pct'] = (
    df['Material_Moisture_pct'].fillna(
        df['Material_Moisture_pct'].mean()
    )
)
# 4. YOUR MANUAL MAPPING
df['Machine_ID'] = df['Machine_ID'].map({
    "EXT-01": 0,
    "EXT-02": 1,
    "EXT-03": 2
})
df['Shift'] = df['Shift'].map({
    "A": 0,
    "B": 1,
    "C": 2
})
df['Material_Type'] = df['Material_Type'].map({
    "PU": 0,
    "PA11": 1,
    "Hytrel": 2,
    "PA12": 3,
    "PVDF": 4
})
df['Cover_Required'] = df['Cover_Required'].map({
    "No": 0,
    "Yes": 1
})
df['Braid_Type_Planned'] = (
    df['Braid_Type_Planned'].map({
        "None": 0,
        "Aramid": 1,
        "SS Wire": 2,
        "Polyester": 3
    })
)
df['Defect_Status'] = df['Defect_Status'].map({
    "Fail": 0,
    "Pass": 1
})
# 5. FINAL FEATURE SET
features = [
    'Material_Type',
    'Target_Tube_OD_mm',
    'Target_Tube_ID_mm',
    'Vacuum_bar',
    'Screw_RPM',
    'Zone3_Temp_C',
    'Melt_Pressure_bar',
    'Puller_Speed_m_min',
    'Material_Moisture_pct'
    
]
# 6. MODEL 1 - PASS / FAIL
X1 = df[features]
y1 = df['Defect_Status']
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
    class_weight='balanced',
    n_jobs=-1
)
model1.fit(
    X1_train,
    y1_train
)
# 7. TEST MODEL 1

y1_pred = model1.predict(X1_test)
print("\n==============================")
print("MODEL 1 - DEFECT STATUS")
print("==============================")
print("Accuracy:",round(accuracy_score(y1_test, y1_pred) * 100,2),"%")

print("\nClassification Report:")

print(classification_report(y1_test,y1_pred,target_names=['Fail', 'Pass']))
print("\nConfusion Matrix:")
print(confusion_matrix(y1_test,y1_pred))
# 8. MODEL 2 - DEFECT TYPE
# Use only failed products
df_failed = df[df['Defect_Status'] == 0].copy()
# Remove failed rows without defect label
df_failed = df_failed.dropna(subset=['Defect_Type'])
X2 = df_failed[features]
y2 = df_failed['Defect_Type']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.20,random_state=42,stratify=y2)
model2 = RandomForestClassifier(
n_estimators=300,random_state=42,class_weight='balanced',n_jobs=-1)
model2.fit(
    X2_train,
    y2_train
)
# 9. TEST MODEL 2
y2_pred = model2.predict(X2_test)
print("\n==============================")
print("MODEL 2 - DEFECT TYPE")
print("==============================")
print("Accuracy:",round(accuracy_score(y2_test, y2_pred) * 100,2),"%")
print("\nClassification Report:")
print(classification_report(y2_test,y2_pred,zero_division=0))
print("\nConfusion Matrix:")

print(confusion_matrix(y2_test,y2_pred))
# 10. FINAL COMBINED PREDICTION FUNCTION
def predict_quality(sample):

    # Force correct feature order
    sample = sample[features]

   
    # MODEL 1
    

    status = model1.predict(sample)[0]

    status_probabilities = (
        model1.predict_proba(sample)[0]
    )

    fail_index = list(
        model1.classes_
    ).index(0)

    failure_risk = (
        status_probabilities[fail_index]
    )


    # PASS RESULT
  

    if status == 1:

        return {
            "quality_status": "PASS",
            "failure_risk_percent": round(
                failure_risk * 100,
                2
            ),
            "likely_defect": "None",
            "defect_confidence_percent": 0.0
        }


    # FAIL RESULT

    defect = model2.predict(sample)[0]

    defect_probabilities = (
        model2.predict_proba(sample)[0]
    )

    defect_confidence = max(
        defect_probabilities
    )


    return {
        "quality_status": "FAIL",
        "failure_risk_percent": round(
            failure_risk * 100,
            2
        ),
        "likely_defect": defect,
        "defect_confidence_percent": round(
            defect_confidence * 100,
            2
        )
    }


# 11. TEST FINAL COMBINED SYSTEM
sample = X1_test.iloc[[0]]

result = predict_quality(sample)

print("\n==============================")
print("FINAL SYSTEM OUTPUT")
print("==============================")

print(
    "Quality Status:",
    result["quality_status"]
)

print(
    "Failure Risk:",
    result["failure_risk_percent"],
    "%"
)

print(
    "Likely Defect:",
    result["likely_defect"]
)

print(
    "Defect Confidence:",
    result["defect_confidence_percent"],
    "%"
)

# 12. SAVE MODELS FOR DEPLOYMENT
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


# Save fill values for future live data
fill_values = {
    "Vacuum_bar": df['Vacuum_bar'].mean(),
    "Melt_Pressure_bar": df['Melt_Pressure_bar'].mean(),
    "Cooling_Water_Temp_C": df['Cooling_Water_Temp_C'].mean(),
    "Material_Moisture_pct": df['Material_Moisture_pct'].mean()
}

joblib.dump(
    fill_values,
    "fill_values.pkl"
)


print("\nModels saved successfully.")