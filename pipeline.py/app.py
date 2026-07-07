import streamlit as st
import pandas as pd
import altair as alt
import joblib

st.set_page_config(
    page_title="Extrusion Quality Predictor",
    page_icon="🧪",
    layout="wide"
)

model1 = joblib.load("quality_status_model.pkl")
model2 = joblib.load("defect_type_model.pkl")
features = joblib.load("model_features.pkl")

material_map = {
    "PU": 0,
    "PA11": 1,
    "Hytrel": 2,
    "PA12": 3,
    "PVDF": 4
}

machine_map = {
    "EXT-01": 0,
    "EXT-02": 1,
    "EXT-03": 2
}

names = {
    "Target_Tube_OD_mm": "Target OD",
    "Target_Tube_ID_mm": "Target ID",
    "Target_Wall_Thickness_mm": "Wall Thickness",
    "OD_Tolerance_mm": "OD Tolerance",
    "ID_Tolerance_mm": "ID Tolerance",
    "Wall_Tolerance_mm": "Wall Tolerance",
    "Vacuum_bar": "Vacuum",
    "Screw_RPM": "Screw Speed",
    "Zone2_Temp_C": "Zone 2 Temperature",
    "Zone3_Temp_C": "Zone 3 Temperature",
    "Melt_Pressure_bar": "Melt Pressure",
    "Puller_Speed_m_min": "Puller Speed",
    "Material_Moisture_pct": "Material Moisture"
}

units = {
    "Target_Tube_OD_mm": "mm",
    "Target_Tube_ID_mm": "mm",
    "Target_Wall_Thickness_mm": "mm",
    "OD_Tolerance_mm": "mm",
    "ID_Tolerance_mm": "mm",
    "Wall_Tolerance_mm": "mm",
    "Vacuum_bar": "bar",
    "Screw_RPM": "RPM",
    "Zone2_Temp_C": "°C",
    "Zone3_Temp_C": "°C",
    "Melt_Pressure_bar": "bar",
    "Puller_Speed_m_min": "m/min",
    "Material_Moisture_pct": "%"
}

st.markdown("""
<style>
.stButton > button {
    background: #0f4c81;
    color: white;
    border-radius: 10px;
    padding: 0.7rem 1.4rem;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

st.title("AI-Based Quality Risk Prediction System")
st.caption(
    "Predict extrusion failure risk, likely defect type "
    "and local parameter sensitivity."
)

with st.sidebar:
    st.header("System Guide")
    st.write(
        "Enter tube requirements, tolerances and current "
        "process settings."
    )
    st.info(
        "Parameter analysis shows how the trained model "
        "responds to small tested input changes. "
        "These are model-based indications, not proven physical causes."
    )
    st.metric("Model 1", "Pass / Fail")
    st.metric("Model 2", "Defect Type")

with st.form("quality_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Tube Requirements")

        material = st.selectbox(
            "Material Type",
            list(material_map)
        )

        machine = st.selectbox(
            "Machine ID",
            list(machine_map)
        )

        target_od = st.number_input(
            "Target OD (mm)",
            min_value=4.0,
            max_value=20.0,
            value=10.0,
            step=0.5
        )

        target_id = st.number_input(
            "Target ID (mm)",
            min_value=1.0,
            max_value=20.0,
            value=6.0,
            step=0.5
        )

        target_wall = st.number_input(
            "Wall Thickness (mm)",
            min_value=0.75,
            max_value=2.5,
            value=1.5,
            step=0.05
        )

    with c2:
        st.subheader("Tolerances")

        od_tol = st.number_input(
            "OD Tolerance (mm)",
            min_value=0.01,
            max_value=0.30,
            value=0.15,
            step=0.01
        )

        id_tol = st.number_input(
            "ID Tolerance (mm)",
            min_value=0.01,
            max_value=0.30,
            value=0.15,
            step=0.01
        )

        wall_tol = st.number_input(
            "Wall Tolerance (mm)",
            min_value=0.008,
            max_value=0.30,
            value=0.10,
            step=0.01
        )

    with c3:
        st.subheader("Process Parameters")

        vacuum = st.number_input(
            "Vacuum (bar)",
            value=-0.60,
            step=0.01
        )

        rpm = st.number_input(
            "Screw Speed (RPM)",
            value=60.0,
            step=1.0
        )

        z2 = st.number_input(
            "Zone 2 Temperature (°C)",
            value=195.0,
            step=1.0
        )

        z3 = st.number_input(
            "Zone 3 Temperature (°C)",
            value=200.0,
            step=1.0
        )

        pressure = st.number_input(
            "Melt Pressure (bar)",
            value=150.0,
            step=1.0
        )

        puller = st.number_input(
            "Puller Speed (m/min)",
            value=15.0,
            step=0.1
        )

        moisture = st.number_input(
            "Material Moisture (%)",
            value=0.07,
            step=0.01
        )

    submitted = st.form_submit_button(
        "Predict Quality Risk"
    )

if submitted:
    values = {
        "Material_Type": material_map[material],
        "Machine_ID": machine_map[machine],
        "Target_Tube_OD_mm": target_od,
        "Target_Tube_ID_mm": target_id,
        "Target_Wall_Thickness_mm": target_wall,
        "OD_Tolerance_mm": od_tol,
        "ID_Tolerance_mm": id_tol,
        "Wall_Tolerance_mm": wall_tol,
        "Vacuum_bar": vacuum,
        "Screw_RPM": rpm,
        "Zone2_Temp_C": z2,
        "Zone3_Temp_C": z3,
        "Melt_Pressure_bar": pressure,
        "Puller_Speed_m_min": puller,
        "Material_Moisture_pct": moisture
    }

    sample = pd.DataFrame([values])[features]

    status = model1.predict(sample)[0]
    status_probs = model1.predict_proba(sample)[0]

    fail_index = list(model1.classes_).index(0)
    fail_risk = status_probs[fail_index] * 100

    def get_fail_risk(data):
        probabilities = model1.predict_proba(data)[0]
        return probabilities[fail_index] * 100

    def sensitivity_analysis():
        best_changes = []
        worst_changes = []

        skip = {
            "Material_Type",
            "Machine_ID"
        }

        for col in features:
            if col in skip:
                continue

            current = float(sample.at[0, col])
            step = max(abs(current) * 0.05, 0.01)
            options = []

            for direction, change in [
                ("Increase", step),
                ("Decrease", -step)
            ]:
                tested_sample = sample.copy()
                tested_value = current + change

                tested_sample.at[0, col] = tested_value

                new_risk = get_fail_risk(tested_sample)
                effect = new_risk - fail_risk

                options.append({
                    "feature": col,
                    "name": names.get(col, col),
                    "direction": direction,
                    "current": current,
                    "tested": tested_value,
                    "risk": new_risk,
                    "effect": effect
                })

            best_changes.append(
                min(options, key=lambda x: x["risk"])
            )

            worst_changes.append(
                max(options, key=lambda x: x["risk"])
            )

        return (
            pd.DataFrame(best_changes),
            pd.DataFrame(worst_changes)
        )

    best_analysis, worst_analysis = sensitivity_analysis()

    risk_reductions = (
        best_analysis[best_analysis["effect"] < -0.2]
        .sort_values("effect")
    )

    risk_contributors = (
        worst_analysis[worst_analysis["effect"] > 0.2]
        .sort_values("effect", ascending=False)
    )

    if fail_risk >= 70:
        risk_level = "High Risk"
    elif fail_risk >= 35:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Failure Risk",
        f"{fail_risk:.2f}%"
    )

    m2.metric(
        "Risk Level",
        risk_level
    )

    m3.metric(
        "Prediction",
        "PASS" if status == 1 else "FAIL"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Prediction Result")

        if status == 1:
            st.success(
                "Predicted Quality Status: PASS"
            )

            st.info(
                "No defect prediction is triggered because "
                "Model 1 predicts PASS."
            )

        else:
            st.error(
                "Predicted Quality Status: FAIL"
            )

            defect = model2.predict(sample)[0]
            defect_probs = model2.predict_proba(sample)[0]
            confidence = max(defect_probs) * 100

            st.warning(
                f"Likely Defect: {defect}"
            )

            st.metric(
                "Defect Confidence",
                f"{confidence:.2f}%"
            )

        st.subheader("Likely Risk Contributors")

        st.caption(
            "Small tested changes that increased the "
            "model's predicted failure risk."
        )

        if risk_contributors.empty:
            st.info(
                "No strong local risk contributor was detected."
            )

        else:
            for _, row in risk_contributors.head(5).iterrows():
                unit = units.get(row["feature"], "")

                st.write(
                    f"**{row['direction']} {row['name']}** "
                    f"from {row['current']:.2f} to "
                    f"{row['tested']:.2f} {unit} increased "
                    f"predicted failure risk by "
                    f"**{row['effect']:.2f} percentage points**."
                )

        st.subheader("Inputs That May Reduce Failure Risk")

        st.caption(
            "For each parameter, only the best tested direction "
            "is shown."
        )

        if risk_reductions.empty:
            st.info(
                "No tested parameter change produced a "
                "meaningful local risk reduction."
            )

        else:
            for _, row in risk_reductions.head(5).iterrows():
                unit = units.get(row["feature"], "")

                st.write(
                    f"**{row['direction']} {row['name']}** "
                    f"from {row['current']:.2f} to "
                    f"{row['tested']:.2f} {unit} reduced "
                    f"predicted failure risk by "
                    f"**{abs(row['effect']):.2f} percentage points**."
                )

        st.subheader("Suggested Adjustments")

        st.caption(
            "Best local changes found by testing both directions "
            "for each parameter."
        )

        if risk_reductions.empty:
            st.info(
                "No meaningful adjustment was found "
                "within the tested range."
            )

        else:
            for i, (_, row) in enumerate(
                risk_reductions.head(5).iterrows(),
                1
            ):
                unit = units.get(row["feature"], "")

                st.write(
                    f"{i}. **{row['direction']} {row['name']}** "
                    f"from {row['current']:.2f} to approximately "
                    f"**{row['tested']:.2f} {unit}** "
                    f"→ tested failure risk: "
                    f"**{row['risk']:.2f}%**"
                )

    with right:
        st.subheader("Quality Probability")

        status_df = pd.DataFrame({
            "Status": [
                "FAIL" if cls == 0 else "PASS"
                for cls in model1.classes_
            ],
            "Probability": status_probs * 100
        })

        status_chart = alt.Chart(
            status_df
        ).mark_bar(
            cornerRadius=5
        ).encode(
            x=alt.X(
                "Probability:Q",
                title="Probability (%)",
                scale=alt.Scale(domain=[0, 100])
            ),
            y=alt.Y(
                "Status:N",
                sort="-x",
                title=""
            ),
            color=alt.Color(
                "Status:N",
                legend=None
            ),
            tooltip=[
                "Status",
                alt.Tooltip(
                    "Probability:Q",
                    format=".2f"
                )
            ]
        ).properties(
            height=220
        )

        st.altair_chart(
            status_chart,
            use_container_width=True
        )

        if status == 0:
            st.subheader("Defect Probability")

            defect_df = pd.DataFrame({
                "Defect": [
                    str(label)
                    for label in model2.classes_
                ],
                "Probability": defect_probs * 100
            })

            defect_chart = alt.Chart(
                defect_df
            ).mark_bar(
                cornerRadius=5
            ).encode(
                x=alt.X(
                    "Probability:Q",
                    title="Probability (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),
                y=alt.Y(
                    "Defect:N",
                    sort="-x",
                    title=""
                ),
                color=alt.Color(
                    "Defect:N",
                    legend=None
                ),
                tooltip=[
                    "Defect",
                    alt.Tooltip(
                        "Probability:Q",
                        format=".2f"
                    )
                ]
            ).properties(
                height=280
            )

            st.altair_chart(
                defect_chart,
                use_container_width=True
            )

        st.subheader("Parameter Impact")

        impact_parts = []

        if not risk_contributors.empty:
            contributor_impact = risk_contributors.copy()
            contributor_impact["Type"] = "Raises Risk"
            impact_parts.append(contributor_impact)

        if not risk_reductions.empty:
            reduction_impact = risk_reductions.copy()
            reduction_impact["Type"] = "Reduces Risk"
            impact_parts.append(reduction_impact)

        if impact_parts:
            impact = pd.concat(
                impact_parts,
                ignore_index=True
            )

            impact = (
                impact.assign(
                    absolute_effect=impact["effect"].abs()
                )
                .sort_values(
                    "absolute_effect",
                    ascending=False
                )
                .head(10)
            )

            impact["Change"] = (
                impact["direction"]
                + " "
                + impact["name"]
            )

            impact_chart = alt.Chart(
                impact
            ).mark_bar().encode(
                x=alt.X(
                    "effect:Q",
                    title="Change in Failure Risk (percentage points)"
                ),
                y=alt.Y(
                    "Change:N",
                    sort="-x",
                    title=""
                ),
                color=alt.Color(
                    "Type:N",
                    title=""
                ),
                tooltip=[
                    "Change",
                    "Type",
                    alt.Tooltip(
                        "effect:Q",
                        format=".2f"
                    )
                ]
            ).properties(
                height=350
            )

            st.altair_chart(
                impact_chart,
                use_container_width=True
            )

    with st.expander("View Input Summary"):
        display_sample = sample.copy()

        display_sample["Material_Type"] = material
        display_sample["Machine_ID"] = machine

        st.dataframe(
            display_sample.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )

    st.caption(
        "Important: risk contributors and suggested adjustments "
        "represent local model sensitivity, not proven physical "
        "causation. Validate process changes with production trials "
        "and engineering limits before changing machine settings."
    )