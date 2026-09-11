import joblib
import numpy as np
import streamlit as st

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_stats = joblib.load("feature_stats.pkl")

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺")
st.title("Breast Cancer Predictor")
st.write(
    "Enter the cell nuclei measurements below — the usual range for each "
    "feature is shown as a guide. The model will then predict whether the "
    "tumor is **benign** or **malignant**."
)

def format_label(name: str) -> str:
    for suffix, prefix in [("_mean", "Average"), ("_se", "Variability In"), ("_worst", "Largest")]:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return f"{prefix} {base.replace('_', ' ').title()}"
    return name.replace("_", " ").title()


values = {}
cols = st.columns(2)

for i, name in enumerate(feature_names):
    stat = feature_stats[name]
    label = format_label(name)
    usual_range = f"Usual range: {stat['min']:g} - {stat['max']:g}"
    with cols[i % 2]:
        values[name] = st.number_input(
            f"{label}  ({usual_range})",
            value=0.0,
            step=stat["step"],
            format=f"%.{stat['decimals']}f",
            help=f"{usual_range} (you can enter values outside this range)",
        )

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    input_vector = np.array([[values[name] for name in feature_names]])
    scaled_input = scaler.transform(input_vector)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    if prediction == 1:
        st.error(f"⚠️ Prediction: **Malignant** (confidence: {probability[1]:.1%})")
    else:
        st.success(f"✅ Prediction: **Benign** (confidence: {probability[0]:.1%})")
