import json
import os

import joblib
import streamlit as st

meta_path = "app/model_meta.json"
model_path = "app/model.joblib"


def load_meta():
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


meta = load_meta()

st.title("Iris Species Predictor")

# Input features
f1 = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
f2 = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
f3 = st.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
f4 = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        prediction = model.predict([[f1, f2, f3, f4]])
        target_names = ["setosa", "versicolor", "virginica"]
        st.success(f"Prediction: **{target_names[prediction[0]]}**")
    else:
        st.error("Model file not found. Please run train_model.py first.")

# Footer
if meta:
    st.markdown("---")
    st.markdown(
        f"**Version:** {meta['version']} • "
        f"**Best Model:** {meta['best_model']} • "
        f"**MLflow Run ID:** `{meta['mlflow_run_id'][:8]}...` • "
        f"**Accuracy:** {meta['metrics']['accuracy']:.3f}"
    )
    st.info("[Open MLflow UI](http://localhost:5000)")
