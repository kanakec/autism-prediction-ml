import streamlit as st
import pandas as pd
import pickle
import joblib

# Load model
model = joblib.load("autism_model.pkl")

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(
    page_title="Autism Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Autism Spectrum Disorder Prediction")
st.write("Enter the details below to get a prediction.")

st.subheader("Screening Scores")

scores = {}

for i in range(1, 11):
    scores[f"A{i}_Score"] = st.selectbox(
        f"A{i} Score",
        [0, 1]
    )

st.subheader("Personal Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

gender = st.selectbox(
    "Gender",
    encoders["gender"].classes_.tolist()
)

ethnicity = st.selectbox(
    "Ethnicity",
    encoders["ethnicity"].classes_.tolist()
)

jaundice = st.selectbox(
    "Jaundice at birth?",
    encoders["jaundice"].classes_.tolist()
)

country = st.selectbox(
    "Country of Residence",
    encoders["contry_of_res"].classes_.tolist()
)

used_app_before = st.selectbox(
    "Used screening app before?",
    encoders["used_app_before"].classes_.tolist()
)

result = st.number_input(
    "Screening Result Score",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

relation = st.selectbox(
    "Relation",
    encoders["relation"].classes_.tolist()
)

if st.button("🔍 Predict ASD"):

    input_data = {
        **scores,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "contry_of_res": country,
        "used_app_before": used_app_before,
        "result": result,
        "relation": relation
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for column in [
        "gender",
        "ethnicity",
        "jaundice",
        "contry_of_res",
        "used_app_before",
        "relation"
    ]:
        input_df[column] = encoders[column].transform(
            input_df[column]
        )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: ASD")
    else:
        st.success("✅ Prediction: No ASD")
