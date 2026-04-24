import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Student Predictor", layout="centered")

# -------------------------------
# Title
# -------------------------------
st.title("🎓 Smart Student Exam Score Predictor")


# -------------------------------
# Inputs
# -------------------------------
st.subheader("📥 Enter Student Details")

study_hours = st.slider("📘 Study Hours per Day", 0, 12, 5)
attendance = st.slider("📊 Attendance (%)", 0, 100, 75)
sleep = st.slider("😴 Sleep Hours per Night", 0, 12, 7)
mental_health = st.slider("🧠 Mental Health (1-10)", 1, 10, 5)
job = st.selectbox("💼 Part-Time Job", ["No", "Yes"])

job = 1 if job == "Yes" else 0

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict Score"):

    # Prepare input
    input_data = np.array([[study_hours, attendance, sleep, mental_health, job]])
    result = model.predict(input_data)[0]

    # -------------------------------
    # Display Score
    # -------------------------------
    st.metric(label="🎯 Predicted Exam Score", value=f"{result:.2f}")

    # -------------------------------
    # Performance Category
    # -------------------------------
    if result < 50:
        category = "Poor"
        st.error("📉 Performance: Poor")
    elif result < 70:
        category = "Average"
        st.warning("📊 Performance: Average")
    elif result < 85:
        category = "Good"
        st.info("📈 Performance: Good")
    else:
        category = "Excellent"
        st.success("🏆 Performance: Excellent")

    st.write(f"### 📌 Final Result: {result:.2f} ({category})")

    # -------------------------------
    # Suggestions
    # -------------------------------
    st.subheader("📢 Suggestions")

    if sleep < 6:
        st.write("- 😴 Increase sleep for better concentration")
    if study_hours < 3:
        st.write("- 📘 Increase study time")
    if attendance < 60:
        st.write("- 📊 Improve attendance")
    if mental_health < 5:
        st.write("- 🧠 Focus on mental well-being")

    # -------------------------------
    # Input Visualization Graph
    # -------------------------------
    st.subheader("📊 Input Overview")

    input_df = pd.DataFrame({
        "Feature": ["Study Hours", "Attendance", "Sleep", "Mental Health", "Job"],
        "Value": [study_hours, attendance, sleep, mental_health, job]
    })

    st.bar_chart(input_df.set_index("Feature"))

    # -------------------------------
    # Progress Bar
    # -------------------------------
    st.subheader("📈 Score Progress")
    st.progress(int(result))

    # -------------------------------
    # Feature Importance (if RF used)
    # -------------------------------
    try:
        st.subheader("📊 Feature Importance")

        features = ["Study Hours", "Attendance", "Sleep", "Mental Health", "Job"]
        importances = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Impact on Score")

        st.pyplot(fig)
    except:
        st.info("Feature importance available only for tree-based models.")

    # -------------------------------
    # Comparison Graph
    # -------------------------------
    st.subheader("🔄 Study Hours vs Predicted Score")

    hours_range = list(range(0, 13))
    predictions = []

    for h in hours_range:
        temp = np.array([[h, attendance, sleep, mental_health, job]])
        pred = model.predict(temp)[0]
        predictions.append(pred)

    chart_data = pd.DataFrame({
        "Study Hours": hours_range,
        "Predicted Score": predictions
    })

    st.line_chart(chart_data.set_index("Study Hours"))
