import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('student_model.pkl')

st.title("🎓 Student Performance Prediction App")
st.write("Enter the student's details below to predict whether they will pass or fail.")

# Input fields for the user
study_hours = st.slider("Daily Study Hours", 0.0, 10.0, 3.0)
attendance_percentage = st.slider("Attendance Percentage (%)", 0, 100, 75)
previous_grades = st.slider("Previous Grades (%)", 0, 100, 70)
extracurricular_activities = st.selectbox("Extracurricular Activities", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.0)

# Prediction button
if st.button("Predict Performance"):
    # Prepare input array
    input_data = np.array([[study_hours, attendance_percentage, previous_grades, extracurricular_activities, sleep_hours]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.success("🎉 Prediction: The student is likely to **PASS**!")
    else:
        st.error("⚠️ Prediction: The student is likely to **FAIL**.")