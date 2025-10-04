import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO

# Load trained model and scaler
model = joblib.load("stroke_lr_model.pkl")
scaler = joblib.load("stroke_scaler.pkl")

# Column names
feature_names = [
    'age', 'chest_pain', 'high_blood_pressure', 'irregular_heartbeat', 'shortness_of_breath',
    'fatigue_weakness', 'dizziness', 'swelling_edema', 'neck_jaw_pain', 'excessive_sweating',
    'persistent_cough', 'nausea_vomiting', 'chest_discomfort', 'cold_hands_feet',
    'snoring_sleep_apnea', 'anxiety_doom', 'stroke_risk_percentage', 'gender_Male'
]

# Streamlit page config
st.set_page_config(page_title="Stroke Risk Analyzer", layout="centered")
st.title("Stroke Risk Analyzer")
st.write("Fill in all details to check stroke risk prediction")
st.write("⚠️ This tool is for informational purposes only. Always consult a doctor for confirmation.")

# Patient Name input
patient_name = st.text_input("Patient Name", "")

# Collect features
age = st.slider("Age", 1, 100, 30)
chest_pain = st.selectbox("Chest Pain (0=No, 1=Yes)", [0, 1])
high_blood_pressure = st.selectbox("High Blood Pressure (0=No, 1=Yes)", [0, 1])
irregular_heartbeat = st.selectbox("Irregular Heartbeat (0=No, 1=Yes)", [0, 1])
shortness_of_breath = st.selectbox("Shortness of Breath (0=No, 1=Yes)", [0, 1])
fatigue_weakness = st.selectbox("Fatigue / Weakness (0=No, 1=Yes)", [0, 1])
dizziness = st.selectbox("Dizziness (0=No, 1=Yes)", [0, 1])
swelling_edema = st.selectbox("Swelling / Edema (0=No, 1=Yes)", [0, 1])
neck_jaw_pain = st.selectbox("Neck / Jaw Pain (0=No, 1=Yes)", [0, 1])
excessive_sweating = st.selectbox("Excessive Sweating (0=No, 1=Yes)", [0, 1])
persistent_cough = st.selectbox("Persistent Cough (0=No, 1=Yes)", [0, 1])
nausea_vomiting = st.selectbox("Nausea / Vomiting (0=No, 1=Yes)", [0, 1])
chest_discomfort = st.selectbox("Chest Discomfort (0=No, 1=Yes)", [0, 1])
cold_hands_feet = st.selectbox("Cold Hands / Feet (0=No, 1=Yes)", [0, 1])
snoring_sleep_apnea = st.selectbox("Snoring / Sleep Apnea (0=No, 1=Yes)", [0, 1])
anxiety_doom = st.selectbox("Feeling of Impending Doom (0=No, 1=Yes)", [0, 1])
stroke_risk_percentage = st.slider("Stroke Risk % (if known, else 0)", 0, 100, 0)
gender_male = st.selectbox("Gender (Male=1, Female=0)", [0, 1])

# Create DataFrame
features = pd.DataFrame([[
    age, chest_pain, high_blood_pressure, irregular_heartbeat, shortness_of_breath,
    fatigue_weakness, dizziness, swelling_edema, neck_jaw_pain, excessive_sweating,
    persistent_cough, nausea_vomiting, chest_discomfort, cold_hands_feet,
    snoring_sleep_apnea, anxiety_doom, stroke_risk_percentage, gender_male
]], columns=feature_names)

# Prediction Button
if st.button("Predict Stroke Risk"):
    if patient_name.strip() == "":
        st.error("Please enter the patient's name.")
    else:
        # Scale input
        features_scaled = scaler.transform(features)

        # Predict probability
        probability = model.predict_proba(features_scaled)[0][1] * 100

        # Determine risk level & advice
        if probability < 30:
            risk_level = "Low Risk"
            st.success(f"✅ Low Stroke Risk\nProbability: {probability:.2f}%")
            advice = "Maintain a healthy lifestyle and regular check-ups."
            risk_color = (0, 200, 0)  # Green
        elif probability < 70:
            risk_level = "Medium Risk"
            st.warning(f"⚠️ Medium Stroke Risk\nProbability: {probability:.2f}%")
            advice = "Consult a doctor for further evaluation."
            risk_color = (255, 200, 0)  # Yellow
        else:
            risk_level = "High Risk"
            st.error(f"❌ High Stroke Risk\nProbability: {probability:.2f}%")
            advice = "Seek medical attention immediately."
            risk_color = (255, 0, 0)  # Red

        st.info(f"Advice: {advice}")

        # --- Generate Professional PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # Title
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, "Stroke Risk Medical Report", ln=True, align="C")
        pdf.ln(5)

        # Patient Details
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Patient Details", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
        pdf.cell(0, 8, f"Age: {age}", ln=True)
        pdf.cell(0, 8, f"Gender: {'Male' if gender_male==1 else 'Female'}", ln=True)
        pdf.cell(0, 8, f"Stroke Probability: {probability:.2f}%", ln=True)
        pdf.cell(0, 8, f"Risk Level: {risk_level}", ln=True)
        pdf.ln(5)

        # Horizontal line
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)

        # Stroke Risk Bar
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Stroke Risk Level", ln=True)
        pdf.set_fill_color(*risk_color)
        bar_width = probability * 1.5
        pdf.rect(x=15, y=pdf.get_y(), w=bar_width, h=10, style="F")
        pdf.set_xy(15 + bar_width + 2, pdf.get_y())
        pdf.cell(0, 10, f"{probability:.2f}%", ln=True)
        pdf.ln(12)

        # Feature Table
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Patient Symptoms / Inputs", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 12)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(90, 8, "Feature", 1, 0, "C", True)
        pdf.cell(30, 8, "Value", 1, 1, "C", True)

        pdf.set_font("Arial", "", 12)
        fill = False
        for col in feature_names:
            value = features.at[0, col]
            # Correct display name for gender
            display_name = col.replace("_", " ").title()
            if col == "gender_Male":
                value = "Male" if value == 1 else "Female"
                display_name = "Gender"
            pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(90, 8, display_name, 1, 0, "L", fill)
            pdf.cell(30, 8, str(value), 1, 1, "C", fill)
            fill = not fill

        pdf.ln(5)

        # Medical Advice
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Medical Advice", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, advice)

        # Footer line and text (appears on last page)
        pdf.set_y(-20)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Developed by Gopi G (B.E)", align="C")

        # Save PDF to BytesIO
        pdf_buffer = BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)

        # Download button
        st.download_button(
            label="Download Professional Medical Report (PDF)",
            data=pdf_buffer,
            file_name=f"{patient_name.replace(' ', '_')}_stroke_report.pdf",
            mime="application/pdf"
        )
