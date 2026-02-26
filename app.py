import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="AI Job Risk Predictor", page_icon="🤖", layout="centered")


@st.cache_resource
def load_assets():
    model = joblib.load('job_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_job = joblib.load('le_job.pkl')
    le_ind = joblib.load('le_ind.pkl')
    return model, scaler, le_job, le_ind

try:
    model, scaler, le_job, le_ind = load_assets()
except:
    st.error("❌ Model files not found! Please upload .pkl files to the repository.")

st.title("🤖 AI Job Risk Predictor")
st.markdown("Enter your professional details to see how much AI could impact your job role.")


col1, col2 = st.columns(2)

with col1:
    job_role = st.selectbox("Select your Job Role", le_job.classes_)
    industry = st.selectbox("Select Industry", le_ind.classes_)
    education = st.slider("Education Requirement Level (1-5)", 1, 5, 3)

with col2:
    ai_score = st.slider("AI Replacement Score (0-100)", 0.0, 100.0, 50.0)
    urgency = st.slider("Reskilling Urgency Score (0-100)", 0.0, 100.0, 50.0)
    skill_gap = st.slider("Skill Gap Index (0-100)", 0.0, 100.0, 50.0)
    adoption = st.slider("AI Adoption Level (0-100)", 0.0, 100.0, 50.0)


if st.button("Predict My Risk Level 🚀"):
   
    job_enc = le_job.transform([job_role])[0]
    ind_enc = le_ind.transform([industry])[0]
    

    features = np.array([[job_enc, ind_enc, education, ai_score, urgency, skill_gap, adoption]])
    

    features_scaled = scaler.transform(features)
    

    prediction = model.predict(features_scaled)[0]
    
    
    st.divider()
    if prediction == 'Low':
        st.balloons()
        st.success(f"✅ Your Risk Category: **{prediction}**")
        st.info("Great! Your role seems stable for now. Keep learning new tools.")
    elif prediction == 'Medium':
        st.warning(f"⚠️ Your Risk Category: **{prediction}**")
        st.write("AI will assist your role. Consider upskilling in AI-driven technologies.")
    else:
        st.error(f"🚨 Your Risk Category: **{prediction}**")
        st.write("High risk of automation. Immediate reskilling is highly recommended.")

st.sidebar.info("Developed by Rukshan Weerasekara | Creative Technologist & AI Enthusiast")
