import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack
import re

# ---------------- Load Model and Encoders ----------------
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ---------------- Text Cleaning Function ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- Prediction Function ----------------
def predict_job(title, description, requirements, company_profile="", benefits="",
                telecommuting=0, has_company_logo=0, has_questions=0,
                employment_type="Unknown", required_experience="Unknown",
                required_education="Unknown", industry="Unknown", function="Unknown"):

    text = clean_text(f"{title} {company_profile} {description} {requirements} {benefits}")
    text_vect = vectorizer.transform([text])

    cat_values = [[employment_type, required_experience, required_education, industry, function]]
    cat_encoded = encoder.transform(cat_values)

    X_input = hstack([text_vect, np.array([[telecommuting, has_company_logo, has_questions]]), cat_encoded])

    prob = best_model.predict_proba(X_input)[0][1]
    prediction = "Fake" if prob > 0.6 else "Real" if prob < 0.4 else "Not Sure"

    return prediction, prob

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Fake Job Detection", page_icon="🕵️‍♂️", layout="wide")

# CSS
st.markdown("""
<style>
body {background-color: #f0f2f6; font-family: 'Segoe UI', sans-serif;}
h1 {color: #0d6efd; text-align: center;}
.stButton>button {background-color: #0d6efd; color: white; font-size: 18px; padding: 10px 20px; border-radius: 8px; width: 200px; margin: 5px;}
.stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div>select {padding: 12px; border-radius: 8px;}
hr {border: 1px solid #ccc; margin-top: 20px;}
.center {display: flex; justify-content: center; align-items: center;}
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ Fake Job Posting Detection")

# ---------------- Input Fields Placeholder ----------------
input_container = st.container()
with input_container:
    st.subheader("Enter Job Posting Details:")
    title = st.text_input("Job Title", "")
    description = st.text_area("Job Description", "", height=150)
    requirements = st.text_area("Requirements", "", height=100)

    with st.expander("Optional Details (Advanced)"):
        company_profile = st.text_area("Company Profile", "")
        benefits = st.text_area("Benefits", "")
        telecommuting = st.checkbox("Telecommuting", value=False)
        has_company_logo = st.checkbox("Has Company Logo", value=False)
        has_questions = st.checkbox("Has Screening Questions", value=False)
        employment_type = st.selectbox("Employment Type", ["Unknown", "Full-time", "Part-time", "Contract", "Temporary", "Internship"])
        required_experience = st.selectbox("Required Experience", ["Unknown", "Entry-level", "Mid-level", "Senior-level", "Director", "Executive"])
        required_education = st.selectbox("Required Education", ["Unknown", "High School", "Associate", "Bachelor's", "Master's", "Doctorate"])
        industry = st.text_input("Industry", "Unknown")
        function = st.text_input("Function", "Unknown")

# ---------------- Buttons at the bottom ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="center">', unsafe_allow_html=True)
    predict_clicked = st.button("Predict")
    refresh_clicked = st.button("Refresh")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Refresh Button Action ----------------
if refresh_clicked:
    input_container.empty()  # Clear all input fields
    st.success("All fields cleared! You can enter new job details.")

# ---------------- Predict Button Action ----------------
if predict_clicked:
    if title == "" or description == "" or requirements == "":
        st.error("Please fill in at least Title, Description, and Requirements!")
    else:
        pred, prob = predict_job(
            title, description, requirements,
            company_profile, benefits,
            int(telecommuting), int(has_company_logo), int(has_questions),
            employment_type, required_experience,
            required_education, industry, function
        )
        st.markdown(f"### Prediction: **{pred}**")
        st.markdown(f"### Probability of Fake: **{prob*100:.2f}%**")

# ---------------- Footer ----------------
st.markdown("<hr><p style='text-align:center;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
