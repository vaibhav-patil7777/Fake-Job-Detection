# Fake Job Detection Project

## Overview
This project is a Fake Job Detection System that predicts whether a job posting is genuine or fraudulent. The system uses features from job listings such as title, description, requirements, and company information.  

The project uses natural language processing techniques including TF-IDF for text data and OneHotEncoding for categorical features. Multiple machine learning models were trained to determine the best-performing model.  

A Streamlit web interface allows users to input job details and receive real-time predictions.

---

## Dataset
- Dataset used: [Fake Job Posting Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
- Features included:
  - job_id, title, location, department, salary_range, company_profile
  - description, requirements, benefits, telecommuting, has_company_logo
  - has_questions, employment_type, required_experience, required_education
  - industry, function, fraudulent (target variable)

The dataset contains both genuine and fraudulent job postings.

---

## Features
- OneHotEncoding for categorical features
- TF-IDF vectorization for text features
- Combined text and categorical/numerical features for model input
- Predicts fraudulent jobs (0 = Genuine, 1 = Fraudulent)

---

## Models Trained
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 97.79% |
| Random Forest | 98.01% |
| Gradient Boosting | 97.90% |
| AdaBoost | 97.09% |
| SVM | 97.76% |

The Random Forest model was selected as the final model due to its highest accuracy of 98.01%.

---

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_link>
   cd fake-job-detection
pip install -r requirements.txt

streamlit run app.py
