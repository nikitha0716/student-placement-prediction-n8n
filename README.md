# 🎓 Student Placement Prediction System

An automated student placement prediction system that uses a Random Forest machine learning model, FastAPI, and n8n workflow automation.

## 🚀 Project Overview

The system collects student academic and profile information and predicts whether the student is likely to be placed.

The prediction is performed by a Random Forest classifier, while n8n handles the automation workflow.

## 🏗️ Architecture

Student
↓
n8n Form
↓
HTTP Request
↓
FastAPI
↓
Random Forest Model
↓
Placement Prediction
↓
n8n Result

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- FastAPI
- Uvicorn
- Joblib
- n8n
- ngrok

## 📊 Input Features

The model uses:

- SSC Percentage
- HSC Percentage
- Degree Percentage
- Work Experience
- E-Test Percentage
- Specialisation
- MBA Percentage

## 🤖 Machine Learning Model

The project uses a Random Forest Classifier with:

- 100 decision trees
- Maximum tree depth of 6
- Balanced class weights

The model is trained using student placement data and saved as:

`model/placement_model.pkl`

## 🔌 API

FastAPI exposes the prediction endpoint:

`POST /predict`

Example request:

```json
{
  "ssc_p": 85,
  "hsc_p": 82,
  "degree_p": 80,
  "workex": "Yes",
  "etest_p": 85,
  "specialisation": "Mkt&Fin",
  "mba_p": 78
}
