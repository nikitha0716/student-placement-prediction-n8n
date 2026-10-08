from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

model = joblib.load(
    "model/placement_model.pkl"
)


# -----------------------------------------
# CREATE FASTAPI APP
# -----------------------------------------

app = FastAPI(
    title="Student Placement Prediction API",
    description="Random Forest based placement prediction system"
)


# -----------------------------------------
# INPUT FORMAT
# -----------------------------------------

class StudentData(BaseModel):

    ssc_p: float
    hsc_p: float
    degree_p: float

    workex: str

    etest_p: float

    specialisation: str

    mba_p: float


# -----------------------------------------
# HOME
# -----------------------------------------

@app.get("/")
def home():

    return {
        "message": "Student Placement Prediction API is running",
        "model": "Random Forest"
    }


# -----------------------------------------
# PREDICTION
# -----------------------------------------

@app.post("/predict")
def predict(data: StudentData):

    student = pd.DataFrame([{

        "ssc_p": data.ssc_p,

        "hsc_p": data.hsc_p,

        "degree_p": data.degree_p,

        "workex": data.workex,

        "etest_p": data.etest_p,

        "specialisation": data.specialisation,

        "mba_p": data.mba_p

    }])


    # Prediction
    prediction = model.predict(student)[0]


    # Probability
    probabilities = model.predict_proba(student)[0]

    not_placed_probability = probabilities[0]

    placed_probability = probabilities[1]


    # Convert prediction
    if prediction == 1:

        result = "PLACED"

    else:

        result = "NOT PLACED"


    # Risk level
    if placed_probability >= 0.75:

        risk = "LOW"

    elif placed_probability >= 0.50:

        risk = "MEDIUM"

    else:

        risk = "HIGH"


    return {

        "prediction": result,

        "placed_probability":
            round(
                placed_probability * 100,
                2
            ),

        "not_placed_probability":
            round(
                not_placed_probability * 100,
                2
            ),

        "risk_level": risk

    }