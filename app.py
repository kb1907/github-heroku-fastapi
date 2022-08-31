from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

app = FastAPI()
model = pickle.load(open("catboost_model-2.pkl", "rb"))


def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.get("/")
async def root():
    return {"message": "Prediction"}


@app.get("/predict")
async def predict(
    Age: int,
    RestingBP: int,
    Cholesterol: int,
    Oldpeak: float,
    FastingBS: int,
    MaxHR: int,
):
    prediction = model.predict(
        [[Age, RestingBP, Cholesterol, Oldpeak, FastingBS, MaxHR]]
    )
    if prediction == 0:
        return {"You are well. No worries :)"}
    else:
        return {
            f"Prediction: {prediction}. Kindly make an appointment with the doctor!"
        }


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
