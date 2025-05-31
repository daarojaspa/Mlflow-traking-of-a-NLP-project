import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.db import engine, create_db_and_tables, PredictionsTickets
from app.utils import preprocessing_fn
from sqlmodel import Session, select
from enum import Enum

app = FastAPI(title="FastAPI, Docker, and Traefik")
global label_mapping

label_mapping = {
    "0": "Bank Account Services",
    "1": "Credit Report or Prepaid Card",
    "2": "Mortgage/Loan"}

class sentence(BaseModel):
    #difines data structure for each imput
    client_name:str
    text:str
class ProcessTextRequestModel(BaseModel):
    sentences:list[sentence]
#entrypoint
@app.post("/predict")
async def read_root(data:ProcessTextRequestModel):
    session=Session(engine)
    model=joblib.load("model.pkl")
    pred_list=[]
    for sentence in data.sentences:
        processed_data_vectoriced=preprocessing_fn(sentence.text)
        X_dense =[sparse_matrix.toArroay() for sparse_matrix in processed_data_vectoriced]
        X_dense=np.vstack(X_dense)
        preds=model.predicts(X_dense)
        decoded_predictions= label_mapping[str(preds[0])]
        #create object with predictions
        prediction_tikets=PredictionsTickets(
            client_name=sentence.client_name,
            predictions=decoded_predictions
        )
        pprint(prediction_ticket)
        pred_list.ppend(
            {
                "client_name":sentence.client_name,
                "prediction":decoded_predictions
            }
        )
        session.add(prediction_tikets)
        session.commit#bulk
        session.close
        return {"predictions":pred_list}