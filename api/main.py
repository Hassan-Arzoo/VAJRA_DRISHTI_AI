from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel,Field
from vajradrishti.model import train_model
from vajradrishti.pipeline import run_prediction
from vajradrishti.synthetic import make_event

app=FastAPI(title="VajraDrishti AI PoC",version="0.1.0",description="Synthetic multimodal nowcast demo, not an operational warning service.")
models=train_model()
class PredictionRequest(BaseModel):
    event_id:str|None=None
    timestamp:str|None=None
    radar:dict[str,Any]=Field(default_factory=dict)
    satellite:dict[str,Any]=Field(default_factory=dict)
    lightning:dict[str,Any]=Field(default_factory=dict)
    atmosphere:dict[str,Any]=Field(default_factory=dict)
@app.get("/health")
def health(): return {"status":"ok","model":"random_forest","data_mode":"synthetic demo"}
@app.get("/sample-data")
def sample_data(): return make_event()
@app.post("/predict")
def prediction(request:PredictionRequest): return run_prediction(request.model_dump(),models)
@app.get("/prediction/{event_id}")
def prediction_by_id(event_id:str):
    event=make_event(); event["event_id"]=event_id
    return run_prediction(event,models)
