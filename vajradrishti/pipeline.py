from .features import extract_features
from .model import predict

def run_prediction(payload,models):
    features,availability=extract_features(payload)
    preds=predict(models,features)
    fraction=sum(availability.values())/len(availability)
    confidence=round(.9*fraction+.1*fraction*fraction,3)
    return {"event_id":payload.get("event_id"),"timestamp":payload.get("timestamp"),"predictions":{"lightning_probability":preds["lightning"],"thunderstorm_probability":preds["thunderstorm"]},"confidence":confidence,"data_availability":availability,"features":features,"data_mode":"SYNTHETIC / DEMO baseline; not operational guidance"}
