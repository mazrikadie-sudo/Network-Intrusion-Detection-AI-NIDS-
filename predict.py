import joblib, numpy as np

def predict(flow_features):
    model = joblib.load("models/nids_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    flow_features = np.array(flow_features).reshape(1, -1)
    flow_features = scaler.transform(flow_features)

    return model.predict(flow_features)[0]
