from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train_model(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=150)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler):
    joblib.dump(model, "models/nids_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
