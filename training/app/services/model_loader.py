import joblib

def load_model():
    model = joblib.load("model/svm_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler