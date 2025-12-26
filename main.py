from src.data_loader import load_data
from src.preprocess import preprocess
from src.train import train_model, save_model
from src.evaluate import evaluate

def main():
    df = load_data("data/nids.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model, scaler)

if __name__ == "__main__":
    main()
