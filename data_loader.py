import pandas as pd

def load_data(path="data/nids.csv"):
    return pd.read_csv(path)
