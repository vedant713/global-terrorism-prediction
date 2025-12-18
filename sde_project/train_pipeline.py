import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import sys

# Configuration
DATA_PATH = "../gt.csv"  # Assumes dataset is in the parent directory
MODELS_DIR = "sde_project/models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        print("Please place 'gt.csv' in the project root.")
        return None
    
    print("Loading dataset...")
    # Loading with low_memory=False to avoid DtypeWarning equivalent
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Handling missing values for critical columns
    df['city'].fillna('Unknown', inplace=True)
    df['provstate'].fillna('Unknown', inplace=True)
    df.fillna(0, inplace=True)

    # Feature Selection
    features = ['iyear', 'imonth', 'iday', 'country', 'region', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']
    target = 'nkill'

    # Filter data to ensure columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return None, None, None, None

    # Encoding Categorical Variables
    encoders = {}
    categorical_cols = ['attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str) # Ensure string for encoding
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Save Encoders
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
    print("Encoders saved.")

    X = df[features]
    y = df[target]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("Scaler saved.")

    return X_scaled, y, scaler, encoders

def train_model(X, y):
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Model Training Complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Save Model
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    print("Model saved.")

    return model

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    if df is not None:
        X, y, scaler, encoders = preprocess_data(df)
        if X is not None:
            train_model(X, y)
            print("Pipeline successfully completed.")
        else:
            print("Preprocessing failed.")
    else:
        print("Pipeline aborted due to missing data.")
