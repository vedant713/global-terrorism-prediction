from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Global Terrorism Prediction API")

# Configuration
MODELS_DIR = "sde_project/models"
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.joblib")
DATA_PATH = "gt.csv" 
if not os.path.exists(DATA_PATH) and os.path.exists("../gt.csv"):
    DATA_PATH = "../gt.csv"

# Global variables
model = None
scaler = None
encoders = None
df_data = None

class PredictionRequest(BaseModel):
    iyear: int
    imonth: int
    iday: int
    country: int
    region: int
    attacktype1_txt: str
    targtype1_txt: str
    weaptype1_txt: str

@app.on_event("startup")
def load_artifacts():
    global model, scaler, encoders, df_data
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            print("Model artifacts loaded.")
        else:
            print("Warning: Model artifacts not found.")

        # Load historical data (optimized)
        if os.path.exists(DATA_PATH):
            print("Loading historical data...")
            cols = ['iyear', 'country', 'country_txt', 'region', 'region_txt', 'latitude', 'longitude', 'attacktype1_txt', 'nkill', 'city', 'summary']
            # Load efficiently but safely
            df_data = pd.read_csv(DATA_PATH, encoding='latin1', usecols=cols, low_memory=False)
            
            # Fill NAs first
            df_data['nkill'] = df_data['nkill'].fillna(0)
            df_data['latitude'] = df_data['latitude'].fillna(0)
            df_data['longitude'] = df_data['longitude'].fillna(0)
            df_data.fillna("Unknown", inplace=True) # Fill text cols
            
            # Optimize memory usage
            optimize_types = {
                'iyear': 'int32',
                'country': 'int32',
                'region': 'int32',
                'nkill': 'float32',
                'latitude': 'float32',
                'longitude': 'float32',
                'country_txt': 'category',
                'region_txt': 'category',
                'attacktype1_txt': 'category'
            }
            df_data = df_data.astype(optimize_types)
            
            # Pre-calculate Country Stats for Globe
            global country_stats
            country_stats = df_data.groupby('country_txt').agg({
                'latitude': 'mean',
                'longitude': 'mean',
                'nkill': ['sum', 'count'],
                'country': 'first'
            }).reset_index()
            country_stats.columns = ['country', 'lat', 'lon', 'fatalities', 'incidents', 'country_id']
            country_stats = country_stats.dropna().to_dict(orient='records')
            
            print("Historical data loaded & aggregated.")
        else:
            print("Warning: gt.csv not found. History features will be disabled.")
            country_stats = []

    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.get("/globe_data")
def get_globe_data():
    """Returns aggregated country data for 3D visualization."""
    if df_data is None:
         return {"stats": []}
    return {"stats": country_stats}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "data_loaded": df_data is not None
    }

@app.get("/metadata")
def get_metadata():
    """Returns mappings for Country and Region IDs to Names."""
    if df_data is None:
        return {"countries": {}, "regions": {}}
    
    # Extract unique mappings
    countries = df_data[['country', 'country_txt']].drop_duplicates().sort_values('country_txt')
    regions = df_data[['region', 'region_txt']].drop_duplicates().sort_values('region_txt')
    
    return {
        "countries": dict(zip(countries['country'], countries['country_txt'])),
        "regions": dict(zip(regions['region'], regions['region_txt']))
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        return {"predicted_fatalities": 0, "status": "warning", "message": "Model not loaded."}

    try:
        input_data = {
            'iyear': request.iyear, 'imonth': request.imonth, 'iday': request.iday,
            'country': request.country, 'region': request.region,
            'attacktype1_txt': request.attacktype1_txt,
            'targtype1_txt': request.targtype1_txt,
            'weaptype1_txt': request.weaptype1_txt
        }
        df = pd.DataFrame([input_data])
        
        # Consistent Encoding
        for col in ['attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']:
            if col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col])
                except ValueError:
                    df[col] = 0
        
        features = ['iyear', 'imonth', 'iday', 'country', 'region', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']
        X_scaled = scaler.transform(df[features])
        prediction = max(0, float(model.predict(X_scaled)[0]))
        
        return {"predicted_fatalities": round(prediction, 2), "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(country_id: int):
    """Returns yearly incident counts for a country."""
    if df_data is None:
        return {"status": "error", "message": "Data not loaded"}
    
    country_df = df_data[df_data['country'] == country_id]
    if country_df.empty:
        return {"years": [], "counts": []}
    
    counts = country_df['iyear'].value_counts().sort_index()
    return {
        "years": counts.index.tolist(),
        "counts": counts.values.tolist(),
        "total_incidents": int(country_df.shape[0])
    }

@app.get("/similar")
def get_similar(region: int, attack_type: str):
    """Returns top 50 incidents matching region and attack type for mapping."""
    if df_data is None:
        return {"status": "error", "message": "Data not loaded"}
    
    # Filter by Region and Attack Type
    filtered = df_data[
        (df_data['region'] == region) & 
        (df_data['attacktype1_txt'] == attack_type)
    ]
    
    # Get top 50 recent ones with valid lat/long
    filtered = filtered[filtered['latitude'] != 0].sort_values(by='iyear', ascending=False).head(50)
    
    records = filtered[['iyear', 'latitude', 'longitude', 'city', 'country', 'country_txt', 'nkill', 'summary']].to_dict(orient='records')
    return {"incidents": records}

@app.post("/genai/advisory")
def generate_advisory(request: dict):
    """Generates a safety advisory using Gemini or Mock."""
    country = request.get("country", "Unknown Country")
    year = request.get("year", "Unknown Year")
    summary_text = request.get("summary_text", "")

    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # User requested specific model
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            prompt = f"""
            You are a global security analyst. Based on the following recent terrorism incident data in {country} (circa {year}), 
            provide a concise 3-bullet point travel safety advisory for civilians.
            
            Incident Context: "{summary_text}"
            
            Format:
            - Threat Level: [Low/Medium/High]
            - Key Risk: [One sentence]
            - Advice: [One sentence]
            """
            
            response = model.generate_content(prompt)
            return {"advisory": response.text, "source": "Gemini Pro"}
        except Exception as e:
            print(f"Gemini Error: {e}")
            # Fallback to mock on error
            pass

    # Mock Response (Fail-safe)
    return {
        "advisory": f"""
        **⚠️ Simulated Security Advisory for {country}**
        (API Key not found or Error. Running in Demo Mode)
        
        *   **Threat Level:** High (Simulated)
        *   **Key Risk:** Potential for {request.get('attack_type', 'violent')} incidents in public areas.
        *   **Advice:** Avoid large gatherings and monitor local news outlets.
        """,
        "source": "Mock (Demo Mode)"
    }
