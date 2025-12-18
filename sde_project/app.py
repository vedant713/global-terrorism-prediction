import streamlit as st
import requests
import json
import pandas as pd
import os

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Terrorism Impact Predictor", layout="wide")

st.title("🌍 Global Terrorism Impact Predictor")
st.markdown("Predict the potential fatalities of a terrorist incident based on historical patterns.")

with st.expander("ℹ️ How to use this tool?"):
    st.markdown("""
    1.  **Enter Incident Date:** Select the Year, Month, and Day of the hypothetical incident.
    2.  **Select Location:** Enter the Country and Region codes (Numeric IDs).
    3.  **Choose Incident Characteristics:** Select the Type of Attack, Target, and Weapon used.
    4.  **Click Predict:** The model will estimate the number of fatalities ('nkill').
    
    *Note: The prediction is based on historical data patterns trained on the Global Terrorism Database.*
    """)

# Sidebar for inputs
with st.sidebar:
    st.header("Incident Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Year", min_value=1970, max_value=2030, value=2017)
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=1)

    # Fetch Metadata for Dropdowns
    @st.cache_data
    def get_metadata():
        try:
            res = requests.get(f"{API_URL}/metadata")
            if res.status_code == 200:
                return res.json()
        except:
            pass
        return {"countries": {}, "regions": {}}
    
    metadata = get_metadata()
    countries = metadata.get("countries", {})
    regions = metadata.get("regions", {})
    
    # Country Selection
    if countries:
        # Sort by name
        country_names = sorted(countries.values())
        selected_country_name = st.selectbox("Country", country_names, index=country_names.index("India") if "India" in country_names else 0)
        # Find ID for selected name
        country = next((k for k, v in countries.items() if v == selected_country_name), 4)
        country = int(country)
    else:
         country = st.number_input("Country Code (ID)", min_value=1, value=4)

    # Region Selection
    if regions:
        region_names = sorted(regions.values())
        selected_region_name = st.selectbox("Region", region_names)
        region = next((k for k, v in regions.items() if v == selected_region_name), 1)
        region = int(region)
    else:
        region = st.number_input("Region Code (ID)", min_value=1, value=1)

    # Note: In a real app, these would be populated dynamically from the backend/database
    attack_type = st.selectbox(
        "Attack Type",
        ["Bombing/Explosion", "Assassination", "Armed Assault", "Kidnapping", "Hijacking", "Unknown"]
    )
    
    target_type = st.selectbox(
        "Target Type",
        ["Private Citizens & Property", "Military", "Police", "Government", "Business", "Unknown"]
    )
    
    weapon_type = st.selectbox(
        "Weapon Type",
        ["Explosives", "Firearms", "Incendiary", "Melee", "Chemical", "Unknown"]
    )

    predict_btn = st.button("Predict Impact", type="primary")

# Initialize Session State
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = {}
if "similar_incidents" not in st.session_state:
    st.session_state.similar_incidents = []
if "historical_data" not in st.session_state:
    st.session_state.historical_data = {}
if "last_payload" not in st.session_state:
    st.session_state.last_payload = {}


# Main Logic
if predict_btn:
    # 1. Prepare Payload
    payload = {
        "iyear": year,
        "imonth": month,
        "iday": day,
        "country": country,
        "region": region,
        "attacktype1_txt": attack_type,
        "targtype1_txt": target_type,
        "weaptype1_txt": weapon_type
    }
    st.session_state.last_payload = payload

    try:
        # 2. Check API Health
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code != 200:
                st.error("Backend API is not reachable.")
        except requests.exceptions.ConnectionError:
            st.error(f"Failed to connect to Backend API at {API_URL}. Is it running?")
            st.info("Run: uvicorn sde_project.api:app --reload")
            st.stop()

        # 3. Get Prediction
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            st.session_state.prediction_data = response.json()
            st.session_state.show_results = True
            
            # 4. Fetch History
            hist_res = requests.get(f"{API_URL}/history", params={"country_id": country})
            if hist_res.status_code == 200:
                st.session_state.historical_data = hist_res.json()
            else:
                st.session_state.historical_data = {}

            # 5. Fetch Similar Incidents
            sim_res = requests.get(f"{API_URL}/similar", params={"region": region, "attack_type": attack_type})
            if sim_res.status_code == 200:
                sim_data = sim_res.json()
                st.session_state.similar_incidents = sim_data.get("incidents", [])
            else:
                st.session_state.similar_incidents = []

        else:
            st.error(f"Error from API: {response.text}")
            st.session_state.show_results = False

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.show_results = False


# Render Results (Persistent)
if st.session_state.show_results:
    result = st.session_state.prediction_data
    
    if result.get("status") == "warning":
        st.warning(result["message"])
    else:
        fatalities = result.get("predicted_fatalities", 0)
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Predicted Fatalities", value=f"{fatalities}")
            if fatalities > 10:
                st.error("High Impact Incident Expected")
            elif fatalities > 1:
                st.warning("Moderate Impact Incident Expected")
            else:
                st.success("Low Impact Incident Expected")
        
        with col_res2:
            st.markdown("### 📊 Historical Context")
            hist_data = st.session_state.historical_data
            if "years" in hist_data and hist_data["years"]:
                    chart_df = pd.DataFrame({"Year": hist_data["years"], "Incidents": hist_data["counts"]})
                    st.line_chart(chart_df, x="Year", y="Incidents")
                    st.caption(f"Total Incidents in Country: {hist_data.get('total_incidents', 'N/A')}")
            else:
                msg = hist_data.get("message", "No historical data found for this country.")
                st.info(msg)

    # Similar Incidents Section
    st.divider()
    incidents = st.session_state.similar_incidents
    region_val = st.session_state.last_payload.get("region", region)
    attack_val = st.session_state.last_payload.get("attacktype1_txt", attack_type)
    
    st.subheader(f"🔍 Recent Similar Incidents (Region {region_val} - {attack_val})")
    
    if incidents:
        # Map Visualization
        map_df = pd.DataFrame(incidents)[['latitude', 'longitude']].dropna()
        st.map(map_df)
        
        # --- GenAI Section ---
        st.markdown("---")
        st.subheader("🤖 AI Security Analyst")
        
        # We use a unique key for the button to avoid state conflicts
        if st.button("Generate Driver Safety Briefing", key="gen_ai_btn"):
            with st.spinner("Analyzing historical data..."):
                # Prepare context from similar incidents
                summary_context = " ".join([inc.get("summary", "") for inc in incidents[:3]])
                gen_payload = {
                    "country": str(st.session_state.last_payload.get("country")),
                    "year": str(st.session_state.last_payload.get("iyear")),
                    "attack_type": attack_val,
                    "summary_text": summary_context
                }
                
                try:
                    gen_res = requests.post(f"{API_URL}/genai/advisory", json=gen_payload)
                    if gen_res.status_code == 200:
                        advisory_data = gen_res.json()
                        st.info(advisory_data["advisory"])
                        st.caption(f"Source: {advisory_data['source']}")
                    else:
                        st.error("Failed to generate advisory.")
                except Exception as e:
                    st.error(f"Error connecting to AI service: {e}")
        # ---------------------

        # Data Table
        st.markdown("#### Detailed Records")
        df_incidents = pd.DataFrame(incidents)[['iyear', 'city', 'nkill', 'summary']]
        df_incidents['summary'] = df_incidents['summary'].astype(str)
        st.dataframe(df_incidents)
    else:
        st.info("No similar incidents found with geolocation data.")

# Footer info
st.divider()
st.caption("Powered by XGBoost & FastAPI | SDE Project Demo")
