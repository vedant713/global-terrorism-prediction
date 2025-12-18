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

# Main Content
if predict_btn:
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

    try:
        # Check API Health
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code != 200:
                st.error("Backend API is not reachable.")
        except requests.exceptions.ConnectionError:
            st.error(f"Failed to connect to Backend API at {API_URL}. Is it running?")
            st.info("Run: uvicorn sde_project.api:app --reload")
            st.stop()

        # Get Prediction
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "warning":
                st.warning(result["message"])
            else:
                fatalities = result["predicted_fatalities"]
                
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
                    # Fetch History
                    hist_res = requests.get(f"{API_URL}/history", params={"country_id": country})
                    if hist_res.status_code == 200:
                        hist_data = hist_res.json()
                        if "years" in hist_data and hist_data["years"]:
                             chart_df = pd.DataFrame({"Year": hist_data["years"], "Incidents": hist_data["counts"]})
                             st.line_chart(chart_df, x="Year", y="Incidents")
                             st.caption(f"Total Incidents in Country {country}: {hist_data['total_incidents']}")
                        else:
                            msg = hist_data.get("message", "No historical data found for this country.")
                            st.info(msg)

        else:
            st.error(f"Error from API: {response.text}")
            
        # Similar Incidents Section
        st.divider()
        st.subheader(f"🔍 Recent Similar Incidents (Region {region} - {attack_type})")
        
        sim_res = requests.get(f"{API_URL}/similar", params={"region": region, "attack_type": attack_type})
        if sim_res.status_code == 200:
            sim_data = sim_res.json()
            incidents = sim_data.get("incidents", [])
            
            if incidents:
                # Map Visualization
                map_df = pd.DataFrame(incidents)[['latitude', 'longitude']].dropna()
                st.map(map_df)
                
                # Data Table
                st.markdown("#### Detailed Records")
                st.dataframe(pd.DataFrame(incidents)[['iyear', 'city', 'nkill', 'summary']])
            else:
                st.info("No similar incidents found with geolocation data.")
        else:
            st.warning("Could not fetch similar incidents.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Footer info
st.divider()
st.caption("Powered by XGBoost & FastAPI | SDE Project Demo")
