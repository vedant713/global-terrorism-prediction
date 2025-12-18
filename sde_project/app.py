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
    1.  **Choose Capability:** Use the sidebar to switch between **Prediction Dashboard** and **Global Data Explorer**.
    2.  **Prediction Mode:** Configure details (Date, Country, Tactics) and click **Predict Impact** to estimate potential fatalities.
    3.  **Explorer Mode:** Select a specific Region and Attack Type to visualize similar historical incidents on the map.
    4.  **AI Insights:** Use the **Generate Driver Safety Briefing** button to get an AI-powered security assessment.
    
    *Note: Predictions use an XGBoost model trained on the Global Terrorism Database (GTD).*
    """)

# Sidebar for inputs
with st.sidebar:
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
    
    # --- Navigation ---
    mode = st.sidebar.radio("🔎 Choose Capability:", ["Prediction Dashboard", "Global Data Explorer"], index=0)
    st.sidebar.divider()

    # --- Mode 1: Prediction Dashboard ---
    if mode == "Prediction Dashboard":
        st.sidebar.header("📝 Incident Configuration")
        
        # Date Inputs
        col_d1, col_d2, col_d3 = st.sidebar.columns(3)
        year = col_d1.number_input("Year", 2000, 2030, 2017)
        month = col_d2.number_input("Month", 1, 12, 1)
        day = col_d3.number_input("Day", 1, 31, 1)

        # Location Inputs
        if countries and regions:
            country_name = st.sidebar.selectbox("Country", sorted(countries.values()), index=list(sorted(countries.values())).index("India") if "India" in countries.values() else 0)
            country = next((k for k, v in countries.items() if v == country_name), 4)

            region_name = st.sidebar.selectbox("Region", sorted(regions.values()), index=list(sorted(regions.values())).index("Australasia & Oceania") if "Australasia & Oceania" in regions.values() else 0)
            region = next((k for k, v in regions.items() if v == region_name), 6)
        else:
            st.sidebar.warning("Metadata not loaded. Using codes.")
            country = st.sidebar.number_input("Country ID", 4)
            region = st.sidebar.number_input("Region ID", 6)

        # Attack Details
        attack_type = st.sidebar.selectbox("Attack Type", 
            ["Bombing/Explosion", "Assassination", "Armed Assault", "Kidnapping", "Hijacking", "Unknown"]
        )
        target_type = st.sidebar.selectbox("Target Type", 
            ["Private Citizens & Property", "Military", "Police", "Government (General)", "Business", "Unknown"]
        )
        weapon_type = st.sidebar.selectbox("Weapon Type", 
            ["Explosives", "Firearms", "Incendiary", "Melee", "Chemical", "Unknown"]
        )
        
        predict_btn = st.sidebar.button("⚡ Predict Impact", type="primary")

    # --- Mode 2: Data Explorer Sidebar ---
    elif mode == "Global Data Explorer":
        st.sidebar.header("🌍 Filters")
        
        if regions:
            # Sidebar Filters strictly for Explorer
            exp_region = st.sidebar.selectbox("Select Region", sorted(regions.values()), key="exp_reg")
            exp_reg_id = next((k for k, v in regions.items() if v == exp_region), 1)
            
            exp_attack = st.sidebar.selectbox("Attack Type", 
                 ["Bombing/Explosion", "Assassination", "Armed Assault", "Kidnapping", "Hijacking"],
                 key="exp_att"
            )
            
            load = st.sidebar.button("Update Map", type="primary")
        else:
             st.sidebar.warning("Unable to load regions.")
             st.sidebar.caption("Backend might be starting via Uvicorn...")
             if st.sidebar.button("🔄 Retry Connection"):
                 st.cache_data.clear()
                 st.rerun()

# Main Area for Prediction
if mode == "Prediction Dashboard":
    st.markdown("## 🔮 Prediction Dashboard")
    
    # Initialize Session State for Results
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
        
    if predict_btn:
        payload = {
            "iyear": year, "imonth": month, "iday": day,
            "country": country, "region": region,
            "attacktype1_txt": attack_type,
            "targtype1_txt": target_type,
            "weaptype1_txt": weapon_type
        }
        st.session_state.last_payload = payload
        
        try:
             # Check API
            try:
                requests.get(f"{API_URL}/health")
            except:
                st.error("Backend offline. Run `uvicorn sde_project.api:app --reload`")
                st.stop()

            # Predict
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                st.session_state.prediction_data = response.json()
                st.session_state.show_results = True
                
                # Fetch Context
                hist_res = requests.get(f"{API_URL}/history", params={"country_id": country})
                st.session_state.historical_data = hist_res.json() if hist_res.status_code == 200 else {}

                sim_res = requests.get(f"{API_URL}/similar", params={"region": region, "attack_type": attack_type})
                st.session_state.similar_incidents = sim_res.json().get("incidents", []) if sim_res.status_code == 200 else []
            else:
                 st.error(response.text)
                 st.session_state.show_results = False
                 
        except Exception as e:
            st.error(str(e))
            st.session_state.show_results = False

    # Display Results
    if st.session_state.get("show_results"):
        res = st.session_state.prediction_data
        
        # Top Metrics
        if res.get("status") == "success":
            fatalities = res.get("predicted_fatalities", 0)
            
            # Layout: Metric | Chart
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Threat Assessment")
                st.metric("Expected Fatalities", f"{fatalities}")
                if fatalities > 10: st.error("CRITICAL THREAT")
                elif fatalities > 1: st.warning("ELEVATED THREAT")
                else: st.success("LOW THREAT")
            
            with c2:
                st.markdown("### 🗓️ Historical Trend")
                hist = st.session_state.get("historical_data", {})
                if hist.get("years"):
                    chart_df = pd.DataFrame({"Year": hist["years"], "Attacks": hist["counts"]})
                    st.line_chart(chart_df, x="Year", y="Attacks")
                    st.caption(f"Total Incidents in Country: {hist.get('total_incidents', 'N/A')}")
                else:
                    st.caption("No history available.")

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
                
                # GenAI Section
                st.markdown("---")
                st.subheader("💡 AI Security Briefing")
                if st.button("Generate Driver Safety Advisory", key="gen_ai_dash"):
                    with st.spinner("Consulting AI Analyst..."):
                        sims = st.session_state.get("similar_incidents", [])
                        context = " ".join([i.get("summary", "") for i in sims[:3]])
                        p = st.session_state.last_payload
                        
                        try:
                            g_res = requests.post(f"{API_URL}/genai/advisory", json={
                                "country": str(p.get("country")),
                                "year": str(p.get("iyear")),
                                "attack_type": p.get("attacktype1_txt"),
                                "summary_text": context
                            })
                            if g_res.status_code == 200:
                                advisory_data = g_res.json()
                                st.info(advisory_data["advisory"])
                                st.caption(f"Source: {advisory_data['source']}")
                            else:
                                st.error("AI Service Unavailable")
                        except Exception as e:
                            st.error(f"Connection Failed: {e}")

                # Data Table
                st.markdown("#### Detailed Records")
                df_incidents = pd.DataFrame(incidents)[['iyear', 'city', 'nkill', 'summary']]
                df_incidents['summary'] = df_incidents['summary'].astype(str)
                st.dataframe(df_incidents)
            else:
                st.info("No similar incidents found with geolocation data.")
        else:
            st.error(res.get("message", "An unknown error occurred during prediction."))


# --- Mode 2: Data Explorer ---
elif mode == "Global Data Explorer":
    # Main Area
    st.markdown(f"## 🗺️ Intelligence Map: {exp_region if 'exp_region' in locals() else 'Select a Region'}")
    
    if "exp_loaded" not in st.session_state:
        st.session_state.exp_loaded = False

    # Only run if inputs are defined (i.e. regions were loaded)
    click_load = globals().get('load', False) or locals().get('load', False)
    if (click_load or st.session_state.get("exp_loaded")) and 'exp_reg_id' in locals():
        st.session_state.exp_loaded = True
        
        try:
            # Check API Health
            try:
                health_response = requests.get(f"{API_URL}/health")
                if health_response.status_code != 200:
                    st.error("Backend API is not reachable.")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to Backend API at {API_URL}. Is it running?")
                st.info("Run: uvicorn sde_project.api:app --reload")
                st.stop()

            res = requests.get(f"{API_URL}/similar", params={"region": exp_reg_id, "attack_type": exp_attack})
            if res.status_code == 200:
                data = res.json().get("incidents", [])
                
                if data:
                    st.success(f"Found {len(data)} incidents for {exp_region} - {exp_attack}.")
                    
                    col_map, col_details = st.columns([1, 1])
                    
                    with col_map:
                        st.subheader("📍 Incident Map")
                        map_df_exp = pd.DataFrame(data)[['latitude', 'longitude']].dropna()
                        st.map(map_df_exp)
                        
                    with col_details:
                        st.subheader("📄 Report Context")
                        # Show first incident summary
                        first_inc = data[0]
                        st.markdown(f"**Latest Incident ({first_inc['iyear']}):**")
                        st.info(first_inc.get('summary', 'No summary available.'))
                        
                    # Detailed Table
                    st.markdown("### Full Records")
                    df_exp = pd.DataFrame(data)[['iyear', 'country_txt', 'city', 'nkill', 'summary']]
                    df_exp['summary'] = df_exp['summary'].astype(str)
                    st.dataframe(df_exp.rename(columns={'country_txt': 'Country', 'nkill': 'Fatalities', 'summary': 'Description'}))
                    
                else:
                    st.warning("No incidents found for this selection.")
            else:
                st.error(f"API Error: {res.text}")
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")

# Footer
st.sidebar.divider()
st.sidebar.caption("v1.2.0 | SDE Project")
