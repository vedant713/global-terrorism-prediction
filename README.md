# 🌍 Global Terrorism Prediction Platform (SDE Extension)

This project transforms a traditional Data Science analysis into a production-grade **Full-Stack Machine Learning Application**. It features a resilient Backend API, an interactive Frontend Dashboard, and is fully Dockerized for cross-platform deployment.

## 🚀 Key Features

*   **Prediction Service:** Real-time fatality estimation using a trained **XGBoost Regressor**.
*   **Interactive Dashboard:**
    *   **Geospatial Visualization:** Dynamic map showing similar historical incidents in the selected region.
    *   **Trend Analytics:** Historical incident charts for the selected country.
    *   **AI Security Analyst:** 🤖 Generates travel safety briefings using LLMs (Gemini) based on incident history.
    *   **Smart Inputs:** Dropdowns populated dynamically from dataset metadata.
*   **REST API:** Fully documented endpoints via FastAPI (Swagger UI included).
*   **Production Ready:** Docker containerization ensures consistent execution on Windows, Mac, and Linux.

## 🏗 Architecture

1.  **Training Pipeline (`sde_project/train_pipeline.py`)**:
    *   Modular script for data ingestion, preprocessing (LabelEncoding, Scaling), and model training.
    *   Serializes model artifacts (`.joblib`) for inference.
2.  **Backend API (`sde_project/api.py`)**:
    *   **FastAPI** microservice serving the model.
    *   Endpoints:
        *   `POST /predict`: Get fatality predictions.
        *   `GET /history`: Fetch yearly trends for charting.
        *   `GET /similar`: Fetch geolocation data for maps.
        *   `GET /metadata`: Dynamic form population.
3.  **Frontend Dashboard (`sde_project/app.py`)**:
    *   **Streamlit** interface for user interaction.
    *   Connects to the backend via REST API calls.

## 🔧 Technologies
*   **Core:** Python 3.9+
*   **ML:** XGBoost, Scikit-learn, Pandas, NumPy
*   **Web/API:** FastAPI, Uvicorn, Streamlit
*   **DevOps:** Docker, Docker Compose

---

## 🛠️ Setup & Usage

### 📋 Prerequisites
*   Ensure the dataset `gt.csv` is present in the project root.

### Option 1: Docker (Recommended) 🐳
The easiest way to run the full stack (Frontend + Backend).

1.  **Build and Run**:
    ```bash
    # Run from the project root directory
    docker-compose up --build
    ```
2.  **Access the App**:
    *   **Dashboard:** [http://localhost:8501](http://localhost:8501)
    *   **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Development 💻

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Navigate to Project Folder**:
    ```bash
    cd sde_project
    ```

3.  **Train the Model**:
    ```bash
    python train_pipeline.py
    ```

4.  **Start the Backend**:
    ```bash
    uvicorn api:app --reload
    ```

5.  **Start the Frontend** (in a new terminal):
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
```
├── gt.csv                  # Dataset (Input)
├── docker-compose.yml      # Docker Envrionment Definition
├── requirements.txt        # Python Dependencies
├── sde_project/
│   ├── Dockerfile          # Container Definition
│   ├── train_pipeline.py   # ML Training Script
│   ├── api.py              # Backend API
│   ├── app.py              # Frontend Dashboard
│   └── models/             # Saved Model Artifacts
```
