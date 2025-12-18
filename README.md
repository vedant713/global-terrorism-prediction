# 🌍 Global Terrorism Analysis & Prediction

This project is an exploratory data analysis and machine learning implementation on the **Global Terrorism Database (GTD)**. The goal is to analyze patterns of terrorist activities worldwide and build predictive models for fatalities and attack types.

## 📌 Table of Contents
- [📊 Data Analysis](#-data-analysis)
- [📈 Machine Learning Models](#-machine-learning-models)
- [⚙️ Installation](#-installation)
- [🚀 Running the Code](#-running-the-code)
- [🔍 Results & Insights](#-results--insights)
- [📜 License](#-license)

---

## 📊 Data Analysis
The dataset used contains global terrorism incidents from 1970 to recent years. The following analyses were performed:

✔️ **Missing Data Handling**: Filled missing values appropriately.  
✔️ **Top Countries & Regions**: Visualized attack frequency across regions.  
✔️ **Attack Trends**: Time-series analysis of attack frequency over the years.  
✔️ **Attack Methods & Weapons**: Analysis of common attack types and weapons used.  
✔️ **Targeted Victims & Groups**: Examined the most affected groups and nationalities.  
✔️ **Geographical Distribution**: Plotted incidents on maps for specific countries.  

**Visualizations**:  
📊 Bar charts, 📈 Line plots, 📍 Geographic clustering, 🎯 Scatter plots.

---

## 📈 Machine Learning Models
### 🔹 Regression Model: Predicting Fatalities
- **Algorithm**: XGBoost Regressor  
- **Features**: Year, month, day, country, region, attack type, target type, weapon type  
- **Performance Metrics**:  
  - **MAE** (Mean Absolute Error)  
  - **MSE** (Mean Squared Error)  
  - **RMSE** (Root Mean Squared Error)  

### 🔹 Classification Model: Predicting Attack Type
- **Algorithm**: Random Forest Classifier & XGBoost Classifier  
- **Hyperparameter Tuning**: Grid Search for optimization  
- **Performance Metrics**:  
  - **Accuracy**  
  - **Classification Report (Precision, Recall, F1-score)**  

### 🔹 Deep Learning Models (Neural Networks)
- **Regression**: Multi-layered neural network for predicting fatalities  
- **Classification**: Neural network for predicting attack type (multi-class classification)  

---

## ⚙️ Installation
### Prerequisites:
- Python 3.x
- Required Libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow folium
  ```
