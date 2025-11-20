# EcoType � Forest Cover Classification

## Project Overview
EcoType is a machine learning project that predicts forest cover type based on cartographic and environmental attributes.
The project includes data preprocessing, model training, and deployment using a Streamlit interface.

## Steps to Run

1. Run `echo_forest_backend.py`
   - Preprocessing: missing, outlier, skew handling
   - SMOTE balancing
   - Model comparison & best model selection
   - Saves trained model as `best_model.pkl`

2. Run `echo_forest_streamlit_ui.py`
   - Opens Streamlit interface at: http://localhost:8501
   - Enter feature values and predict forest cover type

## Tools Used
- Python
- NumPy, Pandas, Scikit-Learn, SMOTE
- Matplotlib, Seaborn
- Streamlit
- Joblib

