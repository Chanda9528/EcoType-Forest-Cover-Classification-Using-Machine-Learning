# EcoType-Forest-Cover-Classification-Using-Machine-Learning
This project applies machine learning to classify forest cover types using 54 environmental features such as elevation, slope, hillshade, soil type, and hydrology distances. The workflow includes data preprocessing, feature engineering, model training and comparison and deployment through a Streamlit application for real-time cover type prediction.
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
