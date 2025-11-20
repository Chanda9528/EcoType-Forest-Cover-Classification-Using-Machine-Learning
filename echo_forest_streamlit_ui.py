import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model
model = joblib.load("best_model.pkl")

st.title("Forest Cover Type Prediction App")

# 3 Columns for main numeric inputs
col1, col2, col3 = st.columns(3)

with col1:
    Elevation = st.number_input("Elevation")
    Aspect = st.number_input("Aspect")
    Slope = st.number_input("Slope")
    Horizontal_Distance_To_Hydrology = st.number_input("Horizontal_Distance_To_Hydrology")
    Vertical_Distance_To_Hydrology = st.number_input("Vertical_Distance_To_Hydrology")

with col2:
    Horizontal_Distance_To_Roadways = st.number_input("Horizontal_Distance_To_Roadways")
    Hillshade_9am = st.number_input("Hillshade_9am")
    Hillshade_Noon = st.number_input("Hillshade_Noon")
    Hillshade_3pm = st.number_input("Hillshade_3pm")
    Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal_Distance_To_Fire_Points")

with col3:
    Wilderness_Area_1 = st.number_input("Wilderness_Area_1", 0, 1)
    Wilderness_Area_2 = st.number_input("Wilderness_Area_2", 0, 1)
    Wilderness_Area_3 = st.number_input("Wilderness_Area_3", 0, 1)
    Wilderness_Area_4 = st.number_input("Wilderness_Area_4", 0, 1)

# Derived feature (AUTO, NOT USER INPUT)
Distance_To_Water = Horizontal_Distance_To_Hydrology - Vertical_Distance_To_Hydrology

# Soil Type Inputs Section
st.subheader("Soil Types (Binary 0/1)")
s1, s2, s3, s4 = st.columns(4)
soil_features = []

for i in range(1, 41):
    col = [s1, s2, s3, s4][(i-1) % 4]
    soil_features.append(col.number_input(f"Soil_Type_{i}", 0, 1))

# Prediction Button
if st.button("Predict Cover Type"):
    input_data = np.array([[
        Elevation, Aspect, Slope,
        Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Distance_To_Water,
        Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon,
        Hillshade_3pm, Horizontal_Distance_To_Fire_Points,
        Wilderness_Area_1, Wilderness_Area_2, Wilderness_Area_3, Wilderness_Area_4,
        *soil_features
    ]])

    result = model.predict(input_data)
    st.success(f"Predicted Forest Cover Type: {result[0]}")
