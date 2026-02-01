import streamlit as st
import numpy as np
from joblib import load

# Load model and scaler
model = load("model.pkl")
scaler = load("scaler.pkl")

st.title("Boston Housing Price Prediction")

crim = st.number_input("CRIM")
rm = st.number_input("RM")
lstat = st.number_input("LSTAT")
ptratio = st.number_input("PTRATIO")

if st.button("Predict"):
    X = np.array([[crim, rm, lstat, ptratio]])
    X = scaler.transform(X)
    pred = model.predict(X)
    st.success(f"Predicted House Price: {pred[0]:.2f}")
