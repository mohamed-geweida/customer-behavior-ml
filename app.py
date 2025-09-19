# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("customer_behavior_unsupervised.csv")

# Create synthetic spending
np.random.seed(42)
df["spending"] = (
    50 * df["feature_0"] +
    30 * df["feature_2"] -
    20 * df["feature_4"] +
    np.random.normal(0, 10, size=len(df))
)
# shift to avoid negatives
df["spending"] = df["spending"] - df["spending"].min() + 1

# Features & Target
X = df.drop("spending", axis=1)
y = df["spending"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Linear Regression
reg = LinearRegression()
reg.fit(X_scaled, y)

st.title("💰 Customer Spending Prediction App")
st.write("Enter customer features to predict expected spending using Linear Regression with MinMaxScaler.")

# Input fields for features
feat0 = st.number_input("feature_0", value=0.0)
feat1 = st.number_input("feature_1", value=0.0)
feat2 = st.number_input("feature_2", value=0.0)
feat3 = st.number_input("feature_3", value=0.0)
feat4 = st.number_input("feature_4", value=0.0)

user_features = np.array([[feat0, feat1, feat2, feat3, feat4]])

if st.button("Predict Spending"):
    scaled = scaler.transform(user_features)
    pred_spending = reg.predict(scaled)[0]

    st.subheader("🔮 Prediction Result")
    st.write(f"**Predicted Spending:** {pred_spending:.2f}")
