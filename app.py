import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_excel("AmesHousing.xlsx")

# Data Preprocessing (Basic example)
df = df.select_dtypes(include=[np.number]).dropna()
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Example using Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit UI
st.title("Ames Housing Price Predictor")

st.write("Enter house details to predict price:")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if st.button("Predict Price"):
    model = pickle.load(open("model.pkl", "rb"))
    pred = model.predict(pd.DataFrame([input_data]))
    st.write(f"Predicted Price: ${pred[0]:,.2f}")

