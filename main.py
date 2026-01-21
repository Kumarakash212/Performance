import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

import streamlit as st

# -------------------------------
# FIRST PAGE / DASHBOARD INTRO
# -------------------------------
st.set_page_config(page_title="Enterprise Sales Analytics Dashboard", layout="wide")

# Title and Subtitle
st.title("📊 Enterprise Sales Analytics Dashboard")
st.subheader("Transform your sales data into actionable insights")

# About / Description
st.markdown("""
Welcome to the **Enterprise Sales Analytics Dashboard**!  
This tool allows you to **analyze, visualize, and forecast your sales data** in a dynamic and interactive way.  
Easily upload your CSV dataset or connect to a database and start exploring trends, KPIs, and top performers.
""")

# Features Section
st.markdown("### 🚀 Features")
st.markdown("""
- **Dynamic KPI Metrics**: Total, Average, Maximum of any numeric column  
- **Time-Series Analysis**: Visualize trends over time  
- **Top Performers**: Identify top products, regions, or categories  
- **Forecasting**: Predict future sales using **Prophet** and **LSTM** models  
- **Flexible Dataset Support**: Works with any CSV containing a date and numeric column  
- **Downloadable Reports**: Export analyzed data for further use  
- **Secure Login & Signup**: Protect your data with username/password authentication  
""")

# Key Benefits
st.markdown("### 💡 Benefits")
st.markdown("""
- Make informed business decisions with clear insights  
- Save time with automated data analysis  
- Identify growth opportunities and top-performing segments  
- Easily visualize historical trends and predict future sales  
- Works with minimal dataset requirements (1 date + 1 numeric column)  
""")

# Optional: Add an image / illustration
st.image("https://cdn-icons-png.flaticon.com/512/3271/3271148.png", width=250)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Dynamic Enterprise Dashboard", layout="wide")

# -------------------------------
# USER DATABASE
# -------------------------------
USER_FILE = "users.csv"

# Create CSV if not exists
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)

# Initialize session state for login persistence
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# -------------------------------
# AUTHENTICATION
# -------------------------------
st.sidebar.title("🔐 Authentication")
auth_mode = st.sidebar.radio("Choose", ["Login", "Sign Up"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

users_df = pd.read_csv(USER_FILE)

# -------------------------------
# SIGNUP
# -------------------------------
if auth_mode == "Sign Up":
    if st.sidebar.button("Create Account"):
        if username.strip() == "" or password.strip() == "":
            st.sidebar.error("Username and password cannot be empty")
        elif username in users_df["username"].values:
            st.sidebar.error("Username already exists")
        else:
            # Append new user
            new_user = pd.DataFrame([[username, password]], columns=["username","password"])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USER_FILE, index=False)
            st.sidebar.success("Account created! Please switch to Login")

# -------------------------------
# LOGIN
# -------------------------------
if auth_mode == "Login":
    if st.sidebar.button("Login"):
        # Reload CSV to get the latest signup users
        users_df = pd.read_csv(USER_FILE)
        match = ((users_df["username"] == username) & (users_df["password"] == password)).any()
        if match:
            st.session_state["logged_in"] = True
            st.sidebar.success("Logged in successfully")
        else:
            st.session_state["logged_in"] = False
            st.sidebar.error("Invalid credentials")

# -------------------------------
# STOP IF NOT LOGGED IN
# -------------------------------
if not st.session_state["logged_in"]:
    st.stop()


# -------------------------------
# SIDEBAR: DATASET UPLOAD
# -------------------------------
st.sidebar.title("📂 Upload Your Dataset")
st.sidebar.info("""
**Dataset Requirements:**  
1. Must contain **at least 1 date column** (e.g., `date`).  
2. Must contain **at least 1 numeric column** (e.g., `revenue`, `units_sold`).  
3. Optional: **Categorical columns** for grouping (e.g., `product`, `region`).  

✅ The dashboard will automatically detect available columns and analyze the data accordingly.

**Example Dataset:**
| date       | revenue | units_sold | product   | region  |
|------------|---------|------------|----------|--------|
| 2023-01-01 | 1000    | 10         | Product A| North  |
| 2023-01-02 | 1500    | 15         | Product B| South  |
| 2023-01-03 | 1200    | 12         | Product A| East   |
""")

# File uploader
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Example: show preview if file uploaded
if file:
    df = pd.read_csv(file)
    st.sidebar.success("Dataset loaded successfully!")
    st.sidebar.dataframe(df.head())

# --------------------------------------------------
# AUTO DETECT COLUMNS
# --------------------------------------------------
date_cols = df.select_dtypes(include=["datetime", "object"]).columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Convert first date-like column
for col in date_cols:
    try:
        df[col] = pd.to_datetime(df[col])
        date_col = col
        break
    except:
        continue
else:
    st.error("No date column found")
    st.stop()

df = df.sort_values(date_col)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.title("🎛 Filters")

metric = st.sidebar.selectbox("Select Numeric Metric", num_cols)

group_col = st.sidebar.selectbox(
    "Group By (Optional)",
    ["None"] + list(cat_cols)
)

start, end = st.sidebar.date_input(
    "Date Range",
    [df[date_col].min(), df[date_col].max()]
)

df = df[(df[date_col] >= pd.to_datetime(start)) &
        (df[date_col] <= pd.to_datetime(end))]

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
st.title("📊 Dynamic Enterprise Analytics Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total", f"{df[metric].sum():,.2f}")
col2.metric("Average", f"{df[metric].mean():,.2f}")
col3.metric("Max Value", f"{df[metric].max():,.2f}")

# --------------------------------------------------
# TIME AGGREGATION
# --------------------------------------------------
df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby("month")[metric].sum().reset_index()

# --------------------------------------------------
# TREND CHART
# --------------------------------------------------
fig_trend = px.line(
    monthly,
    x="month",
    y=metric,
    title="Trend Over Time"
)
st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------------------------
# GROUP ANALYSIS
# --------------------------------------------------
if group_col != "None":
    top_group = (
        df.groupby(group_col)[metric]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_group = px.bar(
        top_group,
        x=group_col,
        y=metric,
        title=f"Top {group_col}"
    )
    st.plotly_chart(fig_group, use_container_width=True)

# --------------------------------------------------
# PROPHET FORECAST
# --------------------------------------------------
st.subheader("🔮 Forecast (Prophet)")

try:
    p_df = monthly.rename(columns={"month": "ds", metric: "y"})
    model = Prophet()
    model.fit(p_df)

    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    fig_prophet = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Future Forecast"
    )
    st.plotly_chart(fig_prophet, use_container_width=True)
except:
    st.warning("Forecasting not possible for this dataset")

# --------------------------------------------------
# LSTM FORECAST
# --------------------------------------------------
st.subheader("🤖 Deep Learning Forecast (LSTM)")

try:
    data = monthly[[metric]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model_lstm.compile(optimizer="adam", loss="mse")
    model_lstm.fit(X, y, epochs=10, verbose=0)

    pred = scaler.inverse_transform(model_lstm.predict(X))

    fig_lstm = px.line(
        x=monthly["month"][10:],
        y=pred.flatten(),
        title="LSTM Forecast"
    )
    st.plotly_chart(fig_lstm, use_container_width=True)
except:
    st.warning("LSTM needs more numeric time data")

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------
st.subheader("⬇ Download Analyzed Data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "analyzed_data.csv",
    "text/csv"
)
