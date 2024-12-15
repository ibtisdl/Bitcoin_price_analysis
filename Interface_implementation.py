import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle


st.title("Bitcoin Price Prediction")
st.sidebar.header("Data Options")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.info("Using default dataset")
    data = pd.read_csv("data/Bitcoin_Historical_Data.csv")


st.header("Data Overview")
st.write("### Dataset Preview")
st.write(data.head())

st.write("### Basic Statistics")
st.write(data.describe())

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

st.write("### Price Trends")
if 'Date' in data.columns and 'Close' in data.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Bitcoin Close Price Over Time')
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Required columns ('Date', 'Close') are missing in the dataset.")

st.header("Preprocessing Data")
if data.isnull().sum().sum() > 0:
    st.warning("Dataset contains missing values. Filling with median values.")
    data.fillna(data.median(), inplace=True)
st.write("No missing values remaining.")

features = ['Open', 'High', 'Low', 'Volume']
if all(f in data.columns for f in features) and 'Close' in data.columns:
    X = data[features]
    y = data['Close']
else:
    st.error("Required columns for prediction are missing.")
    st.stop()


st.header("Model Training")
model = None

if st.button("Train a new model"):
    #st.write("### Splitting Data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #st.write("### Training Linear Regression Model")
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("### Model Performance")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error: {mae:.2f}")


    with open('models/bitcoin_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    st.success("Model trained and saved as 'bitcoin_model.pkl'!")
else:
    try:
        with open('models/bitcoin_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Loaded existing model.")
    except FileNotFoundError:
        st.error("No pre-trained model found. Please train a model first.")
        st.stop()


st.header("Predict Bitcoin Prices")

if model:
    st.write("### Enter Features for Prediction")
    open_price = st.number_input("Open Price", value=40000.0)
    high_price = st.number_input("High Price", value=40500.0)
    low_price = st.number_input("Low Price", value=39500.0)
    volume = st.number_input("Volume", value=100000000.0)


    input_data = np.array([[open_price, high_price, low_price, volume]])


    prediction = model.predict(input_data)
    st.write(f"### Predicted Close Price: ${prediction[0]:.2f}")
