# Bitcoin Price Prediction App

This repository contains a Streamlit-based application for predicting Bitcoin prices using machine learning models. The application allows users to explore data, preprocess it, train a machine learning model, and make predictions.

---

## Features

1. **Data Upload and Exploration**:
   - Upload custom CSV files for analysis.
   - View dataset preview and descriptive statistics.
   - Plot Bitcoin price trends over time.

2. **Preprocessing**:
   - Automatically handle missing values by filling them with the median.
   - Select relevant features for model training.

3. **Model Training**:
   - Train a Linear Regression model with a single click.
   - Evaluate the model's performance with Mean Absolute Error (MAE).
   - Save the trained model for future use.

4. **Prediction**:
   - Use the trained model to predict Bitcoin prices based on user input features (Open Price, High Price, Low Price, Volume).

---

## Installation

### Prerequisites
- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- matplotlib
- seaborn

### Steps
1. Clone the repository:
   git clone https://github.com/ibtisdl/Bitcoin_price_analysis.git
   cd Bitcoin_price_analysis

2. Install dependencies:
 
   pip install streamlit sklearn pandas matplotlib seaborn
  
3. Run the app:
  
   streamlit run Interface_inmplementation.py
  



## Usage

1. **Launch the app**: Open the Streamlit application in your browser.
2. **Upload or explore default data**: Upload your own Bitcoin historical data in CSV format or use the default dataset.
3. **Train a model**: Click the "Train a new model" button to train a Linear Regression model and save it.
4. **Predict prices**: Input values for Open Price, High Price, Low Price, and Volume to predict the Bitcoin closing price.

---

## Files in Repository
- `Bitcoin_analysis` : Visualisation and model building.
- `Interface_implementation.py`: The main application script for the Streamlit app.
- `bitcoin_model.pkl`: The saved model file (generated after training).
- `README.md`: Documentation for the project.

---

## Dataset Format

The dataset should be in CSV format with the following required columns:
- `Open`: Opening price of Bitcoin.
- `High`: Highest price of Bitcoin.
- `Low`: Lowest price of Bitcoin.
- `Close`: Closing price of Bitcoin (target variable).
- `Volume`: Trading volume of Bitcoin.

Optional:
- `Date`: For plotting trends (ensure it is in a datetime-compatible format).

---

## Example Screenshot
![image](https://github.com/user-attachments/assets/d262a615-9885-4657-b1c2-a25f05de0a6c)
![image](https://github.com/user-attachments/assets/95494e2f-47ea-4325-a53b-12f8a1002684)



## Future Improvements

- Add support for additional machine learning models.
- Integrate real-time Bitcoin data from APIs.
- Enhance the UI with interactive visualizations.


## Acknowledgments

- Thanks to the contributors of open-source libraries like Streamlit, scikit-learn, pandas, and matplotlib.
- Inspired by the growing interest in cryptocurrency and its predictive analytics.

