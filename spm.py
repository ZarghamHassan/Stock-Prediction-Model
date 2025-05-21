import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

st.title("📈 Stock Price Forecasting Dashboard")

# --- Sidebar for user input ---
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. TCS.NS)", "TCS.NS")
years = st.sidebar.slider("Years of historical data", 1, 10, 5)
today = dt.date.today()
start_date = today - dt.timedelta(days=years*365)

# --- Load Data ---
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

df = load_data(ticker, start_date, today)
if df.empty:
    st.error("No data found for this ticker.")
    st.stop()

st.write("Data Preview:")
st.dataframe(df.head())

# --- Prepare Data ---
df_model = df[["Date", "Close"]].copy()
df_model.set_index("Date", inplace=True)
train_size = int(len(df_model) * 0.8)
train = df_model.iloc[:train_size]
test = df_model.iloc[train_size:]

# --- ARIMA ---
try:
    arima_model = ARIMA(train, order=(5, 1, 0))
    arima_result = arima_model.fit()
    arima_pred = arima_result.predict(start=len(train), end=len(train)+len(test)-1, typ="levels")
    arima_rmse = np.sqrt(mean_squared_error(test["Close"], arima_pred))
    arima_mae = mean_absolute_error(test["Close"], arima_pred)
    arima_mape = np.mean(np.abs((test["Close"] - arima_pred) / test["Close"])) * 100
except Exception as e:
    arima_pred = np.zeros(len(test))
    arima_rmse = np.nan
    st.warning(f"ARIMA error: {e}")

# ✅ SARIMA
sarima_model= SARIMAX(train['Close'],order=(0,1,0),seasonal_order=(2, 1, 0, 12))
sarima_fit= sarima_model.fit()
sarima_pred = sarima_fit.predict(start=len(train), end=len(train)+len(test)-1, typ="levels")
sarima_rmse = np.sqrt(mean_squared_error(test["Close"], sarima_pred))
sarima_mae = mean_absolute_error(test["Close"], sarima_pred)
sarima_mape = np.mean(np.abs((test["Close"] - sarima_pred) / test["Close"])) * 100


# --- Prophet ---
df2 = df[['Date', 'Close']]
df2.columns = ['ds', 'y']
df2 = df2.sort_values('ds')
df2 = df2.set_index('ds').asfreq('B')  # Use business days to match ARIMA/SARIMA/LSTM
df2['y'] = df2['y'].interpolate()
df2 = df2.reset_index()
train_size_prophet = int(len(df2) * 0.8)
train_prophet = df2[:train_size_prophet]
test_prophet = df2[train_size_prophet:]
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(train_prophet)
future = model_prophet.make_future_dataframe(periods=len(test_prophet), freq='B')
prophet_pred = model_prophet.predict(future)
# Align Prophet predictions to test index and drop NaNs
prophet_pred_df = prophet_pred.set_index('ds').reindex(test.index)
mask = ~prophet_pred_df['yhat'].isna()
prophet_pred_business = prophet_pred_df.loc[mask, 'yhat'].values
actual_prophet = test["Close"].loc[mask].values
prophet_dates = test.index[mask]  # Only plot where both exist
prophet_rmse = np.sqrt(mean_squared_error(actual_prophet, prophet_pred_business))
prophet_mae = mean_absolute_error(actual_prophet, prophet_pred_business)
prophet_mape = np.mean(np.abs((actual_prophet - prophet_pred_business) / actual_prophet)) * 100

# --- LSTM ---
try:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_model)
    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model_lstm.compile(loss="mean_squared_error", optimizer="adam")
    model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    lstm_pred_scaled = model_lstm.predict(X_test)
    lstm_pred = scaler.inverse_transform(
        np.concatenate([lstm_pred_scaled, np.zeros((len(lstm_pred_scaled), scaled_data.shape[1]-1))], axis=1)
    )[:, 0]
    y_test_unscaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1]-1))], axis=1)
    )[:, 0]
    lstm_rmse = np.sqrt(mean_squared_error(y_test_unscaled, lstm_pred))
    lstm_mae = mean_absolute_error(y_test_unscaled, lstm_pred)
    lstm_mape = np.mean(np.abs((y_test_unscaled - lstm_pred) / y_test_unscaled)) * 100

except Exception as e:
    lstm_pred = np.zeros(len(test))
    lstm_rmse = np.nan
    st.warning(f"LSTM error: {e}")

# --- Metrics ---
st.subheader("📊 Model Performance Metrics")
metrics_df = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    "RMSE": [arima_rmse, sarima_rmse, prophet_rmse, lstm_rmse],
    "MAE": [arima_mae, sarima_mae, prophet_mae, lstm_mae],
    "MAPE": [arima_mape, sarima_mape, prophet_mape, lstm_mape]
})
st.dataframe(metrics_df)

# --- Plot ---
st.subheader("📈 Forecast vs Actual Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test["Close"].values, label="Actual", color='black')
ax.plot(test.index, arima_pred, label="ARIMA")
ax.plot(test.index, sarima_pred, label="SARIMA")
ax.plot(prophet_dates, prophet_pred_business, label="Prophet") 
ax.plot(test.index[-len(lstm_pred):], lstm_pred, label="LSTM")
ax.set_title("Actual vs Predicted Close Prices")
ax.legend()
st.pyplot(fig)

# --- Forecast Future ---
st.subheader("Forecast Future Prices")
days = st.slider("Days to forecast", 1, 30, 7)
model_choice = st.selectbox("Choose model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])

if st.button("Predict Future"):
    if model_choice == "ARIMA":
        try:
            future_pred = arima_result.predict(start=len(df_model), end=len(df_model)+days-1, typ="levels")
            st.line_chart(future_pred)
        except Exception as e:
            st.error(f"ARIMA prediction error: {e}")
    elif model_choice == "SARIMA":
        try:
            future_pred = sarima_fit.predict(start=len(df_model), end=len(df_model)+days-1, typ="levels")
            st.line_chart(future_pred)
        except Exception as e:
            st.error(f"SARIMA prediction error: {e}")
    elif model_choice == "Prophet":
        try:
            future_df = model_prophet.make_future_dataframe(periods=days, freq='B')
            future_forecast = model_prophet.predict(future_df)
            future_series = future_forecast.set_index("ds")["yhat"][-days:]
            st.line_chart(future_series)
        except Exception as e:
            st.error(f"Prophet prediction error: {e}")
    elif model_choice == "LSTM":
        try:
            last_seq = scaled_data[-seq_len:]
            input_seq = last_seq.reshape(1, seq_len, 1)
            future = []
            for _ in range(days):
                pred = model_lstm.predict(input_seq)[0][0]
                future.append(pred)
                input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
            future_unscaled = scaler.inverse_transform(
                np.concatenate([np.array(future).reshape(-1, 1), np.zeros((days, scaled_data.shape[1]-1))], axis=1)
            )[:, 0]
            st.line_chart(pd.Series(future_unscaled))
        except Exception as e:
            st.error(f"LSTM prediction error: {e}")