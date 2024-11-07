import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")

class ModelBuilding:
    def __init__(self, df):
        df = df.rename(
            columns={"close": "Close", "volume": "Volume", "sentiment": "sentiments"}
        )
        self.df = df.copy()
        self.prepare_data()
        self.add_technical_indicators()  # Add technical indicators

    def prepare_data(self):
        # Ensure 'Date' column exists and is in datetime format
        if 'Date' not in self.df.columns:
            self.df=self.df.reset_index()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date")
        self.df.set_index("Date", inplace=True)

    def add_technical_indicators(self):
        # Calculate and add the required technical indicators
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        self.df['BB_upper'] = self.df['Close'].rolling(window=20).mean() + 2 * self.df['Close'].rolling(window=20).std()
        self.df['BB_middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_lower'] = self.df['Close'].rolling(window=20).mean() - 2 * self.df['Close'].rolling(window=20).std()

        # Average True Range (ATR)
        high_low = self.df['Close'] - self.df['Close'].shift(1)
        self.df['ATR'] = high_low.rolling(window=14).mean()

        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

        # Stochastic Oscillator
        self.df['Stochastic_K'] = 100 * ((self.df['Close'] - self.df['Close'].rolling(window=14).min()) /
                                         (self.df['Close'].rolling(window=14).max() - self.df['Close'].rolling(window=14).min()))

    def train_sarimax(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        # Select exogenous variables (technical indicators)
        exog = self.df[["SMA_20", "SMA_50", "EMA_20", "RSI", "MACD", "ATR", "Stochastic_K", "OBV"]].fillna(0)
        sarimax_model = SARIMAX(
            self.df["Close"], order=order, seasonal_order=seasonal_order, exog=exog
        )
        self.sarimax_model_fit = sarimax_model.fit(disp=False)
        print(self.sarimax_model_fit.summary())

    def predict_sarimax(self, steps=7):
        exog_forecast = (
            self.df[["SMA_20", "SMA_50", "EMA_20", "RSI", "MACD", "ATR", "Stochastic_K", "OBV"]]
            .fillna(0)
            .iloc[-steps:]
        )
        forecast = self.sarimax_model_fit.get_forecast(steps=steps, exog=exog_forecast)
        forecast_values = forecast.predicted_mean
        forecast_index = pd.date_range(
            self.df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"
        )
        return pd.Series(forecast_values, index=forecast_index)

    def prepare_lstm_data(self, feature_columns, target_column, look_back=60):
        data = self.df[feature_columns].fillna(0)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back : i])
            y.append(scaled_data[i, data.columns.get_loc(target_column)])

        X, y = np.array(X), np.array(y)
        return X, y, scaler

    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        # Define the LSTM model
        self.lstm_model = Sequential()
        self.lstm_model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
            )
        )
        self.lstm_model.add(LSTM(units=50, return_sequences=False))
        self.lstm_model.add(Dense(units=25))
        self.lstm_model.add(Dense(units=1))

        # Compile and fit the LSTM model
        self.lstm_model.compile(optimizer="adam", loss="mean_squared_error")
        self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_lstm(self, X_last_sequence, scaler, steps=7):
        # Predict future steps with LSTM
        predictions = []
        current_input = X_last_sequence.copy()

        for _ in range(steps):
            pred = self.lstm_model.predict(current_input)[0, 0]
            predictions.append(pred)
            # Update the input with the predicted value to continue forecasting
            current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

        # Inverse scale to get actual values
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        forecast_index = pd.date_range(
            self.df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"
        )
        return pd.Series(predictions, index=forecast_index)

# Example usage:
stock = 'AAPL'
df = pd.read_csv(f"/Users/moatasimfarooque/Desktop/CATIA/Latest/processed_stocks/{stock}_processed.csv")

model_building = ModelBuilding(df)
model_building.prepare_data()

actual_values = model_building.df["Close"].iloc[-7:]
# SARIMAX Model
model_building.train_sarimax(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
sarimax_forecast = model_building.predict_sarimax(steps=7)
print("SARIMAX Forecast:\n", sarimax_forecast)
sarimax_comparison_df = pd.DataFrame(
    {"Actual": actual_values.values, "SARIMAX Forecast": sarimax_forecast.values},
    index=sarimax_forecast.index,
)
print("SARIMAX Forecast vs Actual:\n", sarimax_comparison_df)
# Prepare data for LSTM
feature_columns = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'Stochastic_K', 'OBV']
X, y, scaler = model_building.prepare_lstm_data(feature_columns, target_column='Close', look_back=60)

# Train LSTM
model_building.train_lstm(X, y, epochs=50, batch_size=32)

# LSTM Forecast
X_last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
lstm_forecast = model_building.predict_lstm(X_last_sequence, scaler, steps=7)
lstm_comparison_df = pd.DataFrame(
    {"Actual": actual_values.values, "LSTM Forecast": lstm_forecast.values},
    index=lstm_forecast.index,
)

print("LSTM Forecast:\n", lstm_comparison_df)
