import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class FinancialDataAnalysis:
    def __init__(self, df, benchmark_df=None):
        self.df = df.copy()
        self.benchmark_df = benchmark_df
        self.prepare_data()

    def prepare_data(self):
        # Convert Date to datetime and set it as the index for the stock data
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date")
        self.df.index=self.df["Date"]

        # Fill missing sentiment values if any
        if "sentiments" in self.df.columns:
            self.df["sentiments"] = self.df["sentiments"].fillna(0)

        # Add rolling averages for sentiment
        self.df["Sentiment_7D_Rolling"] = self.df["sentiments"].rolling(window=7).mean()
        self.df["Sentiment_14D_Rolling"] = (
            self.df["sentiments"].rolling(window=14).mean()
        )

        # Add lagged rolling averages for sentiment
        self.df["Sentiment_7D_Lagged"] = self.df["Sentiment_7D_Rolling"].shift(7)
        self.df["Sentiment_14D_Lagged"] = self.df["Sentiment_14D_Rolling"].shift(14)

    def add_technical_indicators(self):
        # Moving averages
        self.df["SMA_20"] = self.df["Close"].rolling(window=20).mean()
        self.df["SMA_50"] = self.df["Close"].rolling(window=50).mean()

        # Exponential Moving Average (EMA)
        self.df["EMA_20"] = self.df["Close"].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        self.df["BB_middle"] = self.df["Close"].rolling(window=20).mean()
        self.df["BB_upper"] = (
            self.df["BB_middle"] + 2 * self.df["Close"].rolling(window=20).std()
        )
        self.df["BB_lower"] = (
            self.df["BB_middle"] - 2 * self.df["Close"].rolling(window=20).std()
        )

        # RSI (Relative Strength Index)
        delta = self.df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = self.df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = self.df["Close"].ewm(span=26, adjust=False).mean()
        self.df["MACD"] = exp1 - exp2
        self.df["Signal_Line"] = self.df["MACD"].ewm(span=9, adjust=False).mean()

        # Volatility
        self.df["Daily_Return"] = self.df["Close"].pct_change()
        self.df["Volatility"] = self.df["Daily_Return"].rolling(
            window=20
        ).std() * np.sqrt(252)

        # Volume indicators
        self.df["Volume_MA"] = self.df["Volume"].rolling(window=20).mean()
        self.df["Volume_Ratio"] = self.df["Volume"] / self.df["Volume_MA"]

        return self.df

    def plot_sentiment_analysis(self):
        fig, ax1 = plt.subplots(figsize=(14, 5))

        # Plotting stock price
        ax1.plot(self.df.index, self.df["Close"], label="Close Price", color="blue")
        ax1.set_ylabel("Close Price", color="blue")

        # Adding a secondary y-axis for sentiment rolling averages
        ax2 = ax1.twinx()
        ax2.plot(
            self.df.index,
            self.df["Sentiment_7D_Rolling"],
            label="7-Day Rolling Sentiment",
            color="orange",
            alpha=0.6,
        )
        ax2.plot(
            self.df.index,
            self.df["Sentiment_14D_Rolling"],
            label="14-Day Rolling Sentiment",
            color="red",
            alpha=0.6,
        )
        ax2.set_ylabel("Sentiment (Rolling Avg)", color="orange")

        ax2.legend(loc="upper left")
        plt.title("Rolling Sentiment vs. Close Price")
        fig.tight_layout()
        plt.show()

    def plot_technical_indicators(self):
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plotting price with SMA and EMA
        ax.plot(self.df.index, self.df["Close"], label="Close Price", color="blue")
        ax.plot(self.df.index, self.df["SMA_20"], label="SMA 20", color="orange")
        ax.plot(self.df.index, self.df["SMA_50"], label="SMA 50", color="green")
        ax.plot(self.df.index, self.df["EMA_20"], label="EMA 20", color="red")

        plt.title("Technical Indicators (SMA, EMA) on Close Price")
        plt.legend()
        plt.show()

    def plot_bollinger_bands(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df["Close"], label="Close Price", color="blue")
        plt.plot(
            self.df.index,
            self.df["BB_upper"],
            label="Bollinger Upper Band",
            linestyle="--",
            color="orange",
        )
        plt.plot(
            self.df.index,
            self.df["BB_middle"],
            label="Bollinger Middle Band",
            color="grey",
        )
        plt.plot(
            self.df.index,
            self.df["BB_lower"],
            label="Bollinger Lower Band",
            linestyle="--",
            color="orange",
        )
        plt.title("Bollinger Bands on Close Price")
        plt.legend()
        plt.show()

    def plot_rsi(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["RSI"], label="RSI", color="purple")
        plt.axhline(70, color="red", linestyle="--", label="Overbought")
        plt.axhline(30, color="green", linestyle="--", label="Oversold")
        plt.title("Relative Strength Index (RSI)")
        plt.legend()
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.df[["Close", "Volume", "SMA_20", "SMA_50", "sentiments"]].corr(),
            annot=True,
            cmap="coolwarm",
        )
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_volatility(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.df["Volatility"], label="Volatility", color="purple")
        plt.axhline(
            self.df["Volatility"].mean(),
            color="red",
            linestyle="--",
            label="Average Volatility",
        )
        plt.title("Volatility Over Time")
        plt.legend()
        plt.show()

    def plot_return_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df["Daily_Return"].dropna(), kde=True)
        plt.title("Daily Returns Distribution")
        plt.show()

    def compare_to_benchmark(self):
        if "Benchmark_Close" not in self.df.columns:
            print("Benchmark data not available for comparison.")
            return

        # Calculate daily returns
        self.df["Stock_Return"] = self.df["Close"].pct_change()
        self.df["Benchmark_Return"] = self.df["Benchmark_Close"].pct_change()

        # Calculate cumulative returns
        self.df["Stock_Cumulative_Return"] = (1 + self.df["Stock_Return"]).cumprod() - 1
        self.df["Benchmark_Cumulative_Return"] = (
            1 + self.df["Benchmark_Return"]
        ).cumprod() - 1

        # Plot cumulative returns
        plt.figure(figsize=(14, 7))
        plt.plot(
            self.df.index,
            self.df["Stock_Cumulative_Return"],
            label="Stock Cumulative Return",
            color="blue",
        )
        plt.plot(
            self.df.index,
            self.df["Benchmark_Cumulative_Return"],
            label="Benchmark Cumulative Return",
            color="green",
        )
        plt.title("Stock vs. Benchmark Cumulative Return")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.show()

    def plot_all_EDA(self):
        # Add technical indicators first
        self.add_technical_indicators()

        # Plot all EDA and technical analysis visualizations
        self.plot_sentiment_analysis()
        self.plot_technical_indicators()
        self.plot_bollinger_bands()
        self.plot_rsi()
        self.plot_correlation_heatmap()
        self.plot_volatility()
        self.plot_return_distribution()
        self.compare_to_benchmark()


stock = "AAPL"
df_stock = pd.read_csv(
    f"/Users/moatasimfarooque/Desktop/CATIA/Latest/processed_stocks/{stock}_processed.csv"
).rename(columns={"sentiment": "sentiments"})

# df_benchmark = pd.read_csv('/Users/moatasimfarooque/Desktop/CATIA/stock-tweet-and-price/Stock_Price_Prediction/stockprice/^GSPC.csv')

# Initialize the FinancialDataAnalysis class with stock data and benchmark data
analysis = FinancialDataAnalysis(df_stock, benchmark_df=None)

# Run the analysis and plot all EDA
analysis.plot_all_EDA()
