# Import necessary libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, to_date, col, mean as F_mean, last
import pandas as pd
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.appName("StockPriceAndTweetProcessor").getOrCreate()

# Define paths for stock price and tweet data
stockprice_path = "/Users/moatasimfarooque/Desktop/CATIA/Latest/stockprice"  # Replace with actual path
stocktweet_path = "/Users/moatasimfarooque/Desktop/CATIA/Latest/stocktweet/stocktweet.csv"  # Replace with actual path
output_path = "/Users/moatasimfarooque/Desktop/CATIA/Latest/processed_stocks/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# List of desired stock Tickers
Tickers = ["TSLA", "AAPL", "BA", "DIS", "AMZN"]

# Step 1: Load and process stock price data
stock_dfs = []
for Ticker in Tickers:
    filepath = os.path.join(stockprice_path, f"{Ticker}.csv")
    if os.path.exists(filepath):
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        df = df.withColumn("Ticker", lit(Ticker))
        stock_dfs.append(df)

# Combine all stock data into a single DataFrame
if stock_dfs:
    stockprice_df = stock_dfs[0]
    for df in stock_dfs[1:]:
        stockprice_df = stockprice_df.union(df)
else:
    raise ValueError("No stock price data found for selected Tickers.")

# Standardize Date format and select columns
stockprice_df = stockprice_df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd"))
stockprice_df = stockprice_df.select("Date", "Close", "Volume", "Ticker")

# Fill missing stock prices
window_spec = (
    Window.partitionBy("Ticker")
    .orderBy("Date")
    .rowsBetween(Window.unboundedPreceding, 0)
)
stockprice_df = stockprice_df.withColumn(
    "Close", last("Close", ignorenulls=True).over(window_spec)
)
stockprice_df = stockprice_df.withColumn(
    "Volume", last("Volume", ignorenulls=True).over(window_spec)
)
median_price = stockprice_df.approxQuantile("Close", [0.5], 0.01)[0]  # Median estimate
stockprice_df = stockprice_df.na.fill({"Close": median_price, "Volume": 0})

# Step 2: Create a continuous daily date range
min_date = stockprice_df.agg(F.min("Date")).collect()[0][0]
max_date = stockprice_df.agg(F.max("Date")).collect()[0][0]
date_range_df = spark.sql(
    f"SELECT explode(sequence(to_date('{min_date}'), to_date('{max_date}'), interval 1 day)) as Date"
)

# Step 3: Ensure each Ticker has daily values by joining with date range
Tickers_df = spark.createDataFrame([(Ticker,) for Ticker in Tickers], ["Ticker"])
full_date_Ticker_df = date_range_df.crossJoin(Tickers_df)

# Aggregate stock price data to get daily averages
daily_aggregated_df = stockprice_df.groupBy("Date", "Ticker").agg(
    F_mean("Close").alias("Avg_Close"), F_mean("Volume").alias("Avg_Volume")
)

# Join full date range with aggregated data
final_daily_df = full_date_Ticker_df.join(
    daily_aggregated_df, on=["Date", "Ticker"], how="left"
)

# Fill remaining missing values with last known values
final_daily_df = final_daily_df.withColumn(
    "Avg_Close",
    last("Avg_Close", ignorenulls=True).over(
        Window.partitionBy("Ticker")
        .orderBy("Date")
        .rowsBetween(Window.unboundedPreceding, 0)
    ),
)
final_daily_df = final_daily_df.withColumn(
    "Avg_Volume",
    last("Avg_Volume", ignorenulls=True).over(
        Window.partitionBy("Ticker")
        .orderBy("Date")
        .rowsBetween(Window.unboundedPreceding, 0)
    ),
)
final_daily_df = final_daily_df.toPandas()
final_daily_df["Date"] = pd.to_datetime(final_daily_df["Date"], dayfirst=True)
# Step 4: Load and process tweet data
stocktweet = spark.read.csv(stocktweet_path, header=True, inferSchema=True)
stocktweet = stocktweet.filter(stocktweet.stock_name.isin(Tickers))
# stocktweet_pd = spark.createDataFrame(stocktweet)
stocktweet_pd = stocktweet.toPandas()
# Sentiment analysis setup
sia = SentimentIntensityAnalyzer()


def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()) if isinstance(text, str) else ""


stocktweet_pd["cleaned_tweet"] = stocktweet_pd["tweet"].apply(clean_text)
stocktweet_pd["date"] = pd.to_datetime(stocktweet_pd["date"], dayfirst=True)
stocktweet_pd["sentiment"] = stocktweet_pd["cleaned_tweet"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# Prepare tweet data for merging
stocktweet_pd = stocktweet_pd[["date", "stock_name", "sentiment"]]
stocktweet_df = stocktweet_pd.rename(
    columns={"date": "Date", "stock_name": "Ticker"})
merged_df = pd.merge(stocktweet_df, final_daily_df, on=["Date", "Ticker"], how="left")
merged_df=spark.createDataFrame(merged_df)
# Fill missing sentiment values
merged_df = merged_df.fillna({"sentiment": 0})  # Neutral sentiment if missing

final_aggregated_df = merged_df.groupBy("Date", "Ticker").agg(
    F_mean("Avg_Close").alias("Close"),
    F_mean("Avg_Volume").alias("Volume"),
    F_mean("sentiment").alias("sentiment"),
)

# Step 7: Save individual processed CSVs for each stock
for Ticker in Tickers:
    Ticker_df = final_aggregated_df.filter(
        final_aggregated_df.Ticker == Ticker
    ).toPandas()
    output_filepath = os.path.join(output_path, f"{Ticker}_processed.csv")
    Ticker_df.to_csv(output_filepath, index=False)
print(
    "Processed stock data with single row per date saved in 'processed_stocks' folder."
)
