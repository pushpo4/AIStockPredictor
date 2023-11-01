import pandas as pd
from data_collection import DataCollector
from machine_learning import StockPricePredictor
from technical_indicators import calculate_indicators
from sentiment_analysis import SentimentAnalyzer
from web_interface import start_web_interface

# Initialize data collector, machine learning predictor, and sentiment analyzer
data_collector = DataCollector()
predictor = StockPricePredictor()
sentiment_analyzer = SentimentAnalyzer()

# Collect historical stock data
data = data_collector.collect_data("AAPL", start_date="2022-01-01", end_date="2023-01-01")

# Calculate technical indicators
data = calculate_indicators(data)

# Analyze sentiment
sentiment = sentiment_analyzer.analyze_sentiment("AAPL")

# Train machine learning model and make predictions
predictions = predictor.train_and_predict(data)

# Start the web interface
start_web_interface(predictions, sentiment)
