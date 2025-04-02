import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# 📌 Configuration de l'API CoinGecko
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"

# 📥 Fonction pour récupérer les données historiques du prix TAO
def get_tao_history(days=365):
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(COINGECKO_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        st.error(f"Erreur {response.status_code} : Impossible de récupérer les données.")
        return None

# 📊 Fonction pour ajouter des indicateurs techniques
def add_technical_indicators(df):
    df["SMA_14"] = ta.trend.sma_indicator(df["price"], window=14)
    df["EMA_14"] = ta.trend.ema_indicator(df["price"], window=14)
    df["RSI"] = ta.momentum.rsi(df["price"], window=14)
    df["MACD"] = ta.trend.macd(df["price"])
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["price"])
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["price"])
    
    # 🚀 Simulation des prix "high" et "low" pour l'ATR
    df["high"] = df["price"] * (1 + np.random.uniform(0.001, 0.01, len(df)))
    df["low"] = df["price"] * (1 - np.random.uniform(0.001, 0.01, len(df)))
    
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["price"], window=14)
    
    df.fillna(df.mean(), inplace=True)  # Remplacement des NaN
    return df

# 📌 Préparation des données pour le modèle LSTM
def prepare_data(df, window_size=14):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[["price", "SMA_14", "EMA_14", "RSI", "MACD", "ATR"]].values)

    X, y = [], []
    for i in range(len(df_scaled) - window_size):
        X.append(df_scaled[i : i + window_size])
        y.append(df_scaled[i + window_size, 0])  # Prédiction du prix uniquement

    return np.array(X), np.array(y), scaler

# 🎯 Modèle LSTM amélioré
def train_lstm(X, y):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    # Entraînement avec validation croisée
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)
    return model, scaler

# 📈 Fonction de prédiction des prix futurs
def predict_future_prices(model, df, scaler, days=30):
    last_sequence = df[["price", "SMA_14", "EMA_14", "RSI", "MACD", "ATR"]].values[-14:]
    last_sequence_scaled = scaler.transform(last_sequence)

    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence_scaled.reshape(1, 14, 6))
        future_price = scaler.inverse_transform([[prediction[0][0], 0, 0, 0, 0, 0]])[0][0]
        future_prices.append(future_price)

        last_sequence_scaled = np.roll(last_sequence_scaled, -1, axis=0)
        last_sequence_scaled[-1, 0] = prediction[0][0]

    return future_prices

# 🚀 Interface Streamlit
st.title("📈 TAO Predictor - Prédiction avec indicateurs techniques")

# Bouton pour entraîner le modèle
if st.button("🚀 Entraîner le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        df = add_technical_indicators(df)
        X, y, scaler = prepare_data(df)
        model, scaler = train_lstm(X, y)

        mse = mean_squared_error(y, model.predict(X))
        st.write(f"📊 **Performance du modèle (MSE) : {mse:.4f}**")

        st.write("✅ Modèle entraîné avec succès !")
    else:
        st.error("Erreur : Impossible d'entraîner le modèle.")

# 📊 Bouton pour afficher les prévisions sur 7 jours
if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        df = add_technical_indicators(df)
        X, y, scaler = prepare_data(df)
        model, scaler = train_lstm(X, y)
        future_prices = predict_future_prices(model, df, scaler, days=7)

        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=8, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 7 jours", linestyle="dashed", color="red")

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction du prix TAO sur 7 jours")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Erreur : Impossible d'afficher les prévisions.")

# 📊 Bouton pour afficher les prévisions sur 30 jours
if st.button("📊 Afficher les prévisions sur 30 jours"):
    df = get_tao_history()
    if df is not None:
        df = add_technical_indicators(df)
        X, y, scaler = prepare_data(df)
        model, scaler = train_lstm(X, y)
        future_prices = predict_future_prices(model, df, scaler, days=30)

        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["price"], label="Prix réel", color="blue")
        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=31, freq="D")[1:]
        plt.plot(future_dates, future_prices, label="Prédictions 30 jours", linestyle="dashed", color="green")

        plt.xlabel("Date")
        plt.ylabel("Prix en USD")
        plt.title("📈 Prédiction du prix TAO sur 30 jours")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Erreur : Impossible d'afficher les prévisions.")
