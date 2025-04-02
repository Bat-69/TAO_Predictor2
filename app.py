import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Librairie pour les indicateurs techniques
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Titre de l'application
st.title("📈 TAO Predictor - Prédiction améliorée")

# API CoinGecko pour récupérer l'historique des prix
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bittensor/market_chart"

def get_tao_history(days=365):
    """Récupère les prix de TAO depuis CoinGecko."""
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

def add_technical_indicators(df):
    """Ajoute des indicateurs techniques au DataFrame."""
    df["SMA_14"] = ta.trend.sma_indicator(df["price"], window=14)
    df["EMA_14"] = ta.trend.ema_indicator(df["price"], window=14)
    df["RSI"] = ta.momentum.rsi(df["price"], window=14)
    df["MACD"] = ta.trend.macd(df["price"])
    
    # ✅ Correction de l'ATR
    df["ATR"] = ta.volatility.average_true_range(high=df["price"], 
                                                 low=df["price"], 
                                                 close=df["price"], 
                                                 window=14)
    
    df.fillna(method="bfill", inplace=True)  # Remplissage des valeurs NaN
    return df
def prepare_data(df, window_size=7):
    """Prépare les données pour l'entraînement du modèle."""
    df = add_technical_indicators(df)  # Ajout des indicateurs avant la normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = ["price", "SMA_14", "EMA_14", "RSI", "MACD", "ATR"]
    df_scaled = scaler.fit_transform(df[features].values)

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df_scaled[i : i + window_size])
        y.append(df_scaled[i + window_size, 0])  # Prédire uniquement le prix
    
    return np.array(X), np.array(y), scaler

def train_lstm(X, y):
    """Entraîne un modèle LSTM."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)
    return model

def predict_future_prices(model, df, scaler, days=30):
    """Prédit les prix futurs."""
    features = ["price", "SMA_14", "EMA_14", "RSI", "MACD", "ATR"]
    last_sequence = scaler.transform(df[features].iloc[-7:].values).reshape(1, 7, -1)
    future_prices = []

    for _ in range(days):
        prediction = model.predict(last_sequence)
        future_price = scaler.inverse_transform(np.hstack((prediction, np.zeros((1, 5)))))[0][0]
        future_prices.append(future_price)
        
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = prediction[0][0]

    return future_prices

def plot_real_vs_predicted(df, model, scaler):
    """Affiche le cours réel vs prédictions sur l'année écoulée."""
    X, y, _ = prepare_data(df)
    X = X.reshape(-1, X.shape[1], X.shape[2])

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 5)))))[:, 0]

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"][-len(predictions):], df["price"][-len(predictions):], label="Prix réel", color="blue")
    plt.plot(df["timestamp"][-len(predictions):], predictions, label="Prix prédit", color="orange", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Prix en USD")
    plt.title("📈 Comparaison Cours Réel vs Prédit")
    plt.legend()
    st.pyplot(plt)

# Entraînement du modèle
if st.button("🚀 Entraîner le modèle LSTM"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)

        # 📊 Affichage des performances
        plot_real_vs_predicted(df, model, scaler)
        st.write("✅ Modèle entraîné avec succès !")
    else:
        st.error("Erreur : Impossible d'entraîner le modèle.")

# Bouton pour afficher les prévisions sur 7 jours
if st.button("📊 Afficher les prévisions sur 7 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)
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

# Bouton pour afficher les prévisions sur 30 jours
if st.button("📊 Afficher les prévisions sur 30 jours"):
    df = get_tao_history()
    if df is not None:
        X, y, scaler = prepare_data(df)
        model = train_lstm(X.reshape(-1, X.shape[1], X.shape[2]), y)
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
