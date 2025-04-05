# Sayali Gurav
# Implements Isolation Forest, LSTM, and Autoencoder for fraud detection.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import joblib  # For saving models

#  Load dataset (Temporary until Phase 1 preprocessing is complete)
def load_data(file_path="data/creditcard.csv", sample_size=10000):
    df = pd.read_csv(file_path)
    df = df.sample(sample_size)  # Work with a smaller subset for now
    return df

#  Basic Preprocessing Placeholder (Temporary Scaling)
def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, :-1])  # Exclude labels
    labels = df["Class"]  # 0 = Normal, 1 = Fraud
    return df_scaled, labels, scaler

#  Isolation Forest Implementation
def train_isolation_forest(X_train):
    model = IsolationForest(n_estimators=100, contamination=0.002)  # Adjust contamination later
    model.fit(X_train)
    return model

#  LSTM Autoencoder Model Definition
def create_lstm_autoencoder(input_dim, timesteps):
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)
    
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

#  Autoencoder Model Definition
def create_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(inputs)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

#  Train & Save Models
def train_and_save_models(X_train):
    print("Training Isolation Forest...")
    iso_forest = train_isolation_forest(X_train)
    joblib.dump(iso_forest, "models/isolation_forest.pkl")

    print("Training Autoencoder...")
    autoencoder = create_autoencoder(X_train.shape[1])
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, shuffle=True, validation_split=0.1)
    autoencoder.save("models/autoencoder.h5")

    print("Training LSTM Autoencoder...")
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM
    lstm_autoencoder = create_lstm_autoencoder(X_train.shape[1], 1)
    lstm_autoencoder.fit(X_train_lstm, X_train_lstm, epochs=20, batch_size=256, shuffle=True, validation_split=0.1)
    lstm_autoencoder.save("models/lstm_autoencoder.h5")

    print("Models Trained & Saved!")

#  Evaluate Models (Dummy Evaluation for Now)
def evaluate_models(X_test):
    print("Loading models...")
    iso_forest = joblib.load("models/isolation_forest.pkl")
    autoencoder = keras.models.load_model("models/autoencoder.h5")
    lstm_autoencoder = keras.models.load_model("models/lstm_autoencoder.h5")

    # Predictions
    iso_pred = iso_forest.predict(X_test)
    autoencoder_pred = autoencoder.predict(X_test)
    lstm_pred = lstm_autoencoder.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1])))

    print(" Model Evaluation - To Be Completed in Phase 4")
    return iso_pred, autoencoder_pred, lstm_pred

#  Main Execution
if __name__ == "__main__":
    df = load_data()
    X_train, labels, scaler = preprocess_data(df)
    train_and_save_models(X_train)
    evaluate_models(X_train)
