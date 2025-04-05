# Sayali Gurav
#!/usr/bin/env python
"""
Fraud Detection with Hyperparameter Tuning for Isolation Forest, Autoencoder,
and LSTM Autoencoder.
This script loads the Credit Card Fraud Detection dataset, scales the data,
performs hyperparameter tuning (optimizing for precision) using RandomizedSearchCV
and KerasTuner (with Bayesian Optimization), and saves performance plots and the best models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, precision_recall_curve, f1_score, make_scorer
import joblib
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
def load_and_preprocess_data(file_path='creditcard.csv'):
    # Load dataset (assumed to be in the same directory)
    data = pd.read_csv(file_path)
    
    # Separate features and target; drop "Time" if it exists
    X = data.drop(['Class'], axis=1)
    if 'Time' in X.columns:
        X = X.drop(['Time'], axis=1)
    y = data['Class']
    
    # Split into training and test sets (stratify to preserve fraud ratio)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # In the training set, separate normal and fraudulent transactions
    train_normals = X_train_full[y_train_full == 0]
    train_anomalies = X_train_full[y_train_full == 1]
    
    # Split normal training data further into train and validation sets
    train_normals, val_normals = train_test_split(train_normals, test_size=0.1, random_state=42)
    # Use all anomalies in training set for validation
    val_anomalies = train_anomalies

    # Scale features: fit scaler only on normal training data to avoid leakage
    scaler = StandardScaler()
    scaler.fit(train_normals)
    X_train_full_scaled = scaler.transform(X_train_full)
    train_normals_scaled = scaler.transform(train_normals)
    val_normals_scaled   = scaler.transform(val_normals)
    val_anom_scaled      = scaler.transform(val_anomalies)
    X_test_scaled        = scaler.transform(X_test)
    
    print("Training normals:", train_normals.shape, "Training anomalies:", train_anomalies.shape)
    print("Validation normals:", val_normals.shape, "Validation anomalies:", val_anomalies.shape)
    print("Test set shape:", X_test.shape, "Frauds in test:", (y_test == 1).sum())
    
    return (X_train_full_scaled, train_normals, val_normals, val_anomalies, 
            X_test_scaled, y_train_full.values, y_test, scaler)

# ------------------------------
# Isolation Forest Tuning
# ------------------------------
def tune_isolation_forest(X_train_norm, val_norm, val_anom):
    # Combine training normals and validation sets for tuning
    X_tune = pd.concat([X_train_norm, val_norm, val_anom])
    y_tune = np.concatenate([np.zeros(len(X_train_norm)), 
                             np.zeros(len(val_norm)), 
                             np.ones(len(val_anom))])
    
    # Create a predefined split: indices in validation get 0, training indices get -1.
    val_idx = set(list(val_norm.index) + list(val_anom.index))
    test_fold = np.array([0 if idx in val_idx else -1 for idx in X_tune.index])
    ps = PredefinedSplit(test_fold)
    
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 500, 1000],
        'max_samples': [0.6, 0.8, 1.0],
        'contamination': [0.001, 0.005, 0.01, 0.02, 0.05]
    }
    
    # Custom precision scorer (fraud is labeled as 1)
    precision_scorer = make_scorer(precision_score, pos_label=1, average='binary', zero_division=0)
    
    iso_model = IsolationForest(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=iso_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring=precision_scorer,
        cv=ps,
        refit=True,
        verbose=1,
        n_jobs=-1
    )
    
    search.fit(X_tune, y_tune)
    
    best_params = search.best_params_
    best_precision = search.best_score_
    best_model = search.best_estimator_
    
    print("Best Isolation Forest parameters:", best_params)
    print(f"Best CV Precision: {best_precision:.4f}")
    
    # Save the model
    joblib.dump(best_model, "isolation_forest.pkl")
    return best_model

# ------------------------------
# Autoencoder Tuning (Dense AE)
# ------------------------------
def build_autoencoder(hp, input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    if hp.Boolean('two_hidden', default=True):
        units1 = hp.Int('units1', min_value=16, max_value=64, step=16)
        model.add(keras.layers.Dense(units1, activation='relu'))
        dropout_rate = hp.Choice('dropout_rate', values=[0.0, 0.1, 0.2, 0.3])
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
        units2 = hp.Int('units2', min_value=4, max_value=32, step=4)
        model.add(keras.layers.Dense(units2, activation='relu'))
        model.add(keras.layers.Dense(units1, activation='relu'))
    else:
        units = hp.Int('units', min_value=8, max_value=64, step=8)
        model.add(keras.layers.Dense(units, activation='relu'))
        dropout_rate = hp.Choice('dropout_rate_single', values=[0.0, 0.1, 0.2, 0.3])
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    
    model.add(keras.layers.Dense(input_dim, activation='linear'))
    learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

def tune_autoencoder(X_train, X_val):
    input_dim = X_train.shape[1]
    tuner = kt.BayesianOptimization(
        lambda hp: build_autoencoder(hp, input_dim),
        objective='val_loss',
        max_trials=20,
        seed=42,
        directory='tuner_logs',
        project_name='autoencoder_tuning',
        overwrite=True
    )
    
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    tuner.search(X_train, X_train,
                 validation_data=(X_val, X_val),
                 epochs=50, batch_size=1024,
                 callbacks=[stop_early], verbose=1)
    
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Autoencoder hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"  {param}: {value}")
    
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        X_train, X_train,
        epochs=100, batch_size=1024,
        validation_data=(X_val, X_val),
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0)
    
    best_model.save("best_autoencoder_model.h5")
    
    # Plot training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('autoencoder_loss_curve.png')
    plt.close()
    
    # Determine anomaly threshold based on reconstruction error
    recon_val = best_model.predict(X_val)
    errors = np.mean(np.square(X_val - recon_val), axis=1)
    # Use precision-recall to determine a good threshold
    # Here we assume that the validation set includes only normal data
    # For a more refined threshold, you might combine with known anomalies.
    # For this example, we simply use the 95th percentile of errors.
    thresh = np.percentile(errors, 95)
    print(f"Chosen reconstruction error threshold for autoencoder: {thresh:.6f}")
    return best_model, thresh

# ------------------------------
# LSTM Autoencoder Tuning
# ------------------------------
def build_lstm_autoencoder(hp, seq_length, n_features):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(seq_length, n_features)))
    drop_rate = hp.Choice('dropout_rate', values=[0.0, 0.1, 0.2, 0.3])
    
    if hp.Boolean('two_lstm_layers', default=False):
        units1 = hp.Int('lstm_units1', min_value=16, max_value=64, step=16)
        units2 = hp.Int('lstm_units2', min_value=4, max_value=32, step=4)
        model.add(keras.layers.LSTM(units1, activation='tanh', return_sequences=True,
                                    dropout=drop_rate, recurrent_dropout=drop_rate))
        model.add(keras.layers.LSTM(units2, activation='tanh', return_sequences=False,
                                    dropout=drop_rate, recurrent_dropout=drop_rate))
        model.add(keras.layers.RepeatVector(seq_length))
        model.add(keras.layers.LSTM(units2, activation='tanh', return_sequences=True,
                                    dropout=drop_rate, recurrent_dropout=drop_rate))
        model.add(keras.layers.LSTM(units1, activation='tanh', return_sequences=True,
                                    dropout=drop_rate, recurrent_dropout=drop_rate))
    else:
        units = hp.Int('lstm_units', min_value=8, max_value=64, step=8)
        model.add(keras.layers.LSTM(units, activation='tanh', return_sequences=False,
                                    dropout=drop_rate, recurrent_dropout=drop_rate))
        model.add(keras.layers.RepeatVector(seq_length))
        model.add(keras.layers.LSTM(units, activation='tanh', return_sequences=True,
                                    dropout=drop_rate, recurrent_dropout=drop_rate))
    
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(n_features, activation='linear')))
    learning_rate = hp.Choice('lr', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

def tune_lstm_autoencoder(X_train_seq, X_val_seq):
    seq_length, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
    tuner = kt.BayesianOptimization(
        lambda hp: build_lstm_autoencoder(hp, seq_length, n_features),
        objective='val_loss',
        max_trials=15,
        seed=42,
        directory='tuner_logs',
        project_name='lstm_autoencoder_tuning',
        overwrite=True
    )
    
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    tuner.search(X_train_seq, X_train_seq,
                 validation_data=(X_val_seq, X_val_seq),
                 epochs=50, batch_size=512,
                 callbacks=[stop_early], verbose=1)
    
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best LSTM Autoencoder hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"  {param}: {value}")
    
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        X_train_seq, X_train_seq,
        epochs=100, batch_size=512,
        validation_data=(X_val_seq, X_val_seq),
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0)
    
    best_model.save("best_lstm_autoencoder_model.h5")
    
    # Plot training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('lstm_autoencoder_loss_curve.png')
    plt.close()
    
    # Determine anomaly threshold using 95th percentile of reconstruction errors on validation normals
    recon_val = best_model.predict(X_val_seq)
    errors = np.mean(np.square(X_val_seq - recon_val), axis=(1,2))
    thresh = np.percentile(errors, 95)
    print(f"Chosen reconstruction error threshold for LSTM autoencoder: {thresh:.6f}")
    return best_model, thresh

# ------------------------------
# Model Evaluation and Visualization
# ------------------------------
def evaluate_models(iso_model, ae_model, ae_thresh, lstm_model, lstm_thresh,
                    X_test_scaled, y_test, n_features):
    from sklearn.metrics import recall_score
    
    # Isolation Forest
    y_pred_if = (iso_model.predict(X_test_scaled) == -1).astype(int)
    prec_if = precision_score(y_test, y_pred_if, pos_label=1, zero_division=0)
    rec_if = recall_score(y_test, y_pred_if, pos_label=1, zero_division=0)
    f1_if = f1_score(y_test, y_pred_if, pos_label=1, zero_division=0)
    
    # Autoencoder
    recon_test_ae = ae_model.predict(X_test_scaled)
    errors_test_ae = np.mean(np.square(X_test_scaled - recon_test_ae), axis=1)
    y_pred_ae = (errors_test_ae > ae_thresh).astype(int)
    prec_ae = precision_score(y_test, y_pred_ae, pos_label=1, zero_division=0)
    rec_ae = recall_score(y_test, y_pred_ae, pos_label=1, zero_division=0)
    f1_ae = f1_score(y_test, y_pred_ae, pos_label=1, zero_division=0)
    
    # LSTM Autoencoder (reshape test data as sequences)
    seq_length = 1
    X_test_seq = X_test_scaled.reshape(-1, seq_length, n_features)
    recon_test_lstm = lstm_model.predict(X_test_seq)
    errors_test_lstm = np.mean(np.square(X_test_seq - recon_test_lstm), axis=(1,2))
    y_pred_lstm = (errors_test_lstm > lstm_thresh).astype(int)
    prec_lstm = precision_score(y_test, y_pred_lstm, pos_label=1, zero_division=0)
    rec_lstm = recall_score(y_test, y_pred_lstm, pos_label=1, zero_division=0)
    f1_lstm = f1_score(y_test, y_pred_lstm, pos_label=1, zero_division=0)
    
    print("\n*** Model Performance on Test Set ***")
    print(f"Isolation Forest - Precision: {prec_if:.4f}, Recall: {rec_if:.4f}, F1: {f1_if:.4f}")
    print(f"Autoencoder      - Precision: {prec_ae:.4f}, Recall: {rec_ae:.4f}, F1: {f1_ae:.4f}")
    print(f"LSTM Autoencoder - Precision: {prec_lstm:.4f}, Recall: {rec_lstm:.4f}, F1: {f1_lstm:.4f}")
    
    # Plot Precision-Recall curves for comparison
    anomaly_scores_if = -iso_model.decision_function(X_test_scaled)
    anomaly_scores_ae = errors_test_ae
    anomaly_scores_lstm = errors_test_lstm
    
    prec_curve_if, rec_curve_if, _ = precision_recall_curve(y_test, anomaly_scores_if)
    prec_curve_ae, rec_curve_ae, _ = precision_recall_curve(y_test, anomaly_scores_ae)
    prec_curve_lstm, rec_curve_lstm, _ = precision_recall_curve(y_test, anomaly_scores_lstm)
    
    plt.figure()
    plt.plot(rec_curve_if, prec_curve_if, label='IsolationForest')
    plt.plot(rec_curve_ae, prec_curve_ae, label='Autoencoder')
    plt.plot(rec_curve_lstm, prec_curve_lstm, label='LSTM Autoencoder')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.savefig('precision_recall_curves.png')
    plt.close()

# ------------------------------
# Main Execution
# ------------------------------
def main():
    # Load and preprocess data
    (X_train_full_scaled, train_normals, val_normals, val_anom, 
     X_test_scaled, y_train_full, y_test, scaler) = load_and_preprocess_data('creditcard.csv')
    
    # Tune and train Isolation Forest (using DataFrame versions for indices)
    # Convert scaled arrays back to DataFrame for proper index handling
    train_normals_df = pd.DataFrame(train_normals, columns=train_normals.columns)
    val_normals_df = pd.DataFrame(val_normals, columns=val_normals.columns)
    val_anom_df = pd.DataFrame(val_anom, columns=val_anom.columns)
    iso_model = tune_isolation_forest(train_normals_df, val_normals_df, val_anom_df)
    
    # Tune and train Autoencoder
    # Use training normals for AE training; note: these are already scaled numpy arrays
    ae_model, ae_thresh = tune_autoencoder(scaler.transform(train_normals), scaler.transform(val_normals))
    
    # Prepare data for LSTM Autoencoder (reshape to sequences of length 1)
    n_features = X_train_full_scaled.shape[1]
    X_train_seq = scaler.transform(train_normals).reshape(-1, 1, n_features)
    X_val_seq = scaler.transform(val_normals).reshape(-1, 1, n_features)
    lstm_model, lstm_thresh = tune_lstm_autoencoder(X_train_seq, X_val_seq)
    
    # Evaluate models on the test set
    evaluate_models(iso_model, ae_model, ae_thresh, lstm_model, lstm_thresh,
                    X_test_scaled, y_test, n_features)

if __name__ == "__main__":
    main()