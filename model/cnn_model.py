import pandas as pd
import numpy as np
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ── Load normalized data ─────────────────────────────────
data = pd.read_csv("../data/normalized_stock_data.csv", index_col="Date", parse_dates=True)

os.makedirs("../outputs", exist_ok=True)

# ── Generate Spectrogram for each company ────────────────
def get_spectrogram(signal, nperseg=64, noverlap=32):
    f, t, Zxx = stft(signal, fs=1.0, window='hann', nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx) ** 2  # shape: (freq_bins, time_steps)

# ── Prepare Dataset ──────────────────────────────────────
X = []  # spectrograms (inputs)
y = []  # future prices (targets)

WINDOW = 64   # window length
HOP    = 32   # hop size
FUTURE = 5    # predict 5 days ahead

companies = data.columns.tolist()

for company in companies:
    signal = data[company].values
    # Slide window across signal
    for i in range(0, len(signal) - WINDOW - FUTURE, HOP):
        segment = signal[i:i+WINDOW]
        target  = signal[i+WINDOW+FUTURE-1]  # price 5 days ahead
        f, t, Zxx = stft(segment, fs=1.0, window='hann', nperseg=32, noverlap=16)
        spec = np.abs(Zxx) ** 2
        X.append(spec)
        y.append(target)

X = np.array(X)
y = np.array(y)

# Add channel dimension for CNN
X = X[..., np.newaxis]  # shape: (samples, freq, time, 1)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ── Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── CNN Model ────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=X.shape[1:]),
    layers.Conv2D(32, (2,2), activation='relu', padding='same'),
    layers.MaxPooling2D((2,1)),
    layers.Conv2D(64, (2,2), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ── Train ────────────────────────────────────────────────
print("\nTraining CNN model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────────
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f"\nTest MSE: {mse:.6f}")

# ── Plot: Predictions vs Actual ──────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label='Actual', color='blue')
plt.plot(y_pred[:100], label='Predicted', color='red', linestyle='--')
plt.title("CNN Predictions vs Actual Stock Prices")
plt.xlabel("Sample")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/cnn_predictions.png")
plt.close()

# ── Plot: Training Loss ───────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/training_loss.png")
plt.close()

print("\nDone! Check outputs/ for prediction plots!")

import webbrowser
import subprocess
import sys
import os

# Save results to a summary file for the webpage
with open("../outputs/results.txt", "w") as f:
    f.write(f"Test MSE: {mse:.6f}\n")
    f.write(f"Final Training Loss: {history.history['loss'][-1]:.6f}\n")

# Auto open the webpage in browser
html_path = os.path.abspath("../index.html")
print(f"\nOpening project webpage...")
webbrowser.open(f"file://{html_path}")

## ▶️ Run it in Terminal

## cd ../model
## python cnn_model.py

