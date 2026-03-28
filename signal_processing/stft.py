import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

# Load normalized data
data = pd.read_csv("../data/normalized_stock_data.csv", index_col="Date", parse_dates=True)

os.makedirs("../outputs", exist_ok=True)

companies = data.columns.tolist()

for company in companies:
    signal = data[company].values
    print(f"Processing {company}...")

    # ── 1. Time Series Plot ──────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, signal, color='blue')
    plt.title(f"Time Series - {company}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.tight_layout()
    plt.savefig(f"../outputs/timeseries_{company}.png")
    plt.close()

    # ── 2. Frequency Spectrum (FFT) ──────────────────────
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal))
    plt.figure(figsize=(12, 4))
    plt.plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2], color='green')
    plt.title(f"Frequency Spectrum - {company}")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"../outputs/frequency_{company}.png")
    plt.close()

    # ── 3. Spectrogram (STFT) ────────────────────────────
    f, t, Zxx = stft(signal, fs=1.0, window='hann', nperseg=64, noverlap=32)
    spectrogram = np.abs(Zxx) ** 2
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, 10 * np.log10(spectrogram + 1e-10), shading='gouraud', cmap='inferno')
    plt.colorbar(label='Energy (dB)')
    plt.title(f"Spectrogram - {company}")
    plt.xlabel("Time (days)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"../outputs/spectrogram_{company}.png")
    plt.close()

    print(f"  ✅ Plots saved for {company}")

print("\nAll done! Check the outputs/ folder for your plots.")


## ▶️ Run it in Terminal

## cd ../signal_processing
## python stft.py
