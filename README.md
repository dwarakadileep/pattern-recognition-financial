# Pattern Recognition for Financial Time Series Forecasting

## Objective
This project explores how time-frequency signal processing and deep learning
can be combined to predict stock prices using financial time series data.

## Companies Used
- TCS (TCS.NS)
- Infosys (INFY.NS)
- Reliance (RELIANCE.NS)

## Folder Structure
```
pattern-recognition-financial/
├── data/                  # Raw and normalized stock data (CSV)
├── preprocessing/         # Data fetching and normalization
├── signal_processing/     # STFT and spectrogram generation
├── model/                 # CNN model training and prediction
├── outputs/               # All generated plots and results
└── README.md
```

## How to Run

### Step 1 - Install Dependencies
```
pip install yfinance pandas numpy matplotlib scipy tensorflow scikit-learn
```

### Step 2 - Fetch and Normalize Data
```
cd preprocessing
python fetch_data.py
```

### Step 3 - Generate Spectrograms
```
cd signal_processing
python stft.py
```

### Step 4 - Train CNN Model
```
cd model
python cnn_model.py
```

## Methodology
- Financial time series data is treated as a signal
- Short-Time Fourier Transform (STFT) converts it into spectrograms
- A CNN model learns patterns from spectrograms to predict future prices

## Results
- Training Loss reduced from 0.37 to 0.014 over 20 epochs
- Test MSE: 0.011758
- CNN successfully learned patterns from spectrogram representations

## Output Figures
| File | Description |
|---|---|
| timeseries_*.png | Stock price over time |
| frequency_*.png | Frequency spectrum |
| spectrogram_*.png | STFT Spectrogram |
| cnn_predictions.png | Predicted vs Actual prices |
| training_loss.png | CNN training loss curve |

## Data Sources
- Yahoo Finance: https://finance.yahoo.com

## References
1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting"
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997

## Final Push to GitHub

1. Save the file (**Ctrl+S**)
2. Source Control icon → commit message: `Add project README`
3. **Commit** → **Sync Changes** ✅

```
https://github.com/dwarakadileep/pattern-recognition-financial

```

## Quick Start
Run the full pipeline with:
```
cd preprocessing && python fetch_data.py
cd ../signal_processing && python stft.py
cd ../model && python cnn_model.py
```
The webpage will open automatically after training completes!