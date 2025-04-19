# Anomaly Detection in Time-Series Data (Credit Card Fraud)

## Overview
This project compares three unsupervised learning methods—Isolation Forest, Autoencoder, and LSTM Autoencoder—for detecting anomalies in the Kaggle Credit Card Fraud dataset. The models are evaluated using precision, recall, and F1 score metrics, with visualizations saved for further analysis.

## Project Structure
```
.
├── data/                # Raw dataset (creditcard.csv)
├── models/              # Saved models (.pkl and .h5)
├── reports/             # Evaluation visualizations and loss curves
├── scripts/
│   ├── run_pipeline.py  # Orchestrates the full pipeline end-to-end
│   └── train_and_tune.py # Model training, tuning, evaluation
├── README.md
└── requirements.txt
```

## How to Run
Make sure you have Python 3.8+ installed, then:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

## Output
- Trained models are saved in `/models`
- Loss curves and precision-recall plots are saved in `/reports`
- Evaluation metrics are printed to the console

## Dataset
[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
