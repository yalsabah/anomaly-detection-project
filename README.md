# Anomaly Detection in Credit Card Transactions

## 📋 Overview
This project implements and compares three unsupervised learning techniques — **Isolation Forest**, **Autoencoder**, and **LSTM Autoencoder** — for detecting anomalies in credit card transaction data.  
The models are evaluated using **Precision**, **Recall**, and **F1-Score**, with key insights visualized through loss curves and precision-recall plots.

---

## 🏗️ Project Structure
.
├── data/                # (Git-ignored) Folder for dataset files
├── models/              # Saved models (Isolation Forest, Autoencoder, LSTM Autoencoder)
├── notebooks/           # Jupyter notebooks for visualization and analysis
├── reports/             # Evaluation visualizations and metrics
├── scripts/
│   ├── run_pipeline.py   # Orchestrates the end-to-end pipeline
│   └── train_and_tune.py # Model training, hyperparameter tuning, evaluation
├── README.md
└── requirements.txt     # Python dependencies

---

## ⚙️ How to Run the Project
1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Download the dataset manually** from [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

3. **Place** the downloaded `creditcard.csv` into the `data/` directory.

4. **Run the full pipeline**:
    ```bash
    python scripts/run_pipeline.py
    ```

5. **Explore the visualizations**:
    - Open the notebook in `/notebooks/` to view model evaluation results and plots.

---

## 📈 Outputs
- Trained models are saved in the `/models/` directory.
- Visualizations (loss curves, precision-recall plots) are saved in the `/reports/` directory.
- Evaluation metrics (Precision, Recall, F1-score) are printed to the console and summarized visually.
- Jupyter notebooks for detailed analysis are available in `/notebooks/`.

---

## 📚 Dataset
The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains transactions made by European cardholders in September 2013.  
Note: The dataset is **not included** in this repository due to file size restrictions. Please download it manually.

---
