# Data-Driven Walmart Sales Predictions

## Project Overview

This project focuses on predicting store-level sales at Walmart using time series forecasting models. Accurate sales predictions are critical for managing inventory, optimizing business metrics like Gross Margin and Average Order Value (AOV), and enhancing customer satisfaction. By leveraging multiple machine learning and statistical models, this project demonstrates a significant improvement in forecast accuracy, leading to optimized business performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Analyzed](#models-analyzed)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [DVC (Data & Model Versioning)](#dvc-data--model-versioning)
- [MLflow from scratch (local tracking only)](#mlflow-from-scratch-local-tracking-only)
- [Training (Exponential Smoothing + MLflow)](#training-exponential-smoothing--mlflow)
- [Web API (FastAPI)](#web-api-fastapi)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used in this project is from the [M5 Forecasting Accuracy competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy). It contains sales data from Walmart across various departments and stores, as well as calendar and price data that influence demand. The goal is to predict sales for 31 days in the future at the store-item level.

## Models Analyzed

We explored various models to identify the best fit for accurate sales forecasting:
- **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
- **Exponential Smoothing**
- **LSTM (Long Short-Term Memory)**
- **Random Forest**
- **LightGBM**

Each model was carefully evaluated based on its Root Mean Square Error (RMSE), with Exponential Smoothing emerging as the best performer, achieving a 38.92% improvement in RMSE compared to LSTM.

## Results

- **Exponential Smoothing** showed the best results with a **38.92% improvement in RMSE** over LSTM.
- Optimized inventory management led to improvements in key business metrics such as:
  - **Gross Margin**
  - **Average Order Value (AOV)**
  - **Customer Satisfaction**
- These improvements enhanced sales prediction accuracy, which in turn, led to better stock planning and increased business performance.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**: 
  - Data Analysis: `pandas`, `NumPy`
  - Visualization: `matplotlib`
  - Machine Learning: `scikit-learn`, `TensorFlow`, `Keras`
  - Statistical Modeling: `statsmodels`
  
## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/jacobjk03/Data-Driven-Walmart-Sales-Predictions.git
    cd Data-Driven-Walmart-Sales-Predictions
    ```

2. Create a virtual environment and install dependencies.  
   **Note:** TensorFlow (used in the LSTM notebook) does not support Python 3.13+. Use **Python 3.11 or 3.12** if you want to run that notebook:
    ```bash
    # Option A: venv with Python 3.12 (if installed)
    python3.12 -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

    # Option B: Conda with Python 3.12
    conda create -n walmart python=3.12 -y
    conda activate walmart
    pip install -r requirements.txt
    ```
   To install TensorFlow for the LSTM notebook: `pip install "tensorflow>=2.15"` (only on Python 3.11 or 3.12).

3. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) and place the M5 files in a `data/` folder:
   - `sales_train_validation.csv`
   - `calendar.csv`

## DVC (Data & Model Versioning)

This project uses [DVC](https://dvc.org/) to version the **data** directory and the **models** produced by the pipeline (local storage only). Config: **`params.yaml`** (training parameters), **`dvc.yaml`** (pipeline: train stage).

**One-time setup (from project root):**
```bash
pip install dvc
dvc init
dvc remote add -d storage .dvc-store
dvc add data
git add data.dvc .gitignore dvc.yaml params.yaml .dvc
git commit -m "Add DVC: track data and train pipeline"
```

**Workflow:**
- **Reproduce pipeline (train from data → models):**  
  `dvc repro`
- **Push data/models to local storage (after changing data or re-training):**  
  `dvc push`
- **Pull data (e.g. after clone):**  
  `dvc pull`
- **Change training params:** Edit `params.yaml`, then run `dvc repro`.

The **train** stage in `dvc.yaml` depends on `data/` and `params.yaml`, and writes to `models/`. Data is tracked as a whole directory via `data.dvc`.

## MLflow from scratch (local tracking only)

All tracking is **local** in **`mlflow.db`** (SQLite) in the project root (no remote server). To start clean and run everything:

1. **Start the MLflow UI** from project root:
   ```bash
   mlflow ui
   ```
   Open **http://127.0.0.1:5000** in your browser. (To use another port: `mlflow ui --port 5001`.)

2. **Run the Jupyter notebooks** in **`Notebooks/`** (each creates its own experiment in `mlflow.db`):
   - *Data Exploration and Forecasting using ARIMA, Exponential Smoothing*
   - *Forecasting Using Machine Learning Models*
   - *Forecasting Using LightGBM*
   - *Forecasting Using LSTM* (optional; needs TensorFlow / Python 3.12)

3. **After running the notebooks:**
   - **View runs:** Keep `mlflow ui` running (or start it) and open **http://127.0.0.1:5000** to compare experiments and metrics.
   - **Train the API model:** Run `python train.py --store-id CA_1` (and other stores if needed) so the FastAPI app can serve predictions.
   - **Start the API:** Run `uvicorn src.app:app --reload --host 0.0.0.0 --port 8000` and use **http://127.0.0.1:8000/docs** to call `/predict`.

4. **Log the chosen model** (Exponential Smoothing) for the API:
   ```bash
   python train.py --store-id CA_1
   ```
   This adds runs to the **exponential-smoothing** experiment.

To **reset and start over**: delete **`mlflow.db`** (and the **`mlruns/`** folder if present), then repeat the steps above.

## Training (Exponential Smoothing + MLflow)

1. Train the Exponential Smoothing model (one store per run):
   ```bash
   python train.py --store-id CA_1
   ```
   Options: `--data-dir data`, `--trend add`, `--seasonal add`, `--seasonal-periods 7`, `--train-split 0.8`.  
   The script logs **params** (store_id, trend, seasonal, etc.) and **metrics** (RMSE), and saves the model under `models/`.

2. Train additional stores to serve them via the API:
   ```bash
   python train.py --store-id TX_1
   python train.py --store-id WI_1
   ```

## Web API (FastAPI)

The forecasting API serves predictions from the trained Exponential Smoothing models.

1. Start the API (from project root):
   ```bash
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. Open **http://127.0.0.1:8000/docs** for interactive Swagger UI.

3. **Endpoints:**
   - `GET /health` — health check
   - `GET /stores` — list store IDs with a trained model
   - `POST /predict` — body: `{"store_id": "CA_1", "steps": 7}` → returns `{"store_id", "steps", "forecast": [...]}`


## Conclusion

By analyzing and fine-tuning different models, the project successfully improved Walmart’s sales forecasting, resulting in more effective inventory management and improved business outcomes. The best performance was achieved using Exponential Smoothing, demonstrating its capability in time series forecasting.

## Acknowledgments

- The dataset is provided by the [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition on Kaggle.
- Special thanks to Walmart for their anonymized sales data.
