# Experiments

Jupyter notebooks used for exploratory analysis and model comparison (Linear Regression, SVR, Random Forest, LSTM, LightGBM, ARIMA, Exponential Smoothing). These experiments led to selecting **Exponential Smoothing** for the production API.

**Notebooks:**
- `Data Exploration and Forecasting using ARIMA, Exponential Smoothing.ipynb` — Exploration and ARIMA / Exponential Smoothing (best performer).
- `Forecasting Using Machine Learning Models.ipynb` — M5 store-level preprocessing and baseline ML models (Linear Regression, SVR, Random Forest).
- `Forecasting Using LSTM.ipynb` — LSTM time series experiments.
- `Forecasting Using LightGBM.ipynb` — LightGBM forecasting experiments.

## MLflow logging

Each notebook logs **params, metrics, and artifacts** to MLflow (one experiment per notebook). Run notebooks manually; each run will appear in the MLflow UI.

| Notebook | MLflow experiment | What’s logged |
|----------|-------------------|----------------|
| Data Exploration and Forecasting using ARIMA, Exponential Smoothing | `experiments-arima-exponential-smoothing` | Params (order, seasonal_order, trend, etc.), metric `rmse`, artifact: pickled model per run (SARIMAX-1, SARIMAX-2, SARIMA-auto_arima, Exponential Smoothing) |
| Forecasting Using Machine Learning Models | `experiments-ml-models` | **sklearn autolog**: params, metrics, and model artifact per run (Linear Regression, SVR, Random Forest) |
| Forecasting Using LSTM | `experiments-lstm` | **TensorFlow autolog**: params, metrics, and model artifact |
| Forecasting Using LightGBM | `experiments-lightgbm` | **LightGBM autolog**: params, metrics, and model artifact |

**Exponential Smoothing (chosen model):** Further runs (e.g. from `train.py` or tuning) use the **`exponential-smoothing`** experiment so all follow-up testing is in one place.

**View runs:** From the project root run `mlflow ui` and open http://127.0.0.1:5000. Runs are stored in `./mlflow.db`.

**Running locally:** The notebooks were developed on Colab with data in Google Drive. To run locally with the repo’s `data/` folder, point the data path in each notebook to `../data/` (or the project’s `data` directory). Production training and the FastAPI app use `src/` and `train.py` with data in the project root’s `data/` folder.
