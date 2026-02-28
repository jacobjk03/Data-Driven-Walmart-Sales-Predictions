"""Load and preprocess M5 Walmart sales data for store-level forecasting."""
from pathlib import Path
from datetime import datetime

import pandas as pd

from .config import SALES_FILE, CALENDAR_FILE


def load_sales_and_calendar(
    sales_path: Path = SALES_FILE,
    calendar_path: Path = CALENDAR_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sales_train_validation and calendar CSVs."""
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    return sales, calendar


def build_store_level_series(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate sales by store and merge with calendar (same logic as notebook).
    Returns DataFrame with columns: date, CA_1, CA_2, ... (store_id columns).
    """
    numerical_columns = sales.select_dtypes(include=["number"])
    store_level = numerical_columns.groupby(sales["store_id"]).sum()
    store_levelt = store_level.T
    store_levelt["d"] = store_levelt.index
    merged = store_levelt.merge(calendar[["d", "date"]], on="d", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def get_store_series(store_level_df: pd.DataFrame, store_id: str) -> pd.Series:
    """Get a single store's daily sales as a pandas Series with DatetimeIndex."""
    if store_id not in store_level_df.columns:
        raise ValueError(f"Unknown store_id: {store_id}. Available: {list(store_level_df.columns)}")
    s = store_level_df.set_index("date")[store_id]
    s = s.sort_index()
    return s


def load_store_level_data(
    sales_path: Path = SALES_FILE,
    calendar_path: Path = CALENDAR_FILE,
) -> pd.DataFrame:
    """Load data and return store-level DataFrame (date + one column per store)."""
    sales, calendar = load_sales_and_calendar(sales_path, calendar_path)
    return build_store_level_series(sales, calendar)


def get_available_stores(store_level_df: pd.DataFrame) -> list[str]:
    """Return list of store_id column names (exclude 'd' and 'date')."""
    return [c for c in store_level_df.columns if c not in ("d", "date")]
