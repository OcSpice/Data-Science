"""Reusable EDA summaries for the retail forecasting project."""

from __future__ import annotations

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return core dataset dimensions and date coverage."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "unique_series",
                "unique_stores",
                "unique_items",
                "start_date",
                "end_date",
                "mean_daily_sales",
                "median_daily_sales",
            ],
            "value": [
                len(frame),
                frame["id"].nunique(),
                frame["store_id"].nunique(),
                frame["item_id"].nunique(),
                frame["date"].min().strftime("%Y-%m-%d"),
                frame["date"].max().strftime("%Y-%m-%d"),
                frame["sales"].mean(),
                frame["sales"].median(),
            ],
        }
    )


def daily_sales_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total and average demand by calendar date."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return (
        frame.groupby("date", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_sales=("sales", "mean"),
            active_series=("id", "nunique"),
        )
        .sort_values("date")
    )


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand by product category."""
    return (
        df.groupby("cat_id", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_daily_sales=("sales", "mean"),
            series_count=("id", "nunique"),
        )
        .sort_values("total_sales", ascending=False)
    )


def store_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand by store."""
    return (
        df.groupby(["state_id", "store_id"], as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_daily_sales=("sales", "mean"),
            series_count=("id", "nunique"),
        )
        .sort_values("total_sales", ascending=False)
    )


def weekday_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Measure demand by day of week."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["weekday_name"] = frame["date"].dt.day_name()
    order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    result = (
        frame.groupby("weekday_name", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            average_sales=("sales", "mean"),
        )
    )
    result["weekday_name"] = pd.Categorical(
        result["weekday_name"], categories=order, ordered=True
    )
    return result.sort_values("weekday_name")
