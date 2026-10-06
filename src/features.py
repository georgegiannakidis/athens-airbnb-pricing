"""Feature helpers for the Athens Airbnb pricing project."""
import numpy as np
import pandas as pd

ACROPOLIS = (37.9715, 23.7257)
SYNTAGMA = (37.9755, 23.7348)

# Columns calculated from price. Using them as features would leak the answer.
LEAKY_COLUMNS = [
    "estimated_revenue_l365d",
    "price_quote_checkin_date", "price_quote_checkout_date",
    "price_quote_total_price", "price_quote_price_per_night", "price_quote_raw",
]


def haversine_km(lat, lon, point):
    """Great-circle distance in km from each (lat, lon) to a fixed point."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(point[0]), np.radians(point[1])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean_price(series: pd.Series) -> pd.Series:
    """Turn price into a float. Handles numbers and text like '$1,234.00'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns]).copy()

    # The detailed file leaves `neighbourhood` blank; real names are in `neighbourhood_cleansed`.
    if "neighbourhood_cleansed" in out.columns:
        out["neighbourhood"] = out["neighbourhood_cleansed"]

    out["km_acropolis"] = haversine_km(out["latitude"], out["longitude"], ACROPOLIS)
    out["km_syntagma"] = haversine_km(out["latitude"], out["longitude"], SYNTAGMA)
    out["reviews_per_month"] = out["reviews_per_month"].fillna(0)
    out["minimum_nights"] = out["minimum_nights"].clip(upper=30)
    out["is_multi_host"] = (out["calculated_host_listings_count"] > 1).astype(int)

    for col in ["host_is_superhost", "instant_bookable"]:
        if col in out.columns:
            out[col] = out[col].map({"t": 1, "f": 0})
    return out
