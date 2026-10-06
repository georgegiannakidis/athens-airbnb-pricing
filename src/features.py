"""Feature helpers for the Athens Airbnb pricing project."""
import json

import numpy as np
import pandas as pd

ACROPOLIS = (37.9715, 23.7257)
SYNTAGMA = (37.9755, 23.7348)

# Columns that are the price, or are calculated from it. Never used as features.
LEAKY_COLUMNS = [
    "estimated_revenue_l365d",
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


def parse_quote(raw) -> dict:
    """Pull the pre-discount subtotal and discount out of the quote JSON."""
    try:
        q = json.loads(raw)["quote"]
        return {"nightly_subtotal": float(q.get("nightly_subtotal") or np.nan),
                "discount_amount": float(q.get("discount_amount") or 0)}
    except (TypeError, ValueError, KeyError):
        return {"nightly_subtotal": np.nan, "discount_amount": np.nan}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Two targets for the same quote.

    price_quoted: what Airbnb showed per night. Can include an explicit discount
                  (mostly labelled "Special offer") or taxes.
    price_list:   the pre-discount nightly quote rate (quote subtotal / nights),
                  before explicit discounts, taxes and fees.
    """
    out = df.copy()
    out["price_quoted"] = clean_price(out["price"])
    q = pd.DataFrame([parse_quote(r) for r in out["price_quote_raw"]], index=out.index)
    checkin = pd.to_datetime(out["price_quote_checkin_date"], errors="coerce")
    checkout = pd.to_datetime(out["price_quote_checkout_date"], errors="coerce")
    out["quote_nights"] = (checkout - checkin).dt.days
    out["price_list"] = q["nightly_subtotal"] / out["quote_nights"]
    out["has_discount"] = (q["discount_amount"] > 0).astype(int)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # The detailed file leaves `neighbourhood` blank; real names are in `neighbourhood_cleansed`.
    if "neighbourhood_cleansed" in out.columns:
        out["neighbourhood"] = out["neighbourhood_cleansed"]

    out["km_acropolis"] = haversine_km(out["latitude"], out["longitude"], ACROPOLIS)
    out["km_syntagma"] = haversine_km(out["latitude"], out["longitude"], SYNTAGMA)
    out["reviews_per_month"] = out["reviews_per_month"].fillna(0)
    out["minimum_nights"] = out["minimum_nights"].clip(upper=30)
    out["is_multi_host"] = (out["calculated_host_listings_count"] > 1).astype(int)
    out["host_is_superhost"] = out["host_is_superhost"].map({"t": 1, "f": 0})

    # Quote conditions: when the quoted stay starts and how long it is.
    checkin = pd.to_datetime(out["price_quote_checkin_date"], errors="coerce")
    out["lead_days"] = (checkin - pd.to_datetime(out["last_scraped"])).dt.days
    out["checkin_month"] = checkin.dt.month

    return out.drop(columns=[c for c in LEAKY_COLUMNS if c in out.columns])
