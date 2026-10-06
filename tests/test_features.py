"""Tests for src/features.py. They use small synthetic frames, so no data download is needed."""
import json

import numpy as np
import pandas as pd
import pytest

from src.features import (ACROPOLIS, add_features, add_targets, clean_price,
                          haversine_km, parse_quote)

# Written out independently of src.features.LEAKY_COLUMNS, so deleting an entry
# from that list makes this test fail instead of silently shrinking it.
MUST_NEVER_BE_FEATURES = [
    "estimated_revenue_l365d", "price_quote_total_price",
    "price_quote_price_per_night", "price_quote_raw",
]


def quote(subtotal, discount=None, per_night="0"):
    return json.dumps({"quote": {"nightly_subtotal": str(subtotal),
                                 "discount_amount": None if discount is None else str(discount),
                                 "price_per_night": per_night}})


def listing(**overrides):
    row = {
        "id": 1, "host_id": 10, "price": "$100.00",
        "price_quote_raw": quote(200), "price_quote_checkin_date": "2026-07-01",
        "price_quote_checkout_date": "2026-07-03", "price_quote_total_price": "200",
        "price_quote_price_per_night": "100", "estimated_revenue_l365d": 9999,
        "last_scraped": "2026-06-29", "latitude": ACROPOLIS[0], "longitude": ACROPOLIS[1],
        "neighbourhood": np.nan, "neighbourhood_cleansed": "PLAKA",
        "reviews_per_month": np.nan, "minimum_nights": 400,
        "calculated_host_listings_count": 3, "host_is_superhost": "t",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_haversine_zero_at_same_point():
    assert haversine_km(*ACROPOLIS, ACROPOLIS) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_latitude_is_about_111_km():
    assert haversine_km(ACROPOLIS[0] + 1, ACROPOLIS[1], ACROPOLIS) == pytest.approx(111.2, abs=0.5)


def test_clean_price_handles_symbols_commas_and_blanks():
    out = clean_price(pd.Series(["$1,712.00", "$50.21", "", None]))
    assert out.iloc[0] == 1712.0 and out.iloc[1] == 50.21
    assert out.iloc[2:].isna().all()


def test_parse_quote_reads_subtotal_and_missing_discount_as_zero():
    assert parse_quote(quote(183.64)) == {"nightly_subtotal": 183.64, "discount_amount": 0.0}


def test_parse_quote_survives_bad_json():
    out = parse_quote("not json")
    assert np.isnan(out["nightly_subtotal"])


def test_list_rate_is_subtotal_divided_by_nights():
    out = add_targets(listing(price_quote_raw=quote(300, discount=60)))
    assert out["quote_nights"].iloc[0] == 2
    assert out["price_list"].iloc[0] == pytest.approx(150.0)
    assert out["has_discount"].iloc[0] == 1
    assert out["price_quoted"].iloc[0] == pytest.approx(100.0)


def test_leakage_guard_drops_every_price_derived_column():
    out = add_features(add_targets(listing()))
    for col in MUST_NEVER_BE_FEATURES:
        assert col not in out.columns, f"{col} leaked into features"


def test_features_fill_neighbourhood_and_cap_minimum_nights():
    out = add_features(add_targets(listing()))
    row = out.iloc[0]
    assert row["neighbourhood"] == "PLAKA"
    assert row["minimum_nights"] == 30
    assert row["km_acropolis"] == pytest.approx(0.0, abs=1e-6)
    assert row["host_is_superhost"] == 1 and row["is_multi_host"] == 1
    assert row["lead_days"] == 2 and row["checkin_month"] == 7
    assert row["reviews_per_month"] == 0
