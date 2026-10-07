"""Tests for src/features.py. They use small synthetic frames, so no data download is needed."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (ACROPOLIS, add_features, add_targets, check_inputs,
                          clean_price, haversine_km, leaked_inputs, parse_quote)

ROOT = Path(__file__).resolve().parents[1]

# Written out independently of src.features.LEAKY_COLUMNS, so deleting an entry
# from that list makes this test fail instead of silently shrinking it.
MUST_NEVER_BE_FEATURES = [
    "estimated_revenue_l365d", "price_quote_total_price",
    "price_quote_price_per_night", "price_quote_raw",
]
# The targets and the discount flag. Also written out independently.
TARGETS_AND_DISCOUNT = ["price_list", "price_quoted", "price", "has_discount"]


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


# --- Guards on the input lists the models are actually trained on ---------------------

def list_assignments(source: str) -> dict:
    """Return {name: list} for every `NAME = [..literal strings..]` in Python source."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.List)):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def notebook_lists() -> dict:
    nb = json.loads((ROOT / "notebooks" / "01_athens_price_model.ipynb").read_text())
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            found.update(list_assignments("".join(cell["source"])))
    return found


@pytest.mark.parametrize("col", MUST_NEVER_BE_FEATURES + TARGETS_AND_DISCOUNT)
def test_guard_rejects_every_forbidden_column(col):
    assert leaked_inputs(["bedrooms", col]) == [col]
    with pytest.raises(ValueError):
        check_inputs(["bedrooms", col])


def test_guard_accepts_clean_inputs():
    check_inputs(["bedrooms", "km_acropolis", "room_type"])


def test_main_model_inputs_in_notebook_are_clean():
    lists = notebook_lists()
    inputs = lists["NUMERIC"] + lists["CATEGORICAL"] + lists["BASE_CATEGORICAL"]
    assert len(lists["NUMERIC"]) == 19 and len(lists["CATEGORICAL"]) == 3
    assert leaked_inputs(inputs) == []


def test_demo_model_inputs_are_clean():
    lists = list_assignments((ROOT / "scripts" / "export_demo.py").read_text())
    assert lists["FEATURES"], "demo feature list not found"
    assert leaked_inputs(lists["FEATURES"]) == []


def test_notebook_builder_and_notebook_use_the_same_inputs():
    """The notebook is generated by scripts/build_notebook.py. Both must list the same inputs."""
    builder = {}
    for node in ast.walk(ast.parse((ROOT / "scripts" / "build_notebook.py").read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "NUMERIC = [" in node.value:
            builder.update(list_assignments(node.value))
    nb = notebook_lists()
    for name in ["NUMERIC", "CATEGORICAL", "BASE_CATEGORICAL"]:
        assert builder[name] == nb[name], name


def test_notebook_outputs_show_no_single_listings():
    """Published notebook outputs must not identify listings or hosts (see DATA_PROTECTION.md)."""
    import re
    nb = json.loads((ROOT / "notebooks" / "01_athens_price_model.ipynb").read_text())
    parts = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", "")))
            for kind in ["text/plain", "text/html"]:
                parts.append("".join(out.get("data", {}).get(kind, "")))
    text = "\n".join(parts)
    assert not re.search(r"airbnb\.[a-z.]+/(rooms|users)", text), "listing or host URL in outputs"
    assert "muscache.com" not in text, "photo URL in outputs"
    for col in ["host_name", "host_about", "listing_url", "picture_url", "below_pred_pct"]:
        assert col not in text, f"{col} column shown in outputs"
    assert not re.search(r"\b\d{8,19}\b", text), "listing-ID-like number in outputs"
