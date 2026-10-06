"""Train the small demo model and export it, plus an aggregated price grid, as JSON.

The demo runs entirely in the browser, so it gets a compact LightGBM model on the
inputs a visitor can set, and a map built from grid cells (never single listings).
Run from the repo root:  python scripts/export_demo.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import ACROPOLIS, add_features, add_targets  # noqa: E402

FEATURES = ["latitude", "longitude", "km_acropolis", "room_code", "accommodates",
            "bedrooms", "bathrooms", "review_scores_rating", "host_is_superhost"]
ROOMS = {"Entire home/apt": 0, "Private room": 1}
CELL_LAT, CELL_LON = 0.0045, 0.0057   # roughly 500 m x 500 m in Athens
MIN_PER_CELL = 5
SEED = 42


def load():
    df = add_features(add_targets(pd.read_csv(ROOT / "data" / "listings.csv")))
    df = df.dropna(subset=["price_list", "price_quoted"])
    high = df["price_list"].quantile(0.99)
    df = df[df["price_list"].between(20, high) & df["price_quoted"].between(20, high)]
    df = df[df["room_type"].isin(ROOMS)].copy()
    df["room_code"] = df["room_type"].map(ROOMS)
    return df.reset_index(drop=True)


def make_model():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, random_state=SEED, verbose=-1)


def cv_mae(df):
    X, y = df[FEATURES], np.log1p(df["price_list"].values)
    errs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, df["host_id"]):
        pred = np.expm1(make_model().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
        errs.append(np.mean(np.abs(pred - df["price_list"].values[te])))
    return float(np.mean(errs)), float(np.std(errs))


def flatten(node, feats, thr, left, right, leaf):
    """Store a tree as parallel arrays. Leaves get feature index -1."""
    i = len(feats)
    feats.append(-1); thr.append(0.0); left.append(-1); right.append(-1); leaf.append(0.0)
    if "leaf_value" in node:
        leaf[i] = round(node["leaf_value"], 6)
        return i
    assert node["decision_type"] == "<=", node["decision_type"]
    feats[i] = node["split_feature"]
    thr[i] = float(node["threshold"])
    left[i] = flatten(node["left_child"], feats, thr, left, right, leaf)
    right[i] = flatten(node["right_child"], feats, thr, left, right, leaf)
    return i


def export_trees(model):
    trees = []
    for t in model.booster_.dump_model()["tree_info"]:
        f, th, l, r, v = [], [], [], [], []
        flatten(t["tree_structure"], f, th, l, r, v)
        trees.append([f, th, l, r, v])
    return trees


def grid(df):
    homes = df[df["room_code"] == 0].copy()
    homes["gy"] = np.floor(homes["latitude"] / CELL_LAT).astype(int)
    homes["gx"] = np.floor(homes["longitude"] / CELL_LON).astype(int)
    g = homes.groupby(["gy", "gx"]).agg(n=("price_list", "size"), med=("price_list", "median"))
    g = g[g["n"] >= MIN_PER_CELL].reset_index()
    return [[int(a), int(b), int(n), round(float(m))] for a, b, n, m in g.itertuples(index=False)]


def main():
    df = load()
    mae, sd = cv_mae(df)
    model = make_model().fit(df[FEATURES], np.log1p(df["price_list"].values))
    out = {
        "features": FEATURES, "rooms": ROOMS, "acropolis": ACROPOLIS,
        "cell": [CELL_LAT, CELL_LON], "min_per_cell": MIN_PER_CELL,
        "grid": grid(df), "trees": export_trees(model),
        "cv_mae": round(mae, 2), "cv_mae_sd": round(sd, 2), "n_train": len(df),
        "defaults": {"review_scores_rating": round(float(df["review_scores_rating"].median()), 2)},
        "scraped": "29 June 2026",
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    path = ROOT / "docs" / "model.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    sample = df.sample(500, random_state=SEED)
    sample[FEATURES].assign(pred=np.expm1(model.predict(sample[FEATURES]))).to_json(
        ROOT / "docs" / ".check.json", orient="records")
    print(f"rows {len(df)} | demo CV MAE {mae:.2f} +/- {sd:.2f} | cells {len(out['grid'])} "
          f"| trees {len(out['trees'])} | {path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
