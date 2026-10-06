# Model card: Athens Airbnb nightly rate model

## What it does

Estimates the **pre-discount nightly quote rate** (EUR) of an Athens Airbnb listing from its size, location, property type, quality signals, host behaviour and quote conditions.

It answers "what do comparable listings ask?" It does **not** answer "what will guests pay?" or "what should this listing charge?"

## Intended use

- Portfolio and teaching example of a leakage-aware pricing analysis.
- Exploring how asking rates vary across Athens, especially with distance from the Acropolis.

## Out of scope

- Setting real prices. The model learns from asking prices, not from bookings, so it cannot tell a rate that sells from one that sits empty.
- Judging individual hosts or listings. "Possibly underpriced" is a statistical flag, not evidence of a mistake.
- Other cities, other seasons, or dates far from June to July 2026.
- Any causal claim. The Acropolis gradient is an association.

## Data

| | |
|---|---|
| Source | Inside Airbnb, Athens detailed listings, scraped 29 June 2026 (CC BY 4.0) |
| Rows used | 13,998 of 14,337 listings |
| Target | `price_list` = quote subtotal / nights, before explicit discounts, taxes and fees |
| Comparison target | `price_quoted` = per-night price as shown, which can include discounts or taxes |
| Quote dates | 29 June 2026 to 16 June 2027; 92% in June to July 2026 |

## Model

LightGBM and Random Forest on log price, 19 numeric and 3 categorical features (room type, neighbourhood, grouped property type). Price-derived columns are removed by a guard in `src/features.py`, and a unit test checks that guard.

## Evaluation

5-fold cross-validation grouped by host (each host appears in only one fold).

| Model | MAE on `price_list` (EUR, mean ± sd over folds) |
|---|---|
| Baseline: median of same neighbourhood and room type | 50.53 ± 3.39 |
| Random Forest | 35.55 ± 2.50 |
| LightGBM | 35.49 ± 2.55 |

The two models are statistically tied.

## Known weaknesses

- **Cheap listings (under EUR 60):** about 50% mean error. The model over-predicts almost all of them; the cause is not identified.
- **Expensive listings (EUR 250+):** about 37% mean error.
- **Thin groups:** few hotel rooms and shared rooms; estimates for them are unreliable.
- **One snapshot:** no seasonality, no booking outcomes.

## Fairness and people

The model does not use host names, photos, descriptions or any attribute describing a person. `host_id` is used only to keep each host inside a single validation fold. See [DATA_PROTECTION.md](DATA_PROTECTION.md).

## Demo model

The live estimator uses a smaller LightGBM trained on the inputs a visitor can set. Its accuracy is reported on the demo page and is lower than the main model's.
