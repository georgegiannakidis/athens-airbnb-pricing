# Methods and full results

This page holds the technical detail behind the [README](README.md). The full analysis, with code and outputs, is in the [notebook](notebooks/01_athens_price_model.ipynb).

## Key findings in detail

**A steep price gradient around the Acropolis.** Entire homes within 500 m have a pre-discount nightly rate roughly **77% to 92% higher** than homes 3 to 5 km away, across the adjusted specifications tested. In the main specification, which adjusts for size, property type, review scores, host behaviour and quote dates, the estimate is +91%. Most of the gradient is gone by 1.5 km (about +25%), and past 3 km distance barely matters.

How sure is that number? The 82% to 100% band on the chart only covers sampling noise *within that one regression*. Refitting with other controls and price cutoffs gives **+77% to +92%** across the adjusted specifications. The regression with no controls gives +108%, matching the raw medians. The estimate is most sensitive to the top price cutoff: excluding homes above EUR 302 (the 95th percentile of the already trimmed homes, about 6% of eligible entire homes before trimming) lowers it to +77%. So the most expensive homes near the Acropolis account for part of the gap. This is an **association, not a causal effect**. Views, renovation and amenities are not fully captured, although a rough amenities count barely moves the estimate (+92%).

**The model beats the baseline by about 30%, but not evenly.** Errors are smallest for mid-range homes (EUR 60 to 150, around 20%) and much larger at the extremes. Below EUR 60 the model over-predicts 97% of entire rental units and condos. Some of that is expected: any model pulls extreme prices toward typical values, and these listings were selected *for* having low prices. The model systematically over-predicts this segment; the cause remains unclear. Property type does not explain it.

## Two price targets

Each listing has one Airbnb quote, and the quotes cover different check-in dates (29 June 2026 to June 2027, though 92% fall in June and July 2026) and different stay lengths. The quoted per-night price can also include taxes, or an explicit discount (11.7% of listings). Most discounts are labelled "Special offer" (1,313 line items). Only 305 are stay-length discounts (long, weekly or monthly), and 98 are early-booking discounts. So two targets are modelled side by side:

- `price_list`: the **pre-discount nightly quote rate**, i.e. the quote subtotal divided by nights, before explicit discounts, taxes and fees. **Main target.**
- `price_quoted`: the per-night price Airbnb showed. Differs from the list rate for 20% of listings.

Quote conditions (lead time, quoted nights, check-in month) are included as controls, not as a research question, so the model can tell a 1-night quote for tomorrow from a 3-night quote next spring.

## Features, ground truth and validation

**Ground truth.** The target is `price_list`, the pre-discount nightly quote rate in EUR (quote subtotal divided by nights). `price_quoted` is reported alongside it as a check. Both are modelled on a log scale and converted back to euros before errors are measured.

**Features (19 numeric, 3 categorical):**

| Group | Features |
|---|---|
| Location | distance to the Acropolis (km), distance to Syntagma (km), neighbourhood |
| Size | guests (`accommodates`), bedrooms, beds, bathrooms |
| Property | room type, property type (types with fewer than 30 listings grouped as "Other") |
| Quality | review score (overall), review score (location), number of reviews, reviews in the last 12 months, reviews per month, superhost |
| Host behaviour | host's listing count, multi-listing host flag, availability over 365 days, minimum nights (capped at 30) |
| Quote conditions (controls) | lead time in days, quoted nights, check-in month |

Never used as features: the price itself, anything calculated from it (`price_quote_*`, `estimated_revenue_l365d`), the discount flag, host names or profile text, and `host_id` (used only to build the folds).

**Train and test split.** There is no single train/test split. The data is split into 5 folds with `GroupKFold`, grouped by host. Each model is trained 5 times on about 80% of listings and tested on the remaining 20%, so every listing is tested exactly once, by a model that never saw it or any other listing from the same host. Results are the mean and standard deviation across the 5 folds.

## Results

5-fold cross-validation, grouped by host, on 13,998 listings. MAE in EUR, mean ± standard deviation across folds.

| Model | `price_list` | `price_quoted` |
|---|---|---|
| Baseline: median for same neighbourhood and room type | 50.53 ± 3.39 | 49.20 ± 3.19 |
| Random Forest | 35.55 ± 2.50 | 34.77 ± 2.28 |
| LightGBM | 35.49 ± 2.55 | 34.62 ± 2.39 |

Random Forest and LightGBM are **tied**: per fold, LightGBM is between EUR 0.70 better and EUR 0.34 worse on `price_list`. The two targets are also within fold-to-fold noise of each other.

Adding `property_type` (42 types, rare ones grouped) helps only slightly: Random Forest improves in every fold, by EUR 0.04 to 0.19. LightGBM changes by between -0.32 and +0.15.

Error by price band (`price_list`, LightGBM, out-of-fold):

| List rate (EUR) | Listings | MAE (EUR) | Mean % error |
|---|---|---|---|
| under 60 | 1,223 | 22.5 | 50% |
| 60 to 100 | 5,078 | 18.1 | 23% |
| 100 to 150 | 4,241 | 24.1 | 20% |
| 150 to 250 | 2,350 | 49.0 | 26% |
| 250+ | 1,106 | 144.6 | 37% |

**Possibly underpriced homes.** Among 8,245 entire homes with 10+ reviews, 279 (3.4%) list at least 40% below the model's prediction, measured as `(predicted - asking) / predicted`. That means "well below similar listings", not proven lost revenue.

## Live estimator

[`docs/index.html`](docs/index.html) is a single static page: pick a spot on a map of Athens, describe the home, and get an estimated pre-discount nightly rate. The model runs in the browser, so there is no server and no data leaves the page.

- It uses a smaller LightGBM trained only on inputs a visitor can set (location, room type, guests, bedrooms, bathrooms, rating, superhost). Grouped 5-fold CV MAE: EUR 35.77 ± 1.44, close to the main model. The median miss is about 21% of the asking price, measured as `|predicted - asking| / asking`.
- Instead of a flat ± euro figure, the page scales the estimate by the spread of out-of-fold `asking / predicted` ratios: a typical range (25th to 75th percentile) and a wider range (10th to 90th). These ratios are pooled over all test homes, so the 50% and 80% shares hold on average, not for every kind of home. Coverage by the estimate shown to the visitor:

  | Estimated rate (EUR) | Homes | Inside typical range | Inside wider range |
  |---|---|---|---|
  | under 60 | 308 | 45% | 79% |
  | 60 to 100 | 5,700 | 53% | 83% |
  | 100 to 150 | 5,094 | 50% | 80% |
  | 150 to 250 | 2,133 | 45% | 73% |
  | 250+ | 643 | 42% | 73% |

  Ranges are close to their stated coverage in the mid-range and somewhat too narrow for higher estimates. By true asking price, coverage is much lower at the extremes (24% of homes under EUR 60 fall in the typical range), because the model pulls extreme prices toward typical values.
- Demo numbers above come from a regeneration on 7 October 2026. Fold assignment can differ slightly across platforms, so a rerun elsewhere may shift the MAE by a few cents.
- The browser predictions match Python's to within 0.001% on a held-out check.
- The map shows 500 m grid cells with at least 5 homes each. Single listings are never published (see the [data protection note](DATA_PROTECTION.md)).
- Rebuild with `python scripts/export_demo.py && python scripts/build_demo_page.py`.

## Approach

- **Data:** Inside Airbnb detailed listings for Athens, scraped 29 June 2026. Prices in EUR.
- **No leakage:** `price_quote_price_per_night`, `price_quote_total_price`, `price_quote_raw` and `estimated_revenue_l365d` are the price or calculated from it, so they are never features. Neither are the two targets or the discount flag. `check_inputs()` in `src/features.py` stops the demo training if a forbidden column is in its input list, and the tests read the actual input lists from the notebook and the demo script and fail if any of them contains a price, target, discount or ID column.
- **Log target:** errors become relative instead of being dominated by luxury listings.
- **Grouped by host:** multi-listing hosts often reuse prices, so every host sits entirely inside one fold.
- **Adjusted gradient:** log-linear regression on entire homes with distance bands plus controls. Intervals from 300 host-level bootstrap resamples. Robustness: refit with no controls, size only, the main controls, main plus amenities count, and three alternative price cutoffs. The cutoff variants use percentiles of the already trimmed homes.

## Limitations

- **Asking prices, not paid prices.** The model learns what similar hosts ask, not what guests accept.
- **One quote per listing.** Mostly near-term summer dates, so this is not a seasonal model.
- **Observed controls only.** The adjusted gradient cannot rule out unmeasured differences such as views or renovation, and the bootstrap interval does not cover the choice of specification.
- **Weak at the extremes.** Mean error is 37% above EUR 250 and 50% below EUR 60, where the model systematically over-predicts for reasons not yet identified.
- 339 of 14,337 listings removed: 192 missing a price or quote, 147 outside EUR 20 to 732 (99th percentile of the list rate).
- `instant_bookable` is blank for every listing in this scrape.
