# What sets the price of an Athens Airbnb?

[![tests](https://github.com/georgegiannakidis/athens-airbnb-pricing/actions/workflows/tests.yml/badge.svg)](https://github.com/georgegiannakidis/athens-airbnb-pricing/actions/workflows/tests.yml)

**[Try the live estimator](https://georgegiannakidis.github.io/athens-airbnb-pricing/)** · [Model card](MODEL_CARD.md) · [Data protection note](DATA_PROTECTION.md)

How much does being close to a major demand driver, the Acropolis, add to the nightly rate of an Athens Airbnb? This project measures that price gradient on about 14,000 listings and uses the same data to compare standard machine learning models.

## Scope

- **The question is about location, not season.** The interest is how asking prices change with distance from a demand driver (the Acropolis), holding size, quality and host behaviour constant. Time of year is not the subject. Quote dates are kept only as controls, so a summer quote is not mistaken for a location premium.
- **The models are deliberately standard.** The goal is a fair comparison of well-known tabular models (a median baseline, Random Forest and LightGBM) under the same leakage-safe validation. There is no deep learning and no hyperparameter search; both models use fixed, common settings.

![Price gradient by distance from the Acropolis](reports/figures/acropolis_gradient.png)

## Key findings

**A steep price gradient around the Acropolis.** Entire homes within 500 m have a pre-discount nightly rate roughly **77% to 92% higher** than homes 3 to 5 km away, across the adjusted specifications tested. In the main specification, which adjusts for size, property type, review scores, host behaviour and quote dates, the estimate is +91%. Most of the gradient is gone by 1.5 km (about +25%), and past 3 km distance barely matters.

How sure is that number? The 82% to 100% band on the chart only covers sampling noise *within that one regression*. Refitting with other controls and price cutoffs gives **+77% to +92%** across the adjusted specifications. The regression with no controls gives +108%, matching the raw medians. The estimate is most sensitive to the top price cutoff: excluding homes above EUR 302 (the 95th percentile of the already trimmed homes, about 6% of eligible entire homes before trimming) lowers it to +77%. So the most expensive homes near the Acropolis account for part of the gap. This is an **association, not a causal effect**. Views, renovation and amenities are not fully captured, although a rough amenities count barely moves the estimate (+92%).

**The model beats the baseline by about 30%, but not evenly.** Errors are smallest for mid-range homes (EUR 60 to 150, around 20%) and much larger at the extremes. Below EUR 60 the model over-predicts 97% of entire rental units and condos. Some of that is expected: any model pulls extreme prices toward typical values, and these listings were selected *for* having low prices. The model systematically over-predicts this segment; the cause remains unclear. Property type does not explain it.

## Two price targets

Each listing has one Airbnb quote, and the quotes cover different check-in dates (29 June 2026 to June 2027, though 92% fall in June and July 2026) and different stay lengths. The quoted per-night price can also include taxes, or an explicit discount (11.7% of listings). Most discounts are labelled "Special offer" (1,313 line items). Only 305 are stay-length discounts (long, weekly or monthly), and 98 are early-booking discounts. So two targets are modelled side by side:

- `price_list`: the **pre-discount nightly quote rate**, i.e. the quote subtotal divided by nights, before explicit discounts, taxes and fees. **Main target.**
- `price_quoted`: the per-night price Airbnb showed. Differs from the list rate for 20% of listings.

Quote conditions (lead time, quoted nights, check-in month) are included as controls, not as a research question, so the model can tell a 1-night quote for tomorrow from a 3-night quote next spring.

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

- It uses a smaller LightGBM trained only on inputs a visitor can set (location, room type, guests, bedrooms, bathrooms, rating, superhost). Grouped 5-fold CV MAE: EUR 35.62 ± 2.69, close to the main model. The median miss is about 20% of the asking price, so the page shows a price-scaled range (where half, and 8 in 10, of comparable homes ask) instead of a flat ± euro figure.
- The browser predictions match Python's to within 0.001% on a held-out check.
- The map shows 500 m grid cells with at least 5 homes each. Single listings are never published (see the [data protection note](DATA_PROTECTION.md)).
- Rebuild with `python scripts/export_demo.py && python scripts/build_demo_page.py`.

## Approach

- **Data:** Inside Airbnb detailed listings for Athens, scraped 29 June 2026. Prices in EUR.
- **Features:** property type, size (guests, bedrooms, bathrooms), location (distance to the Acropolis and Syntagma, neighbourhood), quality (review scores, superhost), host behaviour (portfolio size, availability, minimum nights) and quote conditions (lead time, nights, check-in month).
- **No leakage:** `price_quote_price_per_night`, `price_quote_total_price`, `price_quote_raw` and `estimated_revenue_l365d` are the price or calculated from it, so they are never features. Neither is the discount flag.
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

## Run it

```bash
git clone https://github.com/georgegiannakidis/athens-airbnb-pricing.git
cd athens-airbnb-pricing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_athens_price_model.ipynb
```

Download `data/listings.csv` first, as described in `data/README.md`.

Run the tests with `pytest`. They use synthetic data, so no download is needed.

## Structure

```
data/              listings.csv goes here (not committed)
notebooks/         the full analysis, with outputs
src/features.py    targets, features and leakage guard
tests/             unit tests, including one that fails if a price column leaks
scripts/           notebook builder, demo model export, demo page builder
docs/              the live estimator (GitHub Pages)
reports/figures/   charts used in this README
MODEL_CARD.md      intended use, limits, evaluation
DATA_PROTECTION.md how personal data in the source is handled
```

## Data source and credit

Data from [Inside Airbnb](https://insideairbnb.com/), shared under a Creative Commons Attribution 4.0 license. Check the site for current terms.

## About me

Built by George Giannakidis. Ten years in hotel distribution technology at WebHotelier, now doing an MS in Computer Science at the University of Hartford.
