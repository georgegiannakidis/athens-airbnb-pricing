# Data protection note

A short self-assessment of how this project handles personal data, written the way I would expect an auditor to ask about it. It is not legal advice.

## Is there personal data here?

Yes. The Inside Airbnb file includes host IDs, host names, host profile text and photo links, and listing coordinates. Under the GDPR, information about an identifiable person is personal data **even when it is already public** on Airbnb. Inside Airbnb notes that "no 'private' information is being used", which is true, but public is not the same as out of scope.

Listing coordinates are already shifted by Airbnb by up to about 150 m, which lowers but does not remove the risk of identifying a home.

## What this project does with it

| Data | Used? | How |
|---|---|---|
| Host name, about text, photos, profile links | No | Never read into features or outputs |
| `host_id` | Yes, internally | Only to group validation folds, so one host's listings never sit in both train and test |
| Listing coordinates | Yes | Converted to distances; the public demo shows only aggregated grid cells, never single listings |
| Listing IDs and single-listing rows | No public use | Published outputs show only aggregates. The "possibly underpriced" result is reported as counts and medians by neighbourhood, with groups under 5 homes merged |
| Reviews file | No | Not downloaded |

## Principles applied (GDPR Article 5)

- **Purpose limitation:** statistical analysis of asking prices. No profiling of individual hosts.
- **Data minimisation:** only the listings file; personal fields are dropped or used only internally.
- **Storage limitation:** the raw CSV stays on my machine, is excluded by `.gitignore`, and is deleted when the project ends or replaced by a newer snapshot.
- **Integrity and confidentiality:** raw data is never committed to git or uploaded to the demo.

## Lawful basis and transparency

- **Lawful basis:** legitimate interests (Article 6(1)(f)): non-commercial, statistical analysis with low impact on hosts, using data they chose to publish.
- **Transparency:** hosts cannot practically be contacted individually. Article 14(5)(b) allows an exception where informing each person "would involve a disproportionate effort", in particular for statistical purposes, as long as safeguards are applied and the information is made publicly available. This note and the repository are that public information.

## History

Early private drafts of the analysis notebook displayed a few individual listings, including listing IDs. Before the repository was made public, that output was replaced with aggregates and the git history was rewritten so no earlier version contains it.

## Source terms

- Inside Airbnb data is licensed under **CC BY 4.0**; it is credited in the README.
- Inside Airbnb asks users **not to republish the data**. This repository contains no raw data, only code, aggregate results and an aggregated demo.

## If you are a host

Your data comes from Inside Airbnb, which collects it from Airbnb. This project publishes nothing at the level of a single listing. Requests about the source data belong with Inside Airbnb or Airbnb.
