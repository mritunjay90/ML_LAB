# Data Dictionary

This project expects a CSV sentiment dataset in `data/raw/nepali_sentiment_raw.csv`.
The notebook automatically maps common source column names to the standard columns
below.

| Column | Type | Description | Notes |
|---|---|---|---|
| text | string | Nepali-language text such as a review, post, comment, or sentence | The notebook accepts source names like `text`, `sentence`, `review`, `comment`, `content`, `tweet`, or `message` |
| label | categorical | Sentiment class assigned to the text | The notebook accepts source names like `label`, `sentiment`, `class`, `target`, or `category` |
| clean_text | string | Normalized model input created by the notebook | Unicode-normalized Devanagari text with punctuation, numerals, extra spaces, and common stopwords removed |
| text_length | integer | Character count of the original text | Used for exploratory data analysis |

## Label meanings

- **Positive** — text expressing approval, satisfaction, praise, happiness, or favorable opinion.
- **Negative** — text expressing criticism, dissatisfaction, complaint, sadness, anger, or unfavorable opinion.
- **Neutral** (if present) — text that is factual, mixed, unclear, or does not strongly express positive or negative sentiment.

## Known data quality notes

- Missing `text` or `label` rows are removed during cleaning.
- Duplicate rows are removed before modeling.
- Class balance is plotted in `reports/figures/sentiment_distribution.png`.
- The current cleaning function keeps Devanagari script and removes Roman-script text, which is suitable for primarily Nepali-script datasets but may need adjustment for heavily Romanized Nepali data.
