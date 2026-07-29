# Nepali Sentiment Analysis

A complete machine learning pipeline for classifying the sentiment (positive / negative
[/ neutral]) of Nepali-language text, built as a class ML project covering problem
definition, data preprocessing, EDA, model development, and evaluation.

## Project Structure

```
nepali-sentiment-analysis/
├── data/
│   ├── raw/                 # Original dataset as downloaded
│   ├── processed/           # Cleaned, normalized dataset used for modeling
│   └── data_dictionary.md   # Column descriptions and label meanings
├── notebooks/
│   └── nepali_sentiment_analysis.ipynb   # End-to-end pipeline notebook
├── src/
│   └── train.py              # Script version of the full ML pipeline
├── models/
│   └── final_model.pkl      # Saved best model + fitted TF-IDF vectorizer
├── reports/
│   ├── figures/              # Saved plots used in the technical report
│   └── technical_report.pdf  # 5-6 page write-up
├── results/
│   └── model_comparison.csv  # Metrics across all trained models
├── requirements.txt
└── README.md
```

## Dataset

Source: [Nepali Sentiment Analysis dataset, Kaggle](https://www.kaggle.com/datasets/smaheshacharya/nepali-sentiment-analysis)
or any CSV with a text column and a sentiment/label column.

This repository includes a small starter CSV at `data/raw/nepali_sentiment_raw.csv`
so the project runs immediately. For final training, replace it with a larger
downloaded dataset using the same filename.

## Setup

1. Clone or download this project folder.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Use the included starter dataset, or replace it with your full dataset at:
   ```
   data/raw/nepali_sentiment_raw.csv
   ```
   (Via Kaggle CLI, if configured:
   `kaggle datasets download -d smaheshacharya/nepali-sentiment-analysis -p data/raw --unzip`)

## How to Run

### Run the executable pipeline

```bash
python src/train.py
```

This creates:

- `data/processed/nepali_sentiment_clean.csv`
- `results/model_comparison.csv`
- `reports/figures/sentiment_distribution.png`
- `reports/figures/text_length_distribution.png`
- `reports/figures/confusion_matrix.png`
- `models/final_model.pkl`

### Run the notebook

1. Launch Jupyter:
   ```bash
   jupyter lab
   ```
2. Open `notebooks/nepali_sentiment_analysis.ipynb`.
3. Run all cells top to bottom. The notebook will:
   - Load and clean the raw data
   - Perform EDA (class balance, text length, word frequency, word clouds)
   - Preprocess text (Unicode normalization, stopword removal, TF-IDF)
   - Train and compare Naive Bayes, Logistic Regression, and Linear SVM
   - Evaluate models (accuracy, precision, recall, F1, confusion matrix)
   - Save the best model to `models/final_model.pkl`
   - Save figures to `reports/figures/` and metrics to `results/model_comparison.csv`

## Results

See `results/model_comparison.csv` for full metrics and `reports/technical_report.pdf`
for the full write-up including methodology, discussion, and conclusions.

## References

- Add dataset citation here
- Add any papers/resources referenced in preprocessing or modeling choices
