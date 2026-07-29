from pathlib import Path
import os
import re
import unicodedata
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nepali_sentiment_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nepali_sentiment_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

NEPALI_STOPWORDS = {
    "अनि",
    "अथवा",
    "अझै",
    "आज",
    "त्यसैले",
    "त्यसको",
    "त्यसमा",
    "त्यो",
    "त",
    "तर",
    "तथा",
    "तिमी",
    "तिम्रो",
    "तिनले",
    "ती",
    "थिए",
    "थियो",
    "छ",
    "छु",
    "छन्",
    "छैन",
    "जब",
    "जसले",
    "जसमा",
    "जुन",
    "जे",
    "जो",
    "न",
    "नि",
    "पनि",
    "पर्‍यो",
    "पहिले",
    "भए",
    "भएको",
    "भने",
    "भन्ने",
    "भित्र",
    "म",
    "मा",
    "मात्र",
    "मेरो",
    "यति",
    "यदि",
    "यस",
    "यसका",
    "यसको",
    "यसले",
    "यहाँ",
    "या",
    "र",
    "रही",
    "रहेका",
    "लिए",
    "लाई",
    "ले",
    "वा",
    "संग",
    "सँग",
    "सबै",
    "हो",
    "हुन",
    "हुन्छ",
    "हुन्",
}


def ensure_directories() -> None:
    for directory in [PROCESSED_DATA_PATH.parent, FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def find_column(columns, candidates):
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in NEPALI_STOPWORDS)


def load_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. "
            "Place a CSV there with text and label/sentiment columns."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    text_col = find_column(
        df.columns, ["text", "sentence", "review", "comment", "content", "tweet", "message"]
    )
    label_col = find_column(df.columns, ["label", "sentiment", "class", "target", "category"])

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not identify text and label columns automatically. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].drop_duplicates().dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"].astype(str).str.strip().ne("")]
    df["text_length"] = df["text"].astype(str).apply(len)
    df["clean_text"] = df["text"].apply(clean_text).apply(remove_stopwords)
    df = df[df["clean_text"].ne("")].copy()

    if df["label"].nunique() < 2:
        raise ValueError("The dataset must contain at least two sentiment classes.")

    return df


def save_eda_figures(df: pd.DataFrame) -> None:
    ax = df["label"].value_counts().plot(kind="bar", title="Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sentiment_distribution.png", dpi=150)
    plt.close()

    df["text_length"].hist(bins=20)
    plt.title("Text Length Distribution")
    plt.xlabel("Character count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=150)
    plt.close()


def train_and_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, object, object]:
    label_counts = df["label"].value_counts()
    stratify_labels = df["label"] if label_counts.min() >= 2 else None
    test_size = max(0.2, df["label"].nunique() / len(df))

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)[\u0900-\u097F]+",
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        preds = model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision,
                "recall": recall,
                "f1_macro": f1,
            }
        )
        print(f"\n--- {name} ---")
        print(classification_report(y_test, preds, zero_division=0))

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    preds = best_model.predict(x_test)
    labels = list(best_model.classes_)

    cm = confusion_matrix(y_test, preds, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    metadata = {
        "best_model_name": best_model_name,
        "labels": labels,
        "train_size": len(x_train_text),
        "test_size": len(x_test_text),
    }

    final_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)[\u0900-\u097F]+",
    )
    x_all = final_vectorizer.fit_transform(df["clean_text"])
    final_model = clone(best_model)
    final_model.fit(x_all, df["label"])

    return results_df, metadata, final_model, final_vectorizer


def main() -> None:
    ensure_directories()
    df = load_data()
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    save_eda_figures(df)

    results_df, metadata, best_model, vectorizer = train_and_evaluate(df)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    joblib.dump(
        {
            "model": best_model,
            "vectorizer": vectorizer,
            "best_model_name": metadata["best_model_name"],
            "metrics": results_df.to_dict(orient="records"),
            "stopwords": sorted(NEPALI_STOPWORDS),
        },
        MODELS_DIR / "final_model.pkl",
    )

    print("\nPipeline complete.")
    print(f"Rows used: {len(df)}")
    print(f"Best model: {metadata['best_model_name']}")
    print(f"Processed data: {PROCESSED_DATA_PATH}")
    print(f"Results: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Model: {MODELS_DIR / 'final_model.pkl'}")


if __name__ == "__main__":
    main()
