"""
Sentiment Analysis of Amazon Reviews
Author: Tilal Ahmed
Tools: NLTK VADER, TextBlob, WordCloud
Classifies reviews as Positive, Negative, or Neutral
"""

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import warnings

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon", quiet=True)


# ─── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", inplace=True, axis=1)

    df = df.sort_values("wilson_lower_bound", ascending=False)
    df = df.dropna(subset=["reviewText"])

    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {list(df.columns)}\n")
    return df


# ─── 2. Preprocess Text ───────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    rt = lambda x: re.sub("[^a-zA-Z]", " ", str(x))
    df["reviewText"] = df["reviewText"].map(rt).str.lower()
    return df


# ─── 3. Sentiment Scoring ─────────────────────────────────────────────────────

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # TextBlob — polarity & subjectivity
    df[["polarity", "subjectivity"]] = df["reviewText"].apply(
        lambda text: pd.Series(TextBlob(text).sentiment)
    )

    # VADER — Positive / Negative / Neutral label
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for review in df["reviewText"]:
        score = sia.polarity_scores(review)
        if score["neg"] > score["pos"]:
            sentiments.append("Negative")
        elif score["pos"] > score["neg"]:
            sentiments.append("Positive")
        else:
            sentiments.append("Neutral")

    df["sentiment"] = sentiments
    return df


# ─── 4. Accuracy vs Star Ratings ─────────────────────────────────────────────

def calculate_accuracy(df: pd.DataFrame) -> float:
    """
    Compare VADER predictions against star ratings:
      4-5 stars → Positive
      3 stars   → Neutral
      1-2 stars → Negative
    """
    def rating_to_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    df["true_sentiment"] = df["overall"].apply(rating_to_sentiment)

    correct = (df["sentiment"] == df["true_sentiment"]).sum()
    total   = len(df)
    accuracy = correct / total * 100

    print("─── Accuracy vs Star Ratings ─────────────────")
    print(f"  Correct predictions : {correct} / {total}")
    print(f"  Accuracy            : {accuracy:.2f}%")
    print("──────────────────────────────────────────────\n")

    from sklearn.metrics import classification_report
    print("Classification Report (vs star ratings):")
    print(classification_report(df["true_sentiment"], df["sentiment"],
                                 target_names=["Negative", "Neutral", "Positive"]))
    return accuracy


# ─── 5. Summary Stats ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("─── Sentiment Distribution ───────────────────")
    counts = df["sentiment"].value_counts()
    total  = len(df)
    for label, count in counts.items():
        print(f"  {label:<10}: {count:>6}  ({count/total*100:.1f}%)")
    print("──────────────────────────────────────────────\n")

    print("─── TextBlob Polarity Stats ──────────────────")
    print(df["polarity"].describe().round(4))
    print()


# ─── 5. Visualizations ────────────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    colors = ["#2A9D8F", "#E76F51", "#E9C46A"]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 str(val), ha="center", va="bottom", fontsize=11)
    plt.title("Sentiment Distribution of Amazon Reviews", fontsize=13, pad=12)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=150)
    plt.show()
    print("Saved: sentiment_distribution.png")


def plot_wordcloud(df: pd.DataFrame, sentiment: str = "Positive"):
    text = " ".join(df[df["sentiment"] == sentiment]["reviewText"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="viridis", max_words=200).generate(text)

    plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {sentiment} Reviews", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"wordcloud_{sentiment.lower()}.png", dpi=150)
    plt.show()
    print(f"Saved: wordcloud_{sentiment.lower()}.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Amazon Reviews Sentiment Analyzer")
    parser.add_argument("--data", default="data/amazon.csv", help="Path to amazon.csv")
    parser.add_argument("--wordcloud", action="store_true",  help="Generate word clouds")
    args = parser.parse_args()

    df = load_data(args.data)
    df = preprocess(df)

    print("Running sentiment analysis...")
    df = analyze_sentiment(df)

    calculate_accuracy(df)
    print_summary(df)
    plot_sentiment_distribution(df)

    if args.wordcloud:
        plot_wordcloud(df, "Positive")
        plot_wordcloud(df, "Negative")

    df.to_csv("data/amazon_with_sentiment.csv", index=False)
    print("Saved: data/amazon_with_sentiment.csv")
