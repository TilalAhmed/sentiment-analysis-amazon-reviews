# Sentiment Analysis — Amazon Reviews 💬

A natural language processing project that classifies Amazon product reviews as positive, negative, or neutral.

---

## Overview

Performed sentiment analysis on Amazon product reviews using three different NLP 
libraries and compared their results. Achieved approximately 85% accuracy.

---

## Approach

- Preprocessed review text using NLTK
- Applied three sentiment scoring methods and compared results
- Classified reviews as Positive, Negative, or Neutral

---

## Libraries Used

| Library | Approach |
|---------|----------|
| NLTK | Text preprocessing and tokenization |
| TextBlob | Pattern-based sentiment scoring |
| VADER | Rule-based sentiment scoring (optimized for social text) |

---

## Results

- **Accuracy:** ~85%
- Best results achieved using VADER for short review text

---

## Tech Stack

- **Language:** Python
- **Libraries:** NLTK · TextBlob · VADER · Pandas · Matplotlib
- **Dataset:** Amazon Product Reviews

---

## Dataset

Uses a publicly available Amazon product reviews dataset from 
[Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) 
and place it in the project root before running.
