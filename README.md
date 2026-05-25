# Sentiment Analysis of Amazon Reviews

An NLP project that analyzes Amazon product reviews and classifies them as **Positive**, **Negative**, or **Neutral** using VADER and TextBlob sentiment analysis tools.

---

## Results

| Metric | Value |
|---|---|
| Overall Accuracy | ~76% |
| Positive Reviews | 95% precision |
| Negative Reviews | 21% precision |
| Neutral Reviews | 4% precision |
| Dominant Sentiment | Positive (75% of reviews) |
| Tools Used | VADER, TextBlob |

> Accuracy measured by comparing VADER predictions against star ratings (4-5 ⭐ = Positive, 3 ⭐ = Neutral, 1-2 ⭐ = Negative) on 4,914 reviews.

---

## Dataset

Amazon product reviews dataset (`amazon.csv`) containing review text, ratings, and Wilson lower bound scores.

Place the file inside the `data/` folder:
```
sentiment-analysis/
└── data/
    └── amazon.csv
```

---

## Project Structure

```
sentiment-analysis/
├── src/
│   └── analyze.py        # Sentiment analysis script
├── notebooks/
│   └── Sentiment_Analysis_of_Amazon_Reviews_NLP.ipynb
├── data/                 # Add amazon.csv here (not tracked by git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

**1. Clone and install dependencies**
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

**2. Run sentiment analysis**
```bash
python src/analyze.py
```

**3. Generate word clouds**
```bash
python src/analyze.py --wordcloud
```

**4. Custom dataset path**
```bash
python src/analyze.py --data path/to/your/amazon.csv
```

---

## How It Works

1. **Preprocessing** — Removes special characters, lowercases text
2. **TextBlob** — Computes polarity (-1 to 1) and subjectivity scores
3. **VADER** — Assigns Positive, Negative, or Neutral label based on compound score
4. **Visualization** — Bar chart of sentiment distribution + word clouds per sentiment

---

## Tech Stack

- Python 3.x
- NLTK / VADER
- TextBlob
- WordCloud
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly

---

## Author

**Tilal Ahmed**  
BS Computer Science — Iqra University, Karachi  
[LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN) · tilalahmed956@gmail.com
