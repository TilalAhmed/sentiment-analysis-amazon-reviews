import streamlit as st
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #f4faf6;
    color: #0a2a18;
}

#MainMenu, footer, header { visibility: hidden; }

/* Subtle grid */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,180,80,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,80,0.05) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(135deg, #00ff87 0%, #00c853 50%, #1de9b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3a7a5a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, #00ff87, #00c853, transparent);
    margin: 1.2rem 0;
}

/* Textarea */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #b8ddc8 !important;
    border-radius: 12px !important;
    color: #0a2a18 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: #00c853 !important;
    box-shadow: 0 0 0 2px rgba(0,200,83,0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #00c853, #00e676) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.result-positive {
    background: linear-gradient(135deg, #eafaf1, #d5f5e3);
    border: 1px solid #7dcea0;
}
.result-negative {
    background: linear-gradient(135deg, #fdf2f2, #fde8e8);
    border: 1px solid #e6a0a0;
}
.result-neutral {
    background: linear-gradient(135deg, #f4f4fb, #eaeaf5);
    border: 1px solid #b0b0d0;
}

.result-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}
.label-positive { color: #00a040; }
.label-negative { color: #d32f2f; }
.label-neutral  { color: #5555aa; }

.result-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #3a7a5a;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Stat boxes */
.stat-box {
    background: #ffffff;
    border: 1.5px solid #b8ddc8;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.4rem;
    color: #00a040;
}
.stat-label {
    font-size: 0.7rem;
    color: #3a7a5a;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #eaf5ef !important;
    border-right: 1px solid #b8ddc8 !important;
}

/* Score bars */
.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
}
.score-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #3a7a5a;
    text-transform: uppercase;
    width: 80px;
}
.score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00a040;
    width: 50px;
    text-align: right;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00c853, #00ff87) !important;
}

/* Info box */
.info-box {
    background: #ffffff;
    border-left: 3px solid #00c853;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #2a6045;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.7;
}

/* Example buttons */
.stButton.example > button {
    background: #ffffff !important;
    color: #00a040 !important;
    border: 1.5px solid #b8ddc8 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)


# ── Analyzer ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = get_analyzer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", ' ', str(text))
    return text.lower().strip()

def analyze_sentiment(text):
    # VADER scores
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # TextBlob
    blob = TextBlob(text)
    polarity    = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Final label
    if scores['pos'] > scores['neg']:
        label = "Positive"
    elif scores['neg'] > scores['pos']:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "compound": compound,
        "pos": scores['pos'],
        "neg": scores['neg'],
        "neu": scores['neu'],
        "polarity": polarity,
        "subjectivity": subjectivity,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-weight:700; font-size:1rem;
         color:#00a040; letter-spacing:2px; margin-bottom:0.2rem;'>
        SENTIMENT
    </div>
    <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:#3a7a5a;
         text-transform:uppercase; letter-spacing:2px; margin-bottom:1.5rem;'>
        ANALYZER · TILAL AHMED
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**How it works**")
    st.markdown("""
    <div class='info-box'>
        1. Text is cleaned &amp; preprocessed<br>
        2. VADER scores Positive / Negative / Neutral<br>
        3. TextBlob measures polarity &amp; subjectivity<br>
        4. Final label is determined
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:linear-gradient(90deg,#00c853,transparent); margin:1rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("**Metrics Explained**")
    st.markdown("""
    <div class='info-box'>
        <b style='color:#00a040'>Compound</b> — overall score (-1 to +1)<br>
        <b style='color:#00a040'>Polarity</b> — positive vs negative tone<br>
        <b style='color:#00a040'>Subjectivity</b> — opinion vs fact (0–1)<br>
        <b style='color:#00a040'>POS/NEG/NEU</b> — word-level breakdown
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:linear-gradient(90deg,#00c853,transparent); margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:#3a7a5a; line-height:1.8;'>
        VADER · TextBlob · NLTK<br>
        Amazon Reviews Dataset<br>
        NLP · Sentiment Analysis
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Sentiment<br>Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>NLP · VADER · TextBlob · Amazon Reviews</div>", unsafe_allow_html=True)
st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

# Stats
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='stat-box'><div class='stat-number'>VADER</div><div class='stat-label'>Engine</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='stat-box'><div class='stat-number'>3</div><div class='stat-label'>Sentiments</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='stat-box'><div class='stat-number'>Real-time</div><div class='stat-label'>Analysis</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='stat-box'><div class='stat-number'>Amazon</div><div class='stat-label'>Dataset</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#3a7a5a; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.5rem;'>Enter Review</div>", unsafe_allow_html=True)

    review_text = st.text_area(
        "Review",
        placeholder="Paste an Amazon product review here...",
        height=200,
        label_visibility="collapsed"
    )

    word_count = len(review_text.split()) if review_text.strip() else 0
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#3a7a5a; text-align:right; margin-top:-0.5rem;'>{word_count} words</div>",
        unsafe_allow_html=True
    )

    analyze_btn = st.button("ANALYZE SENTIMENT →")

    # Example reviews
    st.markdown("<div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#3a7a5a; text-transform:uppercase; letter-spacing:1px; margin-top:1rem; margin-bottom:0.5rem;'>Try an example:</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("😊 Positive"):
            st.session_state.example = "This product is absolutely amazing! Best purchase I've ever made. Works perfectly and arrived on time. Highly recommend to everyone!"
    with col_b:
        if st.button("😠 Negative"):
            st.session_state.example = "Terrible product. Stopped working after just two days. Complete waste of money. Very disappointed with the quality. Do not buy this!"
    with col_c:
        if st.button("😐 Neutral"):
            st.session_state.example = "The product arrived as described. It works as expected. Nothing special but nothing bad either. Average quality for the price."

    if "example" in st.session_state:
        review_text = st.session_state.example
        st.info(review_text)

with right:
    if analyze_btn or "example" in st.session_state:
        text_to_analyze = st.session_state.get("example", review_text) if "example" in st.session_state else review_text

        if not text_to_analyze.strip():
            st.warning("Please enter a review to analyze.")
        else:
            result = analyze_sentiment(text_to_analyze)
            label  = result["label"]

            # Clear example after use
            if "example" in st.session_state:
                del st.session_state["example"]

            card_class  = f"result-{'positive' if label=='Positive' else 'negative' if label=='Negative' else 'neutral'}"
            label_class = f"label-{'positive' if label=='Positive' else 'negative' if label=='Negative' else 'neutral'}"
            emoji = "😊" if label == "Positive" else "😠" if label == "Negative" else "😐"
            desc  = "This review expresses a positive experience." if label == "Positive" else \
                    "This review expresses a negative experience." if label == "Negative" else \
                    "This review is neutral or mixed."

            st.markdown(f"""
            <div class='result-card {card_class}'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>{emoji}</div>
                <div class='result-label {label_class}'>{label}</div>
                <div class='result-sub' style='margin-top:0.3rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>**Score Breakdown**", unsafe_allow_html=True)

            metrics = [
                ("Positive",     result["pos"],         "#00ff87"),
                ("Negative",     result["neg"],         "#ff4444"),
                ("Neutral",      result["neu"],         "#8888aa"),
                ("Polarity",     (result["polarity"]+1)/2, "#1de9b6"),
                ("Subjectivity", result["subjectivity"], "#ffd740"),
            ]

            for name, val, color in metrics:
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem; "
                    f"color:{color}; text-transform:uppercase; letter-spacing:1px;'>{name} — {val:.1%}</div>",
                    unsafe_allow_html=True
                )
                st.progress(min(int(val * 100), 100))

            compound = result["compound"]
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace; font-size:0.75rem; "
                f"color:#3a7a5a; margin-top:0.5rem;'>Compound Score: {compound:+.3f}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("""
        <div style='background:#ffffff; border:1.5px solid #b8ddc8; border-radius:12px;
             padding:2rem; text-align:center; margin-top:1rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🛍️</div>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.75rem;
                 color:#3a7a5a; text-transform:uppercase; letter-spacing:2px;'>
                Enter a review<br>and click analyze
            </div>
        </div>
        """, unsafe_allow_html=True)
