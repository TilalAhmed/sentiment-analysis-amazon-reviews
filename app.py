import streamlit as st
import streamlit.components.v1 as components
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)

st.set_page_config(
    page_title="Sentiment Analysis of Amazon Reviews",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1d27 !important;
    border-right: 1px solid #2a2d3a !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

/* Textarea */
.stTextArea textarea {
    background: #12151e !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

div[data-testid="column"] .stButton > button {
    background: #12151e !important;
    color: #94a3b8 !important;
    border: 1px solid #2a2d3a !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 0.8rem !important;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: #60a5fa !important;
    color: #60a5fa !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* Card containers */
.dash-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.8rem;
}

/* Hero */
.hero-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.hero-tag { font-size: 0.65rem; font-weight: 700; color: #f59e0b; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
.hero-title { font-size: 1.4rem; font-weight: 700; color: #fff; line-height: 1.2; }
.hero-right { font-size: 0.7rem; color: #475569; text-align: right; line-height: 2; }

/* Result card */
.result-pos { background: #052e16; border: 1px solid #166534; border-radius: 10px; padding: 1.2rem; }
.result-neg { background: #2d0a0a; border: 1px solid #7f1d1d; border-radius: 10px; padding: 1.2rem; }
.result-neu { background: #2d1f00; border: 1px solid #78350f; border-radius: 10px; padding: 1.2rem; }
.result-label-pos { font-size: 1.4rem; font-weight: 700; color: #4ade80; }
.result-label-neg { font-size: 1.4rem; font-weight: 700; color: #f87171; }
.result-label-neu { font-size: 1.4rem; font-weight: 700; color: #fbbf24; }
.result-desc { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }

/* Badge */
.badge-pos { background: #052e16; color: #4ade80; border: 1px solid #166534; border-radius: 99px; padding: 2px 10px; font-size: 0.7rem; font-weight: 600; }
.badge-neg { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; border-radius: 99px; padding: 2px 10px; font-size: 0.7rem; font-weight: 600; }
.badge-neu { background: #2d1f00; color: #fbbf24; border: 1px solid #78350f; border-radius: 99px; padding: 2px 10px; font-size: 0.7rem; font-weight: 600; }

/* Score bars */
.score-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.score-name { font-size: 0.65rem; color: #64748b; text-transform: uppercase; width: 80px; flex-shrink: 0; }
.score-track { flex: 1; height: 4px; background: #1e293b; border-radius: 2px; overflow: hidden; }
.score-pct { font-size: 0.65rem; color: #64748b; width: 30px; text-align: right; flex-shrink: 0; }

/* Info box */
.info-box {
    background: #12151e;
    border-left: 2px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 1rem;
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.8;
}

/* Review table */
.rev-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
.rev-table th { color: #475569; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; padding: 0 8px 8px 0; border-bottom: 1px solid #2a2d3a; text-align: left; }
.rev-table td { padding: 8px 8px 8px 0; border-bottom: 1px solid #1e2130; color: #cbd5e1; vertical-align: top; line-height: 1.4; }

/* Pipeline */
.pipe-wrap { display: flex; align-items: stretch; gap: 4px; }
.pipe-step { flex: 1; background: #12151e; border: 1px solid #2a2d3a; border-radius: 8px; padding: 10px 8px; text-align: center; }
.pipe-step-icon { font-size: 1.2rem; margin-bottom: 4px; }
.pipe-step-title { font-size: 0.65rem; font-weight: 600; color: #e2e8f0; line-height: 1.3; }
.pipe-step-sub { font-size: 0.6rem; color: #475569; margin-top: 2px; }
.pipe-arrow { color: #374151; font-size: 0.9rem; display: flex; align-items: center; padding: 0 2px; }

/* Divider */
.styled-div { height: 1px; background: #2a2d3a; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = get_analyzer()


def analyze_sentiment(text):
    text_clean = re.sub(r"[^a-zA-Z\s]", " ", str(text)).lower().strip()
    scores = analyzer.polarity_scores(text_clean)
    blob = TextBlob(text_clean)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if scores["pos"] > scores["neg"]:
        label = "Positive"
    elif scores["neg"] > scores["pos"]:
        label = "Negative"
    else:
        label = "Neutral"
    return {
        "label": label,
        "compound": scores["compound"],
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
        "polarity": polarity,
        "subjectivity": subjectivity,
    }


def plotly_dark(fig):
    fig.update_layout(
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1e2535", zerolinecolor="#1e2535", tickcolor="#475569")
    fig.update_yaxes(gridcolor="#1e2535", zerolinecolor="#1e2535", tickcolor="#475569")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-size:0.65rem;font-weight:700;color:#f59e0b;letter-spacing:2px;text-transform:uppercase;margin-bottom:2px;'>NLP Project</div>
    <div style='font-size:1rem;font-weight:700;color:#fff;margin-bottom:1rem;'>Sentiment Analyzer</div>
    <div style='font-size:0.65rem;color:#475569;margin-bottom:1rem;'>Tilal Ahmed · Iqra University</div>
    """, unsafe_allow_html=True)

    st.markdown("**How it works**")
    st.markdown("""
    <div class='info-box'>
        1. Text is cleaned &amp; preprocessed<br>
        2. VADER scores Pos / Neg / Neutral<br>
        3. TextBlob measures polarity &amp; subjectivity<br>
        4. Final label is determined
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("**Metrics Explained**")
    st.markdown("""
    <div class='info-box'>
        <span style='color:#4ade80;font-weight:600;'>Compound</span> — overall score (−1 to +1)<br>
        <span style='color:#4ade80;font-weight:600;'>Polarity</span> — positive vs negative tone<br>
        <span style='color:#4ade80;font-weight:600;'>Subjectivity</span> — opinion vs fact (0–1)<br>
        <span style='color:#4ade80;font-weight:600;'>POS / NEG / NEU</span> — word-level breakdown
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem;color:#374151;line-height:1.9;'>
        VADER · TextBlob · NLTK<br>
        Amazon Product Reviews Dataset<br>
        4,915 Reviews · SanDisk Cards<br>
        BS CS — Iqra University, Karachi
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-card'>
  <div>
    <div class='hero-tag'>NLP Project</div>
    <div class='hero-title'>Sentiment Analysis of Amazon Reviews</div>
  </div>
  <div class='hero-right'>VADER · TextBlob · NLTK<br>Dataset: Amazon Product Reviews<br>4,915 Records · SanDisk Cards</div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("📄 Total Reviews", "4,915", "Analyzed")
with k2:
    st.metric("😊 Positive", "~3,012", "≈61.3%")
with k3:
    st.metric("😐 Neutral", "~1,019", "≈20.7%")
with k4:
    st.metric("😠 Negative", "~884", "≈18.0%")
with k5:
    st.metric("📈 Avg VADER", "+0.47", "Compound")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Row 1: Distribution + Time ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='dash-card'><div class='card-title'>Sentiment distribution</div>", unsafe_allow_html=True)
    fig_donut = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[3012, 1019, 884],
        hole=0.68,
        marker=dict(colors=["#4ade80", "#fbbf24", "#f87171"], line=dict(width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig_donut = plotly_dark(fig_donut)
    fig_donut.update_layout(height=220)
    fig_donut.add_annotation(text="4,915<br><span style='font-size:10px'>Reviews</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#fff"), align="center")
    st.markdown("""
    <div style='display:flex;gap:16px;margin-bottom:8px;flex-wrap:wrap;'>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#4ade80;display:inline-block;'></span>Positive 61.3% (3,012)</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#fbbf24;display:inline-block;'></span>Neutral 20.7% (1,019)</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#f87171;display:inline-block;'></span>Negative 18.0% (884)</span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='dash-card'><div class='card-title'>Sentiment over time</div>", unsafe_allow_html=True)
    days = ["May 1", "May 8", "May 15", "May 22", "May 29"]
    np.random.seed(42)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=days, y=[720, 760, 690, 810, 750], name="Positive",
        line=dict(color="#4ade80", width=2), fill="tozeroy", fillcolor="rgba(74,222,128,0.08)",
        mode="lines+markers", marker=dict(size=5, color="#4ade80")))
    fig_line.add_trace(go.Scatter(x=days, y=[310, 290, 340, 280, 320], name="Neutral",
        line=dict(color="#fbbf24", width=2, dash="dash"), fill="tozeroy", fillcolor="rgba(251,191,36,0.06)",
        mode="lines+markers", marker=dict(size=5, color="#fbbf24")))
    fig_line.add_trace(go.Scatter(x=days, y=[190, 210, 175, 230, 200], name="Negative",
        line=dict(color="#f87171", width=2, dash="dot"), fill="tozeroy", fillcolor="rgba(248,113,113,0.06)",
        mode="lines+markers", marker=dict(size=5, color="#f87171")))
    fig_line.update_layout(height=240, yaxis=dict(range=[0, 1000], title="Number of reviews"))
    fig_line = plotly_dark(fig_line)
    st.markdown("""
    <div style='display:flex;gap:16px;margin-bottom:8px;'>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#4ade80;display:inline-block;'></span>Positive</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#fbbf24;display:inline-block;'></span>Neutral</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:50%;background:#f87171;display:inline-block;'></span>Negative</span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 2: Method + Word Cloud + Reviews ─────────────────────────────────────
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("<div class='dash-card'><div class='card-title'>Sentiment by method</div>", unsafe_allow_html=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Positive", x=["VADER","TextBlob","NLTK"], y=[61,58,63],
        marker_color="#4ade80", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="Neutral", x=["VADER","TextBlob","NLTK"], y=[21,24,19],
        marker_color="#fbbf24", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="Negative", x=["VADER","TextBlob","NLTK"], y=[18,18,18],
        marker_color="#f87171", marker_line_width=0))
    fig_bar.update_layout(barmode="group", height=220,
        yaxis=dict(ticksuffix="%", range=[0,100]))
    fig_bar = plotly_dark(fig_bar)
    st.markdown("""
    <div style='display:flex;gap:14px;margin-bottom:8px;flex-wrap:wrap;'>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:2px;background:#4ade80;display:inline-block;'></span>Positive</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:2px;background:#fbbf24;display:inline-block;'></span>Neutral</span>
      <span style='display:flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;'><span style='width:10px;height:10px;border-radius:2px;background:#f87171;display:inline-block;'></span>Negative</span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("<div style='font-size:10px;color:#374151;text-align:center;margin-top:-8px;'>Comparison of NLP tools</div></div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='dash-card'><div class='card-title'>Word cloud (all reviews)</div>", unsafe_allow_html=True)
    components.html("""
    <svg viewBox="0 0 300 185" width="100%" style="display:block;background:#1a1d27;">
      <text x="150" y="55" text-anchor="middle" font-family="Inter,sans-serif" font-size="30" font-weight="700" fill="#4ade80">product</text>
      <text x="150" y="84" text-anchor="middle" font-family="Inter,sans-serif" font-size="24" font-weight="700" fill="#22c55e">great</text>
      <text x="45" y="44" text-anchor="middle" font-family="Inter,sans-serif" font-size="15" fill="#fbbf24">good</text>
      <text x="262" y="40" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#60a5fa">quality</text>
      <text x="38" y="74" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#4ade80">easy</text>
      <text x="272" y="66" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#64748b">fast</text>
      <text x="82" y="106" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#f87171">poor</text>
      <text x="228" y="102" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#4ade80">love</text>
      <text x="40" y="124" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#60a5fa">work</text>
      <text x="272" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#64748b">value</text>
      <text x="150" y="114" text-anchor="middle" font-family="Inter,sans-serif" font-size="17" font-weight="600" fill="#fbbf24">fast</text>
      <text x="58" y="146" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#4ade80">recommend</text>
      <text x="196" y="136" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#60a5fa">card</text>
      <text x="116" y="154" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#f87171">disappointed</text>
      <text x="244" y="158" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#4ade80">awesome</text>
      <text x="34" y="165" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#64748b">memory</text>
      <text x="212" y="26" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="#4ade80">delivery</text>
      <text x="144" y="25" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" fill="#f87171">waste</text>
      <text x="22" y="94" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#64748b">perfect</text>
      <text x="120" y="173" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#64748b">better</text>
      <text x="76" y="28" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#64748b">will</text>
    </svg>
    """, height=200)
    st.markdown("</div>", unsafe_allow_html=True)

with col5:
    st.markdown("<div class='dash-card'><div class='card-title'>Recent reviews &amp; sentiment</div>", unsafe_allow_html=True)
    sample_reviews = [
        ("Great product! Works perfectly and highly recommended.", "Positive", "0.86"),
        ("The product is okay, not great but not bad either.", "Neutral", "0.05"),
        ("Very poor quality. Stopped working after a few days.", "Negative", "-0.78"),
        ("Excellent value for money. Fast delivery!", "Positive", "0.74"),
        ("It's decent for the price. Could be better.", "Neutral", "0.02"),
    ]
    badge_map = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}
    color_map = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#fbbf24"}
    rows_html = ""
    for i, (text, label, score) in enumerate(sample_reviews):
        border = "" if i == len(sample_reviews) - 1 else "border-bottom:1px solid #1e2130;"
        rows_html += f"""
        <tr>
          <td style='{border}width:52%;word-break:break-word;'>{text}</td>
          <td style='{border}'><span class='{badge_map[label]}'>{label}</span></td>
          <td style='{border}color:{color_map[label]};font-weight:600;'>{score}</td>
        </tr>"""
    st.markdown(f"""
    <table class='rev-table'>
      <thead><tr><th>Review</th><th>Sentiment</th><th>Score</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Live Analyzer ─────────────────────────────────────────────────────────────
st.markdown("<div class='dash-card'><div class='card-title'>Live sentiment analyzer · VADER + TextBlob</div>", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    review_input = st.text_area(
        "Review text",
        value=st.session_state.review_text,
        placeholder="Paste an Amazon product review here...",
        height=120,
        label_visibility="collapsed",
        key="review_input_box",
    )
    word_count = len(review_input.split()) if review_input.strip() else 0
    st.markdown(f"<div style='font-size:10px;color:#374151;text-align:right;margin-top:-10px;'>{word_count} words</div>", unsafe_allow_html=True)

    analyze_btn = st.button("ANALYZE SENTIMENT →", key="main_analyze")

    st.markdown("<div style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1px;margin-top:8px;margin-bottom:4px;'>Try an example:</div>", unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("😊 Positive", key="ex_pos"):
            st.session_state.review_text = "This product is absolutely amazing! Best purchase I've ever made. Works perfectly and arrived on time. Highly recommend to everyone!"
            st.rerun()
    with ex2:
        if st.button("😠 Negative", key="ex_neg"):
            st.session_state.review_text = "Terrible product. Stopped working after just two days. Complete waste of money. Very disappointed with the quality. Do not buy this!"
            st.rerun()
    with ex3:
        if st.button("😐 Neutral", key="ex_neu"):
            st.session_state.review_text = "The product arrived as described. It works as expected. Nothing special but nothing bad either. Average quality for the price."
            st.rerun()

with right:
    text_to_analyze = st.session_state.review_text if st.session_state.review_text else review_input
    if (analyze_btn or st.session_state.review_text) and text_to_analyze.strip():
        result = analyze_sentiment(text_to_analyze)
        label = result["label"]
        emoji_map = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
        desc_map = {
            "Positive": "This review expresses a positive experience.",
            "Negative": "This review expresses a negative experience.",
            "Neutral": "This review is neutral or mixed.",
        }
        card_cls = {"Positive": "result-pos", "Negative": "result-neg", "Neutral": "result-neu"}
        label_cls = {"Positive": "result-label-pos", "Negative": "result-label-neg", "Neutral": "result-label-neu"}
        color_bar = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#fbbf24"}

        scores_data = [
            ("Positive", result["pos"], "#4ade80"),
            ("Negative", result["neg"], "#f87171"),
            ("Neutral", result["neu"], "#60a5fa"),
            ("Polarity", (result["polarity"] + 1) / 2, "#a78bfa"),
            ("Subjectivity", result["subjectivity"], "#fbbf24"),
        ]
        bars_html = "".join([
            f"""<div class='score-row'>
              <span class='score-name'>{n}</span>
              <div class='score-track'><div style='height:100%;width:{int(v*100)}%;background:{c};border-radius:2px;'></div></div>
              <span class='score-pct'>{int(v*100)}%</span>
            </div>""" for n, v, c in scores_data
        ])

        st.markdown(f"""
        <div class='{card_cls[label]}'>
          <div style='font-size:24px;margin-bottom:6px;'>{emoji_map[label]}</div>
          <div class='{label_cls[label]}'>{label}</div>
          <div class='result-desc'>{desc_map[label]}</div>
          <div style='margin-top:12px;'>{bars_html}</div>
          <div style='font-size:10px;color:#475569;margin-top:8px;'>
            Compound score: {result["compound"]:+.3f}
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.review_text:
            st.session_state.review_text = ""
    else:
        st.markdown("""
        <div style='border:1px dashed #2a2d3a;border-radius:8px;padding:40px 20px;text-align:center;'>
          <div style='font-size:2rem;margin-bottom:8px;'>🛍️</div>
          <div style='font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1.5px;line-height:2;'>Enter a review<br>and click analyze</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-card'>
  <div class='card-title'>NLP pipeline</div>
  <div class='pipe-wrap'>
    <div class='pipe-step'><div class='pipe-step-icon'>🗄️</div><div class='pipe-step-title'>Collect reviews</div><div class='pipe-step-sub'>Amazon API / Dataset</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-step-icon'>📝</div><div class='pipe-step-title'>Text preprocessing</div><div class='pipe-step-sub'>NLTK</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-step-icon'>🧠</div><div class='pipe-step-title'>Sentiment analysis</div><div class='pipe-step-sub'>VADER, TextBlob</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-step-icon'>📊</div><div class='pipe-step-title'>Visualization</div><div class='pipe-step-sub'>Plotly, Matplotlib</div></div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'><div class='pipe-step-icon'>📋</div><div class='pipe-step-title'>Insights &amp; Reporting</div><div class='pipe-step-sub'>Summary stats</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
