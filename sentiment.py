"""Live headline sentiment: trained on Hugging Face data, advisory only.

Model: TF-IDF + Logistic Regression, trained on `zeroshot/twitter-financial-news-sentiment`
(9,543 train / 2,388 validation rows of labeled financial tweets/headlines: 0=Bearish,
1=Bullish, 2=Neutral).

This is deliberately NOT wired into the EMA/RSI/ATR entry/exit logic in strategy.py.
This strategy's own README documents rejecting the ADX regime filter after testing
showed it made every backtest metric worse -- the standard here is "prove it helps
before adding it to the decision path." There's no way to backtest this signal at
all (the HF dataset has no dates to align to historical trading days), so folding
it into buy/sell decisions would mean shipping an untested rule, which is exactly
what this project's own methodology argues against. It's printed as advisory
context in paper_trade.py instead.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yfinance as yf
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

LABEL_NAMES = {0: "bearish", 1: "bullish", 2: "neutral"}
MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentiment_classifier.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "sentiment_metrics.json"


def train_sentiment_model() -> dict:
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    val_texts, val_labels = ds["validation"]["text"], ds["validation"]["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)),
    ])
    pipeline.fit(train_texts, train_labels)
    val_preds = pipeline.predict(val_texts)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "zeroshot/twitter-financial-news-sentiment",
        "validation_accuracy": accuracy_score(val_labels, val_preds),
        "validation_f1_macro": f1_score(val_labels, val_preds, average="macro"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_sentiment_model() -> Pipeline:
    if not MODEL_PATH.exists():
        train_sentiment_model()
    return joblib.load(MODEL_PATH)


def fetch_live_headlines(ticker: str, limit: int = 10) -> list[str]:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []
    headlines = []
    for item in news[:limit]:
        content = item.get("content", item)
        title = content.get("title") if isinstance(content, dict) else None
        if title:
            headlines.append(title)
    return headlines


def score_ticker_sentiment(ticker: str, model: Pipeline | None = None) -> dict:
    """Advisory-only sentiment read for a ticker's current headlines."""
    model = model or load_sentiment_model()
    headlines = fetch_live_headlines(ticker)
    if not headlines:
        return {"ticker": ticker, "n_headlines": 0, "sentiment_score": None, "label": "no_data"}

    probs = model.predict_proba(headlines)  # columns: [bearish, bullish, neutral]
    agg_score = float((probs[:, 1] - probs[:, 0]).mean())
    if agg_score > 0.15:
        agg_label = "bullish"
    elif agg_score < -0.15:
        agg_label = "bearish"
    else:
        agg_label = "neutral"

    return {
        "ticker": ticker,
        "n_headlines": len(headlines),
        "sentiment_score": agg_score,
        "label": agg_label,
    }


if __name__ == "__main__":
    metrics = train_sentiment_model()
    print(f"Validation accuracy: {metrics['validation_accuracy']:.3f}")
    print(f"Validation F1 (macro): {metrics['validation_f1_macro']:.3f}")
