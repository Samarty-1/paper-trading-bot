import numpy as np

import sentiment


class _FakeModel:
    """Deterministic stand-in for the trained pipeline so aggregation/labeling
    logic can be tested without downloading the real Hugging Face dataset or
    hitting the network on every test run."""

    def __init__(self, probs_by_text):
        self._probs_by_text = probs_by_text

    def predict_proba(self, texts):
        return np.array([self._probs_by_text[t] for t in texts])


def test_score_ticker_sentiment_bullish(monkeypatch):
    headlines = ["Great quarter, stock soars"]
    probs = {headlines[0]: [0.05, 0.90, 0.05]}  # bearish, bullish, neutral
    monkeypatch.setattr(sentiment, "fetch_live_headlines", lambda ticker, limit=10: headlines)

    result = sentiment.score_ticker_sentiment("SPY", model=_FakeModel(probs))

    assert result["label"] == "bullish"
    assert result["sentiment_score"] > 0.15


def test_score_ticker_sentiment_bearish(monkeypatch):
    headlines = ["Shares tumble on guidance cut"]
    probs = {headlines[0]: [0.85, 0.05, 0.10]}
    monkeypatch.setattr(sentiment, "fetch_live_headlines", lambda ticker, limit=10: headlines)

    result = sentiment.score_ticker_sentiment("QQQ", model=_FakeModel(probs))

    assert result["label"] == "bearish"
    assert result["sentiment_score"] < -0.15


def test_score_ticker_sentiment_no_headlines(monkeypatch):
    monkeypatch.setattr(sentiment, "fetch_live_headlines", lambda ticker, limit=10: [])

    result = sentiment.score_ticker_sentiment("AAPL", model=_FakeModel({}))

    assert result["label"] == "no_data"
    assert result["sentiment_score"] is None


def test_fetch_live_headlines_parses_content_title(monkeypatch):
    class _FakeTicker:
        news = [{"content": {"title": "Headline one"}}, {"content": {}}]

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda ticker: _FakeTicker())

    assert sentiment.fetch_live_headlines("AAPL") == ["Headline one"]


def test_fetch_live_headlines_returns_empty_on_error(monkeypatch):
    class _BrokenTicker:
        @property
        def news(self):
            raise RuntimeError("network error")

    monkeypatch.setattr(sentiment.yf, "Ticker", lambda ticker: _BrokenTicker())

    assert sentiment.fetch_live_headlines("AAPL") == []
