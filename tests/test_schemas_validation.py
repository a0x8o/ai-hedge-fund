import pytest
from pydantic import ValidationError

from app.backend.models.schemas import BacktestRequest, HedgeFundRequest, BaseHedgeFundRequest


def _base_kwargs(**overrides):
    """Return minimal valid kwargs for BaseHedgeFundRequest-derived models."""
    defaults = {
        "tickers": ["AAPL"],
        "graph_nodes": [],
        "graph_edges": [],
    }
    defaults.update(overrides)
    return defaults


def _backtest_kwargs(**overrides):
    """Return minimal valid kwargs for BacktestRequest."""
    defaults = _base_kwargs(start_date="2024-01-01", end_date="2024-06-01")
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Ticker validation
# ---------------------------------------------------------------------------

class TestTickerValidation:
    def test_valid_tickers_pass(self):
        req = BacktestRequest(**_backtest_kwargs(tickers=["AAPL", "MSFT"]))
        assert req.tickers == ["AAPL", "MSFT"]

    def test_empty_tickers_raises(self):
        with pytest.raises(ValidationError, match="At least one ticker"):
            BacktestRequest(**_backtest_kwargs(tickers=[]))

    def test_invalid_ticker_special_chars_raises(self):
        with pytest.raises(ValidationError, match="Invalid ticker format"):
            BacktestRequest(**_backtest_kwargs(tickers=["A!@#"]))

    def test_invalid_ticker_too_long_raises(self):
        with pytest.raises(ValidationError, match="Invalid ticker format"):
            BacktestRequest(**_backtest_kwargs(tickers=["toolongticker123"]))

    def test_duplicate_tickers_raises(self):
        with pytest.raises(ValidationError, match="Duplicate ticker"):
            BacktestRequest(**_backtest_kwargs(tickers=["AAPL", "AAPL"]))

    def test_tickers_are_uppercased(self):
        req = BacktestRequest(**_backtest_kwargs(tickers=["aapl", "msft"]))
        assert req.tickers == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

class TestDateValidation:
    def test_valid_date_format_passes(self):
        req = BacktestRequest(**_backtest_kwargs(start_date="2024-01-15", end_date="2024-06-30"))
        assert req.start_date == "2024-01-15"
        assert req.end_date == "2024-06-30"

    def test_invalid_date_format_raises(self):
        with pytest.raises(ValidationError, match="Invalid date format"):
            BacktestRequest(**_backtest_kwargs(start_date="2024-13-45"))

    def test_invalid_date_non_date_string_raises(self):
        with pytest.raises(ValidationError, match="Invalid date format"):
            BacktestRequest(**_backtest_kwargs(end_date="not-a-date"))


# ---------------------------------------------------------------------------
# margin_requirement bounds
# ---------------------------------------------------------------------------

class TestMarginRequirement:
    def test_negative_margin_raises(self):
        with pytest.raises(ValidationError):
            BacktestRequest(**_backtest_kwargs(margin_requirement=-0.1))

    def test_margin_above_one_raises(self):
        with pytest.raises(ValidationError):
            BacktestRequest(**_backtest_kwargs(margin_requirement=1.1))

    def test_margin_zero_and_one_pass(self):
        req0 = BacktestRequest(**_backtest_kwargs(margin_requirement=0.0))
        req1 = BacktestRequest(**_backtest_kwargs(margin_requirement=1.0))
        assert req0.margin_requirement == 0.0
        assert req1.margin_requirement == 1.0


# ---------------------------------------------------------------------------
# initial_capital (BacktestRequest) must be positive
# ---------------------------------------------------------------------------

class TestInitialCapital:
    def test_zero_initial_capital_raises(self):
        with pytest.raises(ValidationError):
            BacktestRequest(**_backtest_kwargs(initial_capital=0))

    def test_negative_initial_capital_raises(self):
        with pytest.raises(ValidationError):
            BacktestRequest(**_backtest_kwargs(initial_capital=-1000))

    def test_positive_initial_capital_passes(self):
        req = BacktestRequest(**_backtest_kwargs(initial_capital=50_000))
        assert req.initial_capital == 50_000.0


# ---------------------------------------------------------------------------
# initial_cash (HedgeFundRequest) must be positive
# ---------------------------------------------------------------------------

class TestInitialCash:
    def test_zero_initial_cash_raises(self):
        with pytest.raises(ValidationError):
            HedgeFundRequest(**_base_kwargs(initial_cash=0))

    def test_negative_initial_cash_raises(self):
        with pytest.raises(ValidationError):
            HedgeFundRequest(**_base_kwargs(initial_cash=-500))

    def test_positive_initial_cash_passes(self):
        req = HedgeFundRequest(**_base_kwargs(initial_cash=10_000))
        assert req.initial_cash == 10_000.0
