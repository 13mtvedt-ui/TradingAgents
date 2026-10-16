"""Paper trading: a rating becomes an order on a simulated account, never a live one."""

from __future__ import annotations

import pytest

from tradingagents.paper_trading import (
    AlpacaPaperBroker,
    LocalPaperBroker,
    execute_rating,
    plan_order,
)

PRICES = {"NVDA": 100.0, "AAPL": 200.0}


def broker(tmp_path, cash=100_000.0):
    return LocalPaperBroker(tmp_path / "book.json", starting_cash=cash, price_fn=PRICES.__getitem__)


@pytest.mark.unit
@pytest.mark.parametrize("rating, held, expected", [
    ("Buy", 0, ("buy", 100)),
    ("Buy", 100, None),
    ("Overweight", 0, ("buy", 50)),
    ("Overweight", 80, None),
    ("Hold", 40, None),
    ("Underweight", 80, ("sell", 40)),
    ("Sell", 80, ("sell", 80)),
    ("Sell", 0, None),
    ("REVIEW", 80, None),
])
def test_rating_sizes_the_order(rating, held, expected):
    order = plan_order("nvda", rating, price=100.0, equity=100_000.0, cash=50_000.0, current_qty=held)
    assert (None if order is None else (order.side, order.quantity)) == expected


@pytest.mark.unit
def test_buy_is_capped_by_cash():
    order = plan_order("NVDA", "Buy", price=100.0, equity=100_000.0, cash=2_550.0, current_qty=0)
    assert (order.side, order.quantity) == ("buy", 25)


@pytest.mark.unit
def test_local_round_trip_keeps_cash_and_equity_consistent(tmp_path):
    account = broker(tmp_path)
    bought = execute_rating(account, "NVDA", "Buy")
    assert bought["action"] == "submitted" and bought["quantity"] == 100
    assert account.account() == (90_000.0, 100_000.0)

    reopened = broker(tmp_path)  # the book persists between runs
    assert reopened.position_qty("NVDA") == 100
    assert reopened.portfolio_context().position_in("NVDA").quantity == 100

    sold = execute_rating(reopened, "NVDA", "Sell")
    assert sold["side"] == "sell" and sold["quantity"] == 100
    assert reopened.account() == (100_000.0, 100_000.0)
    assert reopened.position_qty("NVDA") == 0


@pytest.mark.unit
def test_dry_run_and_review_leave_the_book_alone(tmp_path):
    account = broker(tmp_path)
    assert execute_rating(account, "AAPL", "Buy", dry_run=True)["action"] == "planned"
    assert execute_rating(account, "AAPL", "REVIEW")["action"] == "none"
    assert account.account() == (100_000.0, 100_000.0)
    assert not (tmp_path / "book.json").exists()


@pytest.mark.unit
def test_alpaca_refuses_the_live_endpoint():
    with pytest.raises(ValueError, match="non-paper"):
        AlpacaPaperBroker("k", "s", base_url="https://api.alpaca.markets")


@pytest.mark.unit
def test_alpaca_submits_a_market_order_to_the_paper_endpoint():
    class FakeResponse:
        def __init__(self, data, status=200):
            self.data, self.status_code = data, status

        def json(self):
            return self.data

        def raise_for_status(self):
            pass

    class FakeSession:
        headers: dict = {}

        def __init__(self):
            self.posted = []

        def get(self, url, timeout):
            if url.endswith("/v2/account"):
                return FakeResponse({"cash": "100000", "equity": "100000"})
            if "/v2/positions/" in url:
                return FakeResponse({}, status=404)
            return FakeResponse({"trade": {"p": 100.0}})

        def post(self, url, timeout, json):
            self.posted.append((url, json))
            return FakeResponse({"status": "accepted", "id": "abc"})

    session = FakeSession()
    result = execute_rating(AlpacaPaperBroker("k", "s", session=session), "NVDA", "Buy")
    assert result["status"] == "accepted" and result["quantity"] == 100
    url, body = session.posted[0]
    assert url == "https://paper-api.alpaca.markets/v2/orders"
    assert body == {"symbol": "NVDA", "qty": "100", "side": "buy", "type": "market", "time_in_force": "day"}
