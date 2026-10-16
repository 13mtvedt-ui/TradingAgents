"""Turn the Portfolio Manager's rating into an order on a paper (simulated) account.

The graph ends in a rating, not a trade. This module is the one step after it:
read the paper account, size an order from the rating with fixed rules, and
submit it. Nothing here can reach a real-money account. The local broker keeps
its book in a JSON file; the Alpaca broker only ever talks to Alpaca's paper
endpoint and refuses any other base URL.

Sizing rules (long only, whole shares, ``max_position_pct`` of equity per name):

- Buy: hold the full target, ``max_position_pct`` of equity
- Overweight: hold at least half the target; an existing larger position stays
- Hold: no trade
- Underweight: sell half the current position
- Sell: close the position
- REVIEW (no readable rating): no trade
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import requests

from tradingagents.agents.rating import is_review
from tradingagents.portfolio import PortfolioContext, Position

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


@dataclass
class Order:
    ticker: str
    side: str  # "buy" or "sell"
    quantity: int
    reason: str


def plan_order(
    ticker: str,
    rating: str,
    price: float,
    equity: float,
    cash: float,
    current_qty: float,
    max_position_pct: float = 0.10,
) -> Order | None:
    """The order a rating calls for, or ``None`` when it calls for no trade."""
    if is_review(rating) or rating == "Hold":
        return None
    if price <= 0:
        raise ValueError(f"price for {ticker} must be positive, got {price}")
    held = max(current_qty, 0)
    full = equity * max_position_pct / price

    if rating == "Buy":
        target = full
    elif rating == "Overweight":
        target = max(held, full / 2)
    elif rating == "Underweight":
        target = held / 2
    elif rating == "Sell":
        target = 0
    else:
        raise ValueError(f"unknown rating {rating!r}")

    delta = target - held
    if delta > 0:
        quantity = min(math.floor(delta), math.floor(max(cash, 0) / price))
        side = "buy"
    else:
        quantity = math.ceil(-delta) if target == 0 else math.floor(-delta)
        side = "sell"
    if quantity <= 0:
        return None
    return Order(ticker.upper(), side, int(quantity), f"{rating}: target {math.floor(target)} shares")


class PaperBroker(Protocol):
    name: str

    def account(self) -> tuple[float, float]:
        """(cash, equity)."""

    def position_qty(self, ticker: str) -> float: ...

    def price(self, ticker: str) -> float: ...

    def submit(self, order: Order) -> dict: ...

    def portfolio_context(self) -> PortfolioContext: ...


def yahoo_last_price(ticker: str) -> float:
    """The latest close, read through the data layer's cached Yahoo loader."""
    from tradingagents.dataflows.date_window import get_current_date
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    closes = load_ohlcv(ticker, get_current_date(), fill_gaps=False)["Close"].dropna()
    if closes.empty:
        raise RuntimeError(f"no recent price for {ticker}")
    return float(closes.iloc[-1])


class LocalPaperBroker:
    """A simulated account kept in a JSON file, filled at the last close."""

    name = "local"

    def __init__(self, path: str | Path, starting_cash: float = 100_000.0,
                 price_fn: Callable[[str], float] = yahoo_last_price):
        self.path = Path(path)
        self.price_fn = price_fn
        if self.path.exists():
            self.book = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.book = {"cash": starting_cash, "starting_cash": starting_cash, "positions": {}, "trades": []}
        self._prices: dict[str, float] = {}

    def price(self, ticker: str) -> float:
        symbol = ticker.upper()
        if symbol not in self._prices:
            self._prices[symbol] = self.price_fn(symbol)
        return self._prices[symbol]

    def position_qty(self, ticker: str) -> float:
        return self.book["positions"].get(ticker.upper(), {}).get("quantity", 0)

    def account(self) -> tuple[float, float]:
        value = sum(p["quantity"] * self.price(t) for t, p in self.book["positions"].items())
        return self.book["cash"], self.book["cash"] + value

    def submit(self, order: Order) -> dict:
        price = self.price(order.ticker)
        positions = self.book["positions"]
        held = positions.get(order.ticker, {"quantity": 0, "average_price": 0.0})
        if order.side == "buy":
            cost = order.quantity * price
            if cost > self.book["cash"] + 1e-9:
                raise ValueError(f"not enough paper cash for {order.quantity} {order.ticker}")
            qty = held["quantity"] + order.quantity
            held = {"quantity": qty,
                    "average_price": (held["quantity"] * held["average_price"] + cost) / qty}
            self.book["cash"] -= cost
        else:
            if order.quantity > held["quantity"]:
                raise ValueError(f"cannot sell {order.quantity} {order.ticker}, holding {held['quantity']}")
            held = {**held, "quantity": held["quantity"] - order.quantity}
            self.book["cash"] += order.quantity * price
        if held["quantity"]:
            positions[order.ticker] = held
        else:
            positions.pop(order.ticker, None)
        fill = {**asdict(order), "price": price, "time": datetime.now(UTC).isoformat()}
        self.book["trades"].append(fill)
        self.save()
        return {"status": "filled", **fill}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.book, indent=2), encoding="utf-8")

    def portfolio_context(self) -> PortfolioContext:
        return PortfolioContext(
            cash=self.book["cash"], currency="USD",
            positions=[Position(ticker=t, quantity=p["quantity"], average_price=p["average_price"])
                       for t, p in self.book["positions"].items()],
        )


class AlpacaPaperBroker:
    """Alpaca's paper-trading account. Never the live endpoint."""

    name = "alpaca"

    def __init__(self, key_id: str, secret_key: str, base_url: str = ALPACA_PAPER_URL,
                 session: requests.Session | None = None):
        if base_url.rstrip("/") != ALPACA_PAPER_URL:
            raise ValueError(f"refusing non-paper Alpaca endpoint {base_url!r}; only {ALPACA_PAPER_URL} is allowed")
        self.base_url = ALPACA_PAPER_URL
        self.http = session or requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    @classmethod
    def from_env(cls) -> AlpacaPaperBroker:
        key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError("set ALPACA_API_KEY and ALPACA_SECRET_KEY to use the Alpaca paper account")
        return cls(key, secret, os.getenv("ALPACA_BASE_URL") or ALPACA_PAPER_URL)

    def _get(self, url: str) -> requests.Response:
        response = self.http.get(url, timeout=30)
        response.raise_for_status()
        return response

    def account(self) -> tuple[float, float]:
        data = self._get(f"{self.base_url}/v2/account").json()
        return float(data["cash"]), float(data["equity"])

    def position_qty(self, ticker: str) -> float:
        response = self.http.get(f"{self.base_url}/v2/positions/{ticker.upper()}", timeout=30)
        if response.status_code == 404:
            return 0
        response.raise_for_status()
        return float(response.json()["qty"])

    def price(self, ticker: str) -> float:
        data = self._get(f"{ALPACA_DATA_URL}/v2/stocks/{ticker.upper()}/trades/latest").json()
        return float(data["trade"]["p"])

    def submit(self, order: Order) -> dict:
        response = self.http.post(f"{self.base_url}/v2/orders", timeout=30, json={
            "symbol": order.ticker, "qty": str(order.quantity), "side": order.side,
            "type": "market", "time_in_force": "day",
        })
        response.raise_for_status()
        data = response.json()
        return {"status": data.get("status"), "id": data.get("id"), **asdict(order)}

    def portfolio_context(self) -> PortfolioContext:
        cash, _ = self.account()
        rows = self._get(f"{self.base_url}/v2/positions").json()
        return PortfolioContext(
            cash=cash, currency="USD",
            positions=[Position(ticker=r["symbol"], quantity=float(r["qty"]),
                                average_price=float(r["avg_entry_price"])) for r in rows],
        )


def execute_rating(broker: PaperBroker, ticker: str, rating: str,
                   max_position_pct: float = 0.10, dry_run: bool = False) -> dict:
    """Size and (unless ``dry_run``) submit the order for one rating; return what happened."""
    record = {"ticker": ticker.upper(), "rating": rating, "broker": broker.name}
    if is_review(rating) or rating == "Hold":
        return {**record, "action": "none"}
    cash, equity = broker.account()
    price = broker.price(ticker)
    order = plan_order(ticker, rating, price, equity, cash, broker.position_qty(ticker), max_position_pct)
    record.update(price=price, cash=cash, equity=equity)
    if order is None:
        return {**record, "action": "none"}
    if dry_run:
        return {**record, "action": "planned", **asdict(order)}
    return {**record, "action": "submitted", **broker.submit(order)}
