"""Run the agents on one or more tickers and trade their decisions on a paper account.

    python paper_trade.py NVDA AAPL                # local simulated account
    python paper_trade.py NVDA --broker alpaca     # Alpaca paper account
    python paper_trade.py NVDA --rating Buy        # skip the agents, test the order path

Needs an LLM key (e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY) in .env unless
--rating is given. Alpaca needs ALPACA_API_KEY and ALPACA_SECRET_KEY from a
*paper* account. Every result is appended to paper_trades.jsonl in the results
directory. No real money is ever traded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.paper_trading import AlpacaPaperBroker, LocalPaperBroker, execute_rating


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--date", default=get_current_date(), help="analysis date, YYYY-MM-DD (default today)")
    parser.add_argument("--broker", choices=["auto", "local", "alpaca"], default="auto",
                        help="auto uses Alpaca when ALPACA_API_KEY is set, else the local account")
    parser.add_argument("--max-position-pct", type=float, default=0.10, help="largest position, share of equity")
    parser.add_argument("--starting-cash", type=float, default=100_000.0, help="local account's opening cash")
    parser.add_argument("--rating", choices=["Buy", "Overweight", "Hold", "Underweight", "Sell"],
                        help="use this rating instead of running the agents")
    parser.add_argument("--dry-run", action="store_true", help="show the orders without submitting them")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    results = Path(config["results_dir"])
    use_alpaca = args.broker == "alpaca" or (args.broker == "auto" and os.getenv("ALPACA_API_KEY"))
    broker = AlpacaPaperBroker.from_env() if use_alpaca else LocalPaperBroker(
        results / "paper_account.json", starting_cash=args.starting_cash)

    graph = None
    if args.rating is None:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        graph = TradingAgentsGraph(config=config)

    for ticker in args.tickers:
        if graph is None:
            rating = args.rating
        else:
            print(f"Running the agents on {ticker} for {args.date} ...", flush=True)
            state, rating = graph.propagate(ticker, args.date, portfolio=broker.portfolio_context())
            report = graph.save_reports(state, ticker)
            print(f"Full report from every agent: {report}")
            summary = os.getenv("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a", encoding="utf-8") as out:
                    out.write(f"## {ticker}: {rating}\n\n{state.get('final_trade_decision', '')}\n\n")
        result = execute_rating(broker, ticker, rating, args.max_position_pct, args.dry_run)
        result["date"] = args.date
        print(json.dumps(result, indent=2, default=str))
        results.mkdir(parents=True, exist_ok=True)
        with open(results / "paper_trades.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps(result, default=str) + "\n")

    cash, equity = broker.account()
    print(f"\nPaper account ({broker.name}): cash {cash:,.2f} USD, equity {equity:,.2f} USD")


if __name__ == "__main__":
    main()
