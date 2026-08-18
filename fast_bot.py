from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from monitor import market_data

ROOT = Path(__file__).parent
STATE_PATH = ROOT / "fast_state.json"
OUTPUT_PATH = ROOT / "docs" / "data" / "fast.json"
ALERT_PATH = ROOT / ".fast_trade_alerts.json"
STARTING_BALANCE = 1_000.0
ALLOCATION = 0.25
FEE_RATE = 0.001
SLIPPAGE = 0.0005


@dataclass(frozen=True)
class Rules:
    fast_ema: int = 12
    slow_ema: int = 48
    breakout: int = 12
    exit_window: int = 6


def signals(raw: pd.DataFrame, rules: Rules = Rules()) -> pd.DataFrame:
    x = raw.copy().sort_values("timestamp")
    current_open = pd.Timestamp.now(tz="UTC").floor("4h")
    x = x[x.timestamp < current_open].copy()
    x["fast"] = x.close.ewm(span=rules.fast_ema, adjust=False).mean()
    x["slow"] = x.close.ewm(span=rules.slow_ema, adjust=False).mean()
    x["breakout_high"] = x.high.rolling(rules.breakout).max().shift(1)
    x["exit_low"] = x.low.rolling(rules.exit_window).min().shift(1)
    trend = (x.fast > x.slow) & (x.slow > x.slow.shift(6))
    x["entry_signal"] = trend & (x.close > x.breakout_high)
    x["exit_signal"] = (x.close < x.exit_low) | (x.fast < x.slow)
    return x.dropna().reset_index(drop=True)


def simulate(x: pd.DataFrame, fee: float = FEE_RATE, slippage: float = SLIPPAGE) -> dict:
    if len(x) < 2:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "trades": 0,
                "win_rate_pct": 0.0, "trades_per_year": 0.0, "average_win_pct": 0.0,
                "average_loss_pct": 0.0, "largest_win_pct": 0.0,
                "largest_loss_pct": 0.0, "expectancy_pct": 0.0, "profit_factor": 0.0}
    cash, qty, entry_cost = STARTING_BALANCE, 0.0, 0.0
    pending = None
    trades = wins = 0
    curve = []
    trade_returns = []
    for _, row in x.iterrows():
        if pending == "SELL" and qty > 0:
            proceeds = qty * float(row.open) * (1 - slippage) * (1 - fee)
            trade_returns.append((proceeds / entry_cost - 1) * 100)
            wins += int(proceeds > entry_cost)
            cash += proceeds
            qty, pending = 0.0, None
            trades += 1
        if pending == "BUY" and qty == 0:
            price = float(row.open) * (1 + slippage)
            budget = cash * ALLOCATION
            qty = budget / (price * (1 + fee))
            entry_cost = qty * price * (1 + fee)
            cash -= entry_cost
            pending = None
        if qty > 0 and bool(row.exit_signal):
            pending = "SELL"
        elif qty == 0 and bool(row.entry_signal):
            pending = "BUY"
        curve.append(cash + qty * float(row.close))
    if qty > 0:
        proceeds = qty * float(x.iloc[-1].close) * (1 - slippage) * (1 - fee)
        trade_returns.append((proceeds / entry_cost - 1) * 100)
        wins += int(proceeds > entry_cost)
        cash += proceeds
        trades += 1
        curve[-1] = cash
    equity = pd.Series(curve)
    drawdown = equity / equity.cummax() - 1
    years = max((x.timestamp.iloc[-1] - x.timestamp.iloc[0]).days / 365.25, 1 / 12)
    winning_returns = [value for value in trade_returns if value > 0]
    losing_returns = [value for value in trade_returns if value <= 0]
    gross_profit = sum(winning_returns)
    gross_loss = abs(sum(losing_returns))
    return {"return_pct": round((cash / STARTING_BALANCE - 1) * 100, 2),
            "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
            "trades": trades, "win_rate_pct": round(wins / trades * 100, 2) if trades else 0.0,
            "trades_per_year": round(trades / years, 1),
            "average_win_pct": round(sum(winning_returns) / len(winning_returns), 2) if winning_returns else 0.0,
            "average_loss_pct": round(sum(losing_returns) / len(losing_returns), 2) if losing_returns else 0.0,
            "largest_win_pct": round(max(winning_returns), 2) if winning_returns else 0.0,
            "largest_loss_pct": round(min(losing_returns), 2) if losing_returns else 0.0,
            "expectancy_pct": round(sum(trade_returns) / trades, 2) if trades else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0.0}


def execute(state: dict, row: pd.Series, alerts: list[dict]) -> None:
    cash, qty = float(state["cash"]), float(state["quantity"])
    pending = state.get("pending_order")
    if pending == "SELL" and qty > 0:
        price = float(row.open) * (1 - SLIPPAGE)
        proceeds = qty * price * (1 - FEE_RATE)
        pnl = proceeds - float(state["entry_cost"])
        cash += proceeds
        state["trades"].append({"date": row.timestamp.strftime("%d %b %Y %H:%M UTC"),
                                "side": "SELL", "price": round(price, 2), "pnl": round(pnl, 2)})
        alerts.append({"bot": "Faster four-hour bot", "side": "SELL",
                       "date": row.timestamp.isoformat(), "price": round(price, 2),
                       "pnl": round(pnl, 2)})
        qty, pending = 0.0, None
        state["entry_price"], state["entry_cost"] = None, None
    if pending == "BUY" and qty == 0:
        price = float(row.open) * (1 + SLIPPAGE)
        budget = cash * ALLOCATION
        qty = budget / (price * (1 + FEE_RATE))
        cost = qty * price * (1 + FEE_RATE)
        cash -= cost
        state["entry_price"], state["entry_cost"] = price, cost
        state["trades"].append({"date": row.timestamp.strftime("%d %b %Y %H:%M UTC"),
                                "side": "BUY", "price": round(price, 2), "pnl": None})
        alerts.append({"bot": "Faster four-hour bot", "side": "BUY",
                       "date": row.timestamp.isoformat(), "price": round(price, 2), "pnl": None})
        pending = None
    if qty > 0 and bool(row.exit_signal):
        pending = "SELL"
    elif qty == 0 and bool(row.entry_signal):
        pending = "BUY"
    equity = cash + qty * float(row.close)
    state.update({"cash": cash, "quantity": qty, "pending_order": pending,
                  "last_processed": row.timestamp.isoformat()})
    history = state.setdefault("equity_history", [])
    history.append({"date": row.timestamp.isoformat(), "value": round(equity, 2)})
    state["equity_history"] = history[-2190:]


def main() -> None:
    raw = market_data()
    data = signals(raw)
    alerts: list[dict] = []
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {"cash": STARTING_BALANCE, "quantity": 0.0, "entry_price": None,
                 "entry_cost": None, "pending_order": None, "last_processed": None,
                 "started_at": None, "trades": [], "equity_history": []}
    if not state.get("last_processed"):
        latest = data.iloc[-1]
        state["started_at"] = datetime.now(timezone.utc).isoformat()
        state["last_processed"] = latest.timestamp.isoformat()
        state["pending_order"] = "BUY" if bool(latest.entry_signal) else None
        state["equity_history"] = [{"date": latest.timestamp.isoformat(), "value": STARTING_BALANCE}]
    else:
        last = pd.Timestamp(state["last_processed"])
        for _, row in data[data.timestamp > last].iterrows():
            execute(state, row, alerts)

    latest = data.iloc[-1]
    equity = float(state["cash"]) + float(state["quantity"]) * float(latest.close)
    position = float(state["quantity"]) * float(latest.close)
    if state.get("pending_order"):
        signal = state["pending_order"]
    elif float(state["quantity"]) > 0:
        signal = "HOLD"
    else:
        signal = "WAIT"

    full = simulate(data)
    yearly = []
    for year in sorted(data.timestamp.dt.year.unique()):
        section = data[data.timestamp.dt.year == year]
        if len(section) >= 1000:
            yearly.append({"period": str(year), **simulate(section)})
    variations = []
    for fast in (8, 12, 16):
        for slow in (40, 48, 64):
            for breakout in (8, 12, 16):
                result = simulate(signals(raw, Rules(fast, slow, breakout, max(8, breakout // 2))))
                variations.append({"settings": f"EMA {fast}/{slow}, breakout {breakout}", **result})
    cost_tests = [{"test": label, **simulate(data, fee, slip)} for label, fee, slip in
                  [("Normal", .001, .0005), ("Double costs", .002, .001),
                   ("Severe costs", .003, .002)]]
    positive_years = sum(x["return_pct"] > 0 for x in yearly)
    positive_variations = sum(x["return_pct"] > 0 for x in variations)
    checks = [full["return_pct"] > 0, full["max_drawdown_pct"] >= -25,
              12 <= full["trades_per_year"] <= 72,
              positive_years / len(yearly) >= .60 if yearly else False,
              positive_variations / len(variations) >= .70,
              cost_tests[-1]["return_pct"] > 0]
    passed = sum(checks)
    verdict = "PROMISING" if passed >= 5 else "MIXED" if passed >= 3 else "FRAGILE"
    output = {"updated": datetime.now(timezone.utc).isoformat(), "signal": signal,
              "equity": round(equity, 2), "cash": round(float(state["cash"]), 2),
              "position_value": round(position, 2), "btc_quantity": round(float(state["quantity"]), 8),
              "btc_price": round(float(latest.close), 2),
              "profit": round(equity - STARTING_BALANCE, 2),
              "return_pct": round((equity / STARTING_BALANCE - 1) * 100, 2),
              "allocation_pct": 25, "history": state.get("equity_history", []),
              "trades": state.get("trades", [])[-30:], "paper_only": True,
              "historical": full, "yearly": yearly, "variations": variations,
              "cost_tests": cost_tests, "verdict": verdict,
              "checks_passed": passed, "checks_total": len(checks),
              "positive_years_pct": round(positive_years / len(yearly) * 100, 1) if yearly else 0,
              "positive_variations_pct": round(positive_variations / len(variations) * 100, 1)}
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    ALERT_PATH.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    print(f"Fast bot: {verdict}, {full['trades_per_year']} trades/year, "
          f"return {full['return_pct']}%, drawdown {full['max_drawdown_pct']}%")


if __name__ == "__main__":
    main()
