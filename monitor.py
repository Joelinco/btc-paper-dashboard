from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).parent
STATE_PATH = ROOT / "paper_state.json"
DASHBOARD_PATH = ROOT / "docs" / "data" / "dashboard.json"
BASE_URL = "https://data-api.binance.vision/api/v3/klines"
STARTING_BALANCE = 1000.0
ALLOCATION = 0.25
FEE_RATE = 0.001
SLIPPAGE = 0.0005


def market_data() -> pd.DataFrame:
    cursor = int((datetime.now(timezone.utc) - timedelta(days=365 * 8 + 5)).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    while cursor < end_ms:
        query = urlencode({"symbol": "BTCUSDT", "interval": "4h", "startTime": cursor,
                           "endTime": end_ms, "limit": 1000})
        with urlopen(f"{BASE_URL}?{query}", timeout=30) as response:
            batch = json.load(response)
        if not batch:
            break
        rows.extend(batch)
        new_cursor = int(batch[-1][0]) + 1
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        time.sleep(0.05)
    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume",
                                               "close_time", "quote", "trades", "tb", "tq", "ignore"])
    frame["timestamp"] = pd.to_datetime(frame.pop("time"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col])
    return frame.drop_duplicates("timestamp").sort_values("timestamp")


def signals(raw: pd.DataFrame) -> pd.DataFrame:
    daily = raw.set_index("timestamp").resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    daily = daily[daily["timestamp"] < pd.Timestamp.now(tz="UTC").normalize()].copy()
    daily["ema50"] = daily.close.ewm(span=50, adjust=False).mean()
    daily["ema200"] = daily.close.ewm(span=200, adjust=False).mean()
    rising = daily.ema200 > daily.ema200.shift(20)
    entry = (daily.close > daily.ema200 * 1.02) & (daily.ema50 > daily.ema200) & rising
    exit_ = (daily.close < daily.ema200 * 0.98) | (daily.ema50 < daily.ema200)
    daily["entry_signal"] = entry & entry.shift(1, fill_value=False)
    daily["exit_signal"] = exit_ & exit_.shift(1, fill_value=False)
    return daily.dropna().reset_index(drop=True)


def execute(state: dict, row: pd.Series) -> None:
    cash, qty = float(state["cash"]), float(state["quantity"])
    pending = state.get("pending_order")
    if pending == "SELL" and qty > 0:
        price = float(row.open) * (1 - SLIPPAGE)
        proceeds = qty * price * (1 - FEE_RATE)
        pnl = proceeds - float(state["entry_cost"])
        cash += proceeds
        state["trades"].append({"date": row.timestamp.strftime("%d %b %Y"), "side": "SELL",
                                "price": round(price, 2), "pnl": round(pnl, 2)})
        qty, pending = 0.0, None
        state["entry_price"], state["entry_cost"] = None, None
    if pending == "BUY" and qty == 0:
        price = float(row.open) * (1 + SLIPPAGE)
        amount = cash * ALLOCATION
        qty = amount / (price * (1 + FEE_RATE))
        cost = qty * price * (1 + FEE_RATE)
        cash -= cost
        state["entry_price"], state["entry_cost"] = price, cost
        state["trades"].append({"date": row.timestamp.strftime("%d %b %Y"), "side": "BUY",
                                "price": round(price, 2), "pnl": None})
        pending = None
    if qty > 0 and bool(row.exit_signal):
        pending = "SELL"
    elif qty == 0 and bool(row.entry_signal):
        pending = "BUY"
    equity = cash + qty * float(row.close)
    state.update({"cash": cash, "quantity": qty, "pending_order": pending,
                  "last_processed": row.timestamp.isoformat()})
    history = state.setdefault("equity_history", [])
    history.append({"date": row.timestamp.strftime("%Y-%m-%d"), "value": round(equity, 2)})
    state["equity_history"] = history[-365:]


def main() -> None:
    daily = signals(market_data())
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not state.get("last_processed"):
        latest = daily.iloc[-1]
        state["started_at"] = datetime.now(timezone.utc).isoformat()
        state["last_processed"] = latest.timestamp.isoformat()
        state["pending_order"] = "BUY" if bool(latest.entry_signal) else None
        state["equity_history"] = [{"date": latest.timestamp.strftime("%Y-%m-%d"), "value": 1000.0}]
    else:
        last = pd.Timestamp(state["last_processed"])
        for _, row in daily[daily.timestamp > last].iterrows():
            execute(state, row)

    latest = daily.iloc[-1]
    equity = float(state["cash"]) + float(state["quantity"]) * float(latest.close)
    position = float(state["quantity"]) * float(latest.close)
    ema_gap = (float(latest.close) / float(latest.ema200) - 1) * 100
    history = state.get("equity_history", [])
    weekly_returns = []
    peak_equity = STARTING_BALANCE
    current_drawdown = max_drawdown = 0.0
    if history:
        history_frame = pd.DataFrame(history)
        history_frame["date"] = pd.to_datetime(history_frame["date"], utc=True)
        history_frame = history_frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
        weekly = history_frame["value"].resample("W-SUN").last().dropna()
        weekly_returns = [{"week": date.strftime("%d %b"), "return_pct": round(float(value), 2)}
                          for date, value in (weekly.pct_change() * 100).dropna().items()][-26:]
        running_peak = history_frame["value"].cummax()
        drawdowns = (history_frame["value"] / running_peak - 1) * 100
        peak_equity = float(running_peak.iloc[-1])
        current_drawdown = float(drawdowns.iloc[-1])
        max_drawdown = float(drawdowns.min())
    weekly_values = [item["return_pct"] for item in weekly_returns]
    if state.get("pending_order"):
        signal = state["pending_order"]
    elif float(state["quantity"]) > 0:
        signal = "HOLD"
    else:
        signal = "WAIT"
    dashboard = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "latest_candle": latest.timestamp.isoformat(),
        "equity": round(equity, 2), "cash": round(float(state["cash"]), 2),
        "position_value": round(position, 2), "btc_quantity": round(float(state["quantity"]), 8),
        "btc_price": round(float(latest.close), 2), "profit": round(equity - STARTING_BALANCE, 2),
        "return_pct": round((equity / STARTING_BALANCE - 1) * 100, 2),
        "signal": signal, "allocation_pct": 25, "ema50": round(float(latest.ema50), 2),
        "ema200": round(float(latest.ema200), 2), "price_vs_ema200_pct": round(ema_gap, 2),
        "history": history, "weekly_returns": weekly_returns,
        "current_week_pct": weekly_values[-1] if weekly_values else 0.0,
        "best_week_pct": max(weekly_values) if weekly_values else 0.0,
        "worst_week_pct": min(weekly_values) if weekly_values else 0.0,
        "peak_equity": round(peak_equity, 2), "current_drawdown_pct": round(current_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown, 2), "trades": state.get("trades", [])[-20:],
        "paper_only": True
    }
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"Dashboard updated: ${equity:,.2f}, signal {signal}")


if __name__ == "__main__":
    main()
