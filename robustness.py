from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from monitor import market_data

ROOT = Path(__file__).parent
OUTPUT = ROOT / "docs" / "data" / "robustness.json"
STARTING_BALANCE = 1_000.0
ALLOCATION = 0.25


@dataclass(frozen=True)
class Rules:
    fast: int = 50
    slow: int = 200
    confirmation: float = 0.02
    slope_days: int = 20


def prepare(raw: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    daily = raw.set_index("timestamp").resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    daily = daily[daily["timestamp"] < pd.Timestamp.now(tz="UTC").normalize()].copy()
    daily["fast"] = daily.close.ewm(span=rules.fast, adjust=False).mean()
    daily["slow"] = daily.close.ewm(span=rules.slow, adjust=False).mean()
    rising = daily.slow > daily.slow.shift(rules.slope_days)
    entry = ((daily.close > daily.slow * (1 + rules.confirmation))
             & (daily.fast > daily.slow) & rising)
    exit_ = ((daily.close < daily.slow * (1 - rules.confirmation))
             | (daily.fast < daily.slow))
    daily["entry"] = entry & entry.shift(1, fill_value=False)
    daily["exit"] = exit_ & exit_.shift(1, fill_value=False)
    return daily.dropna().reset_index(drop=True)


def simulate(data: pd.DataFrame, fee: float = 0.001, slippage: float = 0.0005) -> dict:
    if len(data) < 2:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "trades": 0,
                "buy_hold_pct": 0.0, "win_rate_pct": 0.0}
    cash, qty, entry_cost = STARTING_BALANCE, 0.0, 0.0
    pending = None
    trades = wins = 0
    curve = []
    for _, row in data.iterrows():
        if pending == "SELL" and qty > 0:
            proceeds = qty * float(row.open) * (1 - slippage) * (1 - fee)
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
        if qty > 0 and bool(row.exit):
            pending = "SELL"
        elif qty == 0 and bool(row.entry):
            pending = "BUY"
        curve.append(cash + qty * float(row.close))
    if qty > 0:
        proceeds = qty * float(data.iloc[-1].close) * (1 - slippage) * (1 - fee)
        wins += int(proceeds > entry_cost)
        cash += proceeds
        trades += 1
        curve[-1] = cash
    equity = pd.Series(curve)
    drawdown = equity / equity.cummax() - 1
    buy_hold = (float(data.iloc[-1].close) / float(data.iloc[0].close) - 1) * 100
    return {"return_pct": round((cash / STARTING_BALANCE - 1) * 100, 2),
            "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
            "trades": trades, "buy_hold_pct": round(buy_hold, 2),
            "win_rate_pct": round(wins / trades * 100, 2) if trades else 0.0}


def main() -> None:
    raw = market_data()
    base = prepare(raw, Rules())
    full = simulate(base)

    # Calendar-year tests are genuinely unseen slices and expose changing regimes.
    years = []
    for year in sorted(base.timestamp.dt.year.unique()):
        section = base[base.timestamp.dt.year == year]
        if len(section) >= 180:
            result = simulate(section)
            years.append({"period": str(year), **result})

    # Rolling 24-month tests reduce dependence on arbitrary calendar boundaries.
    rolling = []
    first = base.timestamp.min().normalize()
    last = base.timestamp.max().normalize()
    start = first
    while start + pd.DateOffset(months=24) <= last + pd.Timedelta(days=1):
        end = start + pd.DateOffset(months=24)
        section = base[(base.timestamp >= start) & (base.timestamp < end)]
        if len(section) >= 600:
            rolling.append({"period": f"{start:%b %Y}–{end:%b %Y}", **simulate(section)})
        start += pd.DateOffset(months=6)

    costs = []
    for label, fee, slip in [("Normal", .001, .0005), ("Double costs", .002, .001),
                              ("Severe costs", .003, .002)]:
        costs.append({"test": label, "fee_pct": fee * 100,
                      "slippage_pct": slip * 100, **simulate(base, fee, slip)})

    sensitivity = []
    for fast in (40, 50, 60):
        for slow in (180, 200, 220):
            for confirmation in (.01, .02, .03):
                result = simulate(prepare(raw, Rules(fast, slow, confirmation, 20)))
                sensitivity.append({"settings": f"EMA {fast}/{slow}, band {confirmation*100:.0f}%", **result})

    yearly_positive = sum(x["return_pct"] > 0 for x in years)
    rolling_positive = sum(x["return_pct"] > 0 for x in rolling)
    sensitivity_positive = sum(x["return_pct"] > 0 for x in sensitivity)
    scores = {
        "positive_years_pct": round(yearly_positive / len(years) * 100, 1) if years else 0,
        "positive_rolling_pct": round(rolling_positive / len(rolling) * 100, 1) if rolling else 0,
        "positive_variations_pct": round(sensitivity_positive / len(sensitivity) * 100, 1),
        "total_trades": full["trades"]
    }
    passes = [scores["positive_years_pct"] >= 60, scores["positive_rolling_pct"] >= 60,
              scores["positive_variations_pct"] >= 70, costs[-1]["return_pct"] > 0,
              full["max_drawdown_pct"] >= -30, full["trades"] >= 10]
    passed = sum(passes)
    verdict = "PROMISING" if passed >= 5 else "MIXED" if passed >= 3 else "FRAGILE"
    report = {"updated": datetime.now(timezone.utc).isoformat(), "verdict": verdict,
              "checks_passed": passed, "checks_total": len(passes), "full_history": full,
              "scores": scores, "yearly": years, "rolling": rolling, "costs": costs,
              "sensitivity": sensitivity,
              "note": "Historical robustness reduces uncertainty but cannot reproduce live execution or future markets."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Robustness report: {verdict} ({passed}/{len(passes)} checks passed)")


if __name__ == "__main__":
    main()
