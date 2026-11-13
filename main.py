"""Forecast Polymarket's next NVIDIA CEO Yes prices, minute by minute, 4 hours ahead.

Model: per-market volatility-scaled persistence with fat outer tails.
  - Median = the market's price at the checkpoint. The 1-minute changes show
    ~zero autocorrelation and ~zero mean drift, and both a drift term and
    mean-reversion toward the long-run mean were tested on the practice cases
    and made the median worse: the price is (locally) a martingale.
  - Spread: each market gets its own 1-minute volatility sigma, the standard
    deviation of its 1-minute price changes over the last 120 minutes,
    capped at 0.008 (a longer window is poisoned by rare one-off repricing
    jumps; the cap stops a jump inside the window from blowing the spread up).
  - The spread grows like h**0.35, not sqrt(h): over 4 hours the prices are
    far stickier than a random walk (mean-reverting / bounded), so a
    square-root spread is far too wide at long horizons.
  - Quantile multipliers: normal z-scores for the inner quantiles, with the
    outer (5% / 95%) multipliers widened x1.4: the change distribution is a
    spike at zero with fat tails (occasional 0.05-0.25 jumps), so the tails
    need more room than a Gaussian while the core stays tight.
  - No sum-to-one pull: forcing the medians to sum to 1.0 made the practice
    median error worse (the true sum wanders 0.81-1.09); the checkpoint prices
    already carry the cross-market information.
  - Quantiles are clamped to [0, 1], rounded to 6 decimals, non-decreasing.

Practice backtest (local scorer, validated to reproduce the house baseline
exactly: loss 0.363589, median error 0.767503, coverage 0.949642):
  this entry: loss 0.2707, median error 0.7675, 90% coverage 0.7814
  (house baseline 0.3636; best leaderboard title claims 0.266).

Standard library only. Deterministic: no randomness, no time, no network.
Reads $MAKEFUN_INPUT_DIR (else /input/), writes $MAKEFUN_OUTPUT_DIR (else
/output/), one <id>.out per <id>.in found, in name order.
"""

import json
import math
import os

# --- tuned parameters (see docstring) ---
SIGMA_WINDOW = 120        # minutes of 1-min changes for the per-market sigma
SIGMA_CAP = 0.008         # cap on sigma (price units per sqrt-minute)
ALPHA = 0.35              # spread grows like h ** ALPHA
K = (-1.6448536269514722 * 1.4, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722 * 1.4)
HORIZON = 240


def forecast_market(prices):
    """Return the 240 x 5 quantile rows for one market."""
    x = prices[-1]
    n = len(prices)
    w = min(SIGMA_WINDOW, n - 1)
    if w < 2:
        sigma = 0.0
    else:
        tail = prices[n - 1 - w:n]
        mean = sum(tail[i + 1] - tail[i] for i in range(w)) / w
        var = sum((tail[i + 1] - tail[i] - mean) ** 2 for i in range(w)) / w
        sigma = math.sqrt(var) if var > 0.0 else 0.0
    if sigma > SIGMA_CAP:
        sigma = SIGMA_CAP
    rows = []
    for h in range(1, HORIZON + 1):
        s = sigma * (h ** ALPHA)
        row = []
        for k in K:
            q = x + k * s
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            row.append(round(q, 6))
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def run_case(case):
    return [{"id": m["id"], "q": forecast_market(m["prices"])}
            for m in case["markets"]]


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = {"forecasts": run_case(case)}
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
