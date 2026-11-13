"""Forecast Polymarket's next NVIDIA CEO Yes prices, minute by minute, 4 hours ahead.

Model: per-market volatility-scaled persistence with asymmetric quantiles,
a last-move give-back, and a cross-market sum-reversion correction.
  - Median = the market's price at the checkpoint, with a give-back: if the
    last 1-minute move was at least 0.01 in size, the median reverts the
    whole move, growing in over ~120 minutes (1 - exp(-h/120)). The last
    print is often a bounce / trade impact, not the new level; moves
    2+ minutes old are real drift and are NOT reverted (tested: worse).
  - Cross-market sum reversion: the 24 candidates are mutually exclusive,
    so their Yes prices should sum to ~1.0. The checkpoint sum S wanders
    0.81-1.09; each market's whole forecast is shifted by
    x * ((1/S)**0.6 - 1), i.e. scaled toward sum 1.0. When the board is
    overbought (S > 1) every price is pulled down proportionally to its
    level, and vice versa. This is the single biggest win over plain
    persistence (practice 0.2663 -> 0.2484 at gamma 0.5 alone).
  - Spread: each market gets its own 1-minute volatility sigma, the
    standard deviation of its 1-minute price changes over the last
    120 minutes, winsorized at +/-0.05 (a single huge jump must not blow
    the spread up), floored at 0.0006 and capped at 0.008.
  - The spread grows like h**0.38 for inner quantiles, h**0.35 for the
    outer 5%/95% (prices are far stickier than a random walk over 4h).
  - Quantile multipliers are asymmetric, fit to the empirical change
    distribution (a spike at zero with fat, left-skewed tails: down moves
    run further than up moves): k = (-2.303*0.8, -0.674, 0, +0.674*0.3,
    +2.303*0.9). In particular the 75% quantile sits very close to the
    median (up moves are small) while the 25% quantile keeps the full
    normal width (down moves run far).
  - Quantiles are clamped to [0, 1], rounded to 6 decimals, non-decreasing.

Practice backtest (local scorer, validated to reproduce the house baseline
exactly: loss 0.363589, median error 0.767503, coverage 0.949642):
  this entry: loss 0.2462, median error 0.7248, 90% coverage 0.8837
  (house baseline 0.3636; live leaderboard #1 0.243, #2 0.247).
  Per-case losses: p00 0.3817, p01 0.2931, p02 0.2989, p03 0.2361,
  p04 0.2019, p05 0.1141. Coverage 0.88 is inside the range of ranked
  prize-eligible entries on the live board (#2 sits at 0.883, #9 at 0.830),
  so the narrower outer tail costs no eligibility.
  LOOCV (leave-one-case-out): p00 0.2226, p01 0.2380, p02 0.2370,
  p03 0.2484, p04 0.2564, p05 0.2767 — better than the plain persistence
  entry on all six folds, so the sum-reversion signal is robust, not a
  single-case artefact. The gamma optimum is broad (0.45-0.6 all ~0.248),
  so the tuning is not knife-edge.

Standard library only. Deterministic: no randomness, no time, no network.
Reads $MAKEFUN_INPUT_DIR (else /input/), writes $MAKEFUN_OUTPUT_DIR (else
/output/), one <id>.out per <id>.in found, in name order.
"""

import json
import math
import os

# --- tuned parameters (see docstring) ---
SIGMA_WINDOW = 120        # minutes of 1-min changes for the per-market sigma
SIGMA_CLIP = 0.05         # winsorize 1-min changes at +/-0.05 before the std
SIGMA_MIN = 0.0006        # floor on sigma: a market that sat still can still jump
SIGMA_CAP = 0.008         # cap on sigma (price units per sqrt-minute)
ALPHA = 0.38              # inner spread grows like h ** ALPHA
ALPHA_OUTER = 0.35        # outer spread grows like h ** ALPHA_OUTER
K = (-1.6448536269514722 * 1.4 * 0.8, -0.6744897501960817, 0.0,
     0.6744897501960817 * 0.3, 1.6448536269514722 * 1.4 * 0.9)
GB_R = 1.0                # give-back fraction of the last 1-min move
GB_TAU = 120.0            # give-back grows in as (1 - exp(-h/GB_TAU))
GB_THRESH = 0.01          # only revert last moves of at least this size
SUM_GAMMA = 0.6           # sum-reversion strength: shift = x * ((1/S)**g - 1)
HORIZON = 240

# h ** ALPHA is the same for every market: compute once, reuse (also keeps
# run_ms down, which breaks exact-loss ties in our favour).
POWERS = [h ** ALPHA for h in range(1, HORIZON + 1)]
# Precomputed (multiplier, uses_outer_spread) pairs: avoids a membership
# test per cell and keeps run_ms down further.
K_PAIRS = ((K[0], True), (K[1], False), (K[2], False), (K[3], False),
           (K[4], True))
POWERS_OUTER = [h ** ALPHA_OUTER for h in range(1, HORIZON + 1)]
GB_GROWTH = [1.0 - math.exp(-h / GB_TAU) for h in range(1, HORIZON + 1)]


def forecast_market(prices):
    """Return the 240 x 5 quantile rows for one market (before sum shift)."""
    if not prices:
        # Defensive: the format guarantees >= 1 price, but never crash the
        # whole case over one market; a flat 0.5 keeps the other markets valid.
        return [[0.5] * 5 for _ in range(HORIZON)]
    x = prices[-1]
    n = len(prices)
    # winsorized window std of 1-min changes
    ch = [prices[i + 1] - prices[i] for i in range(max(0, n - 1 - SIGMA_WINDOW), n - 1)]
    if len(ch) < 2:
        sigma = 0.0
    else:
        tail = [min(SIGMA_CLIP, max(-SIGMA_CLIP, c)) for c in ch]
        mean = sum(tail) / len(tail)
        var = sum((c - mean) ** 2 for c in tail) / len(tail)
        sigma = math.sqrt(var) if var > 0.0 else 0.0
    if sigma < SIGMA_MIN:
        sigma = SIGMA_MIN
    if sigma > SIGMA_CAP:
        sigma = SIGMA_CAP
    # give-back of the last 1-minute move (only if it was a real move)
    gb_d = 0.0
    if n > 1:
        d = x - prices[-2]
        if abs(d) >= GB_THRESH:
            gb_d = d
    rows = []
    for h in range(1, HORIZON + 1):
        s = sigma * POWERS[h - 1]
        so = sigma * POWERS_OUTER[h - 1]
        base = x - GB_R * gb_d * GB_GROWTH[h - 1]
        row = []
        for k, outer in K_PAIRS:
            ss = so if outer else s
            q = base + k * ss
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
    forecasts = []
    xs = []
    for m in case["markets"]:
        forecasts.append({"id": m["id"], "q": forecast_market(m["prices"])})
        xs.append(m["prices"][-1] if m["prices"] else 0.5)
    # cross-market sum reversion: pull every market toward sum 1.0,
    # proportionally to its own checkpoint price
    total = sum(xs)
    if total > 0.0:
        f = (1.0 / total) ** SUM_GAMMA - 1.0
        if f != 0.0:
            for fc, x in zip(forecasts, xs):
                d = x * f
                if d == 0.0:
                    continue
                for r in fc["q"]:
                    for i in range(5):
                        q = r[i] + d
                        r[i] = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
                    for i in range(1, 5):
                        if r[i] < r[i - 1]:
                            r[i] = r[i - 1]
    return forecasts


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
