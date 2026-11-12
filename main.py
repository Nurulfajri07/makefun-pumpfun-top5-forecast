#!/usr/bin/env python3
"""make.fun pumpfun-top5-forecast-01 entry.

Reads /input/task.json + /input/history.ndjson (10-minute market-cap
snapshots of pump.fun coins launched since t0) and writes
/output/forecast.json: one forecast scored at each of the seven answer
reads (mean log score).

Method (stdlib only, fully deterministic, no network, no wall clock):
  * For every coin ever seen, build its log10(market cap in SOL) series.
  * Fit a time-decayed log-linear trend (recent slope), plus age, venue
    (pool/graduated vs bonding curve), peak/drawdown and volatility.
  * Forecast each coin's log cap at the mean answer-read horizon H
    (read from task.json, so the same code adapts from the 0.5 h dev
    rehearsal to the ~10 h final): trend persists ~1 h, then the coin
    mean-reverts toward a low "graveyard" level (pool coins revert
    slower than curve coins, survivors slower than newborns).
  * Predictive sd grows with horizon, volatility, youth, curve venue
    and staleness (coin missing from the last mark).
  * Deterministic Monte Carlo (fixed-seed inverse-CDF draws, no RNG
    state that can vary): draw every candidate coin's log cap,
    rank, and tally which coin holds each of the 5 places and which
    size bin each place's cap falls in.  Candidates are the top 100
    by upper quantile (mu + sigma).
  * Leave an explicit unnamed share per place for coins born after
    the cut / long-tail movers (grows with horizon); scale the named
    probabilities to (1 - unnamed) and mix the size bins with a broad
    newcomer size distribution with the same weight.
  * Floor every probability at 1e-6 (6-decimal flooring, as the house
    baseline does) so no named outcome can score the -9.21 floor
    merely for being rounded to zero; per-place sums stay <= 1.

Env overrides (as the starter): MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR,
else /input and /output.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunables (fixed in advance, not tuned on the practice answers) ----
TREND_TAU_H = 1.2        # trend persistence time constant (hours)
TREND_CAP = 0.5          # max |trend contribution| in dex
SLOPE_CLIP = 0.8         # dex per hour
GRAVEYARD_LOG = 1.7      # log10 SOL a dead coin reverts toward (~50 SOL)
TAU_REV_POOL_BASE = 6.0  # hours; + up to 8 h for old survivors
TAU_REV_CURVE_BASE = 3.0
TAU_REV_AGE_BONUS = 8.0
UNNAMED_BASE = 0.18      # unnamed share at horizon 0
UNNAMED_PER_H = 0.022
UNNAMED_MAX = 0.38
N_CANDIDATES = 100       # MC universe; named rows per place are cut off below
N_NAMED_MAX = 60         # max rows named at one place
SIZE_NAMED_MAX = 0.95    # leave 5% of each place's size mass unnamed
NEWCOMER_SD = 0.55       # size sd of a newcomer holding a place
NEWCOMER_DROP = 0.10     # dex a newcomer's cap falls per place below #1
                         # (top-5 caps bunch near the top: a newcomer
                         #  taking place k is nearly as big as place 1)
N_SAMPLES = 8000
SEED = 20261112
EPS = 1e-6
MICRO = 1_000_000
NEWCOMER_SD = 0.70       # size sd of a newcomer holding a place


def parse_time(s):
    """ISO-8601 UTC -> aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def log_cap_sol(market_cap_lamports):
    """log10 of market cap in SOL: log10(lamports) - 9."""
    lamports = int(market_cap_lamports)
    if lamports <= 0:
        return None
    return math.log10(lamports) - 9.0


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_norm_cdf(p):
    """Acklam's rational approximation to Phi^-1; deterministic."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1  # 82
    out_name = task.get("output", "forecast.json")

    history_until = task.get("history_until")
    t_cut = parse_time(history_until) if history_until else None
    reads = task.get("reads") or []
    read_times = [parse_time(r) for r in reads]

    # ---- parse history into per-coin series ----
    # mint -> {"times": [hours since first mark], "xs": [log cap],
    #          "created": datetime|None, "pool": bool, "mayhem": bool,
    #          "last_seen": datetime, "last_x": float}
    coins = {}
    mark_times = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                mark = json.loads(line)
            except ValueError:
                continue
            if not mark.get("ok") or not isinstance(mark.get("rows"), list):
                continue
            try:
                at = parse_time(mark["at"])
            except (KeyError, ValueError):
                continue
            mark_times.append(at)
            for row in mark["rows"]:
                try:
                    mint = row["mint"]
                    x = log_cap_sol(row["market_cap_lamports"])
                except (KeyError, ValueError, TypeError):
                    continue
                if x is None:
                    continue
                c = coins.get(mint)
                if c is None:
                    c = coins[mint] = {
                        "times": [], "xs": [], "created": None,
                        "pool": False, "mayhem": False,
                        "last_seen": at, "last_x": x,
                    }
                c["times"].append(at)
                c["xs"].append(x)
                c["last_seen"] = at
                c["last_x"] = x
                if row.get("complete") or row.get("venue") == "pool":
                    c["pool"] = True
                if row.get("mayhem"):
                    c["mayhem"] = True
                ca = row.get("created_at")
                if ca:
                    try:
                        c["created"] = parse_time(ca)
                    except ValueError:
                        pass

    if not coins:
        raise SystemExit("the history holds no coin rows")
    if t_cut is None:
        t_cut = max(mark_times) if mark_times else max(
            c["last_seen"] for c in coins.values())

    # ---- horizons (hours from cut to each answer read) ----
    if read_times:
        horizons = [(rt - t_cut).total_seconds() / 3600.0 for rt in read_times]
        horizons = [h for h in horizons if h >= -1.0]
    else:
        horizons = [10.0]
    h_mean = sum(horizons) / len(horizons)
    h_span = (max(horizons) - min(horizons)) if len(horizons) > 1 else 0.0
    h_mean = max(h_mean, 0.0)

    # ---- per-coin forecast (mu, sigma) at the mean horizon ----
    forecast = {}  # mint -> (mu, sigma)
    for mint, c in coins.items():
        times = c["times"]
        xs = c["xs"]
        x0 = c["last_x"]
        # age at cut (hours)
        age_h = 0.0
        if c["created"] is not None:
            age_h = max(0.0, (t_cut - c["created"]).total_seconds() / 3600.0)
        stale_h = max(0.0, (t_cut - c["last_seen"]).total_seconds() / 3600.0)

        # time-decayed linear trend over the last 2 h (or since birth)
        # weights exp(-(t_cut - t)/1h); OLS on (t_hours, x).
        pts = []
        for t, x in zip(times, xs):
            dt_h = (t_cut - t).total_seconds() / 3600.0
            if dt_h <= 2.0 + 1e-9:
                pts.append((dt_h, x))
        slope = 0.0
        resid_sd = 0.0
        if len(pts) >= 2:
            ws = [math.exp(-dt / 1.0) for dt, _ in pts]
            sw = sum(ws)
            mt = sum(w * dt for w, (dt, _) in zip(ws, pts)) / sw
            mx = sum(w * x for w, (_, x) in zip(ws, pts)) / sw
            num = sum(w * (dt - mt) * (x - mx)
                      for w, (dt, x) in zip(ws, pts))
            den = sum(w * (dt - mt) ** 2 for w, (dt, _) in zip(ws, pts))
            if den > 1e-9:
                slope = num / den
            # residual volatility around the fitted line
            if len(pts) >= 3:
                ss = 0.0
                for w, (dt, x) in zip(ws, pts):
                    pred = mx + slope * (dt - mt)
                    ss += w * (x - pred) ** 2
                resid_sd = math.sqrt(ss / sw)
        elif len(pts) == 1:
            # single observation: hot-coin prior (newly graduated coins
            # that just appeared near the top tend to keep rising briefly)
            slope = 0.15 if c["pool"] else 0.0
        slope = max(-SLOPE_CLIP, min(SLOPE_CLIP, slope))

        # trend contribution, saturating with horizon
        trend = slope * TREND_TAU_H * (1.0 - math.exp(-h_mean / TREND_TAU_H))
        trend = max(-TREND_CAP, min(TREND_CAP, trend))
        x_trend = x0 + trend

        # mean reversion toward the graveyard level
        if c["pool"]:
            tau_rev = TAU_REV_POOL_BASE + min(age_h, TAU_REV_AGE_BONUS)
        else:
            tau_rev = TAU_REV_CURVE_BASE + 0.5 * min(age_h, TAU_REV_AGE_BONUS)
        rev = 1.0 - math.exp(-h_mean / tau_rev)
        mu = x_trend + (GRAVEYARD_LOG - x_trend) * rev
        # staleness: a coin missing from recent marks has likely faded
        mu -= 0.20 * min(stale_h, 3.0)

        sigma = math.sqrt(
            0.25 ** 2 + (0.11 * math.sqrt(h_mean)) ** 2
            + (0.30 * resid_sd) ** 2 + (0.05 * h_span) ** 2 / 3.0)
        if not c["pool"]:
            sigma += 0.10
        if stale_h > 0.25:
            sigma += 0.20
        if len(pts) < 3:
            sigma += 0.08
        sigma = max(0.20, min(1.1, sigma))
        forecast[mint] = (mu, sigma)

    # ---- candidates: top 100 by upper quantile ----
    ranked = sorted(
        forecast.items(),
        key=lambda kv: (-(kv[1][0] + 1.0 * kv[1][1]), kv[0]),
    )
    cand = ranked[:N_CANDIDATES]
    cand_mints = [m for m, _ in cand]
    cand_mu = [forecast[m][0] for m in cand_mints]
    cand_sigma = [forecast[m][1] for m in cand_mints]
    n_cand = len(cand_mints)

    # ---- deterministic Monte Carlo over candidate log caps ----
    rng = random.Random(SEED)
    place_counts = [[0] * n_cand for _ in range(places)]
    size_counts = [[0] * n_bins for _ in range(places)]
    size_mean = [0.0] * places
    # bin index of a log cap
    def bin_of(v):
        # edges ascending; bin 0: v < edges[0]; bin i: edges[i-1] <= v < edges[i]
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if v < edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo  # 0..len(edges)

    for _ in range(N_SAMPLES):
        draws = []
        for j in range(n_cand):
            z = inv_norm_cdf(rng.random())
            draws.append(cand_mu[j] + cand_sigma[j] * z)
        order = sorted(range(n_cand), key=lambda j: (-draws[j], cand_mints[j]))
        for k in range(places):
            j = order[k]
            place_counts[k][j] += 1
            size_counts[k][bin_of(draws[j])] += 1
            size_mean[k] += draws[j] / N_SAMPLES

    unnamed = min(UNNAMED_MAX, UNNAMED_BASE + UNNAMED_PER_H * h_mean)
    named_share = 1.0 - unnamed

    # ---- assemble output ----
    # Name, at each place, only the coins whose probability beats what
    # they would get left unnamed: naming a coin also shrinks the
    # unnamed pool (its share is split over the chart's unnamed coins),
    # so a tail coin named below that value loses more than it gains.
    # Greedy in descending probability; the threshold can only rise as
    # coins are added, so stop at the first coin that fails it.
    probabilities = []
    for k in range(places):
        scored = sorted(
            ((named_share * place_counts[k][j] / N_SAMPLES, cand_mints[j])
             for j in range(n_cand) if place_counts[k][j] > 0),
            key=lambda t: -t[0])
        named = []
        used = 0.0
        for p, mint in scored:
            if len(named) >= N_NAMED_MAX:
                break
            unnamed_now = 1.0 - used  # share left if we stop here
            denom = 100 - len(named)  # chart coins left unnamed (worst case)
            if denom <= 0 or p <= unnamed_now / denom:
                break
            named.append((mint, p))
            used += p
        for mint, p in named:
            p_floor = math.floor(p * MICRO) / MICRO
            if p_floor >= EPS:
                probabilities.append({"mint": mint, "place": k + 1, "p": p_floor})

    # sizes: MC histogram mixed with the newcomer size distribution
    sizes = []
    for k in range(places):
        # A newcomer holding place k is nearly as big as the place-1
        # coin (top-5 caps bunch near the top), not as small as the
        # k-th known candidate: centre its size on the place-1 mean
        # minus a small per-place drop.
        mean_k = size_mean[0] - NEWCOMER_DROP * k
        probs = []
        for b in range(n_bins):
            p_mc = size_counts[k][b] / N_SAMPLES
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            p_new = norm_cdf((hi - mean_k) / NEWCOMER_SD) - \
                norm_cdf((lo - mean_k) / NEWCOMER_SD)
            p = named_share * p_mc + unnamed * p_new
            p_floor = math.floor(p * MICRO) / MICRO
            if p_floor >= EPS:
                probs.append((b, p_floor))
        # leave a little size mass unnamed as insurance for far-tail bins
        tot_p = sum(p for _, p in probs)
        if tot_p > SIZE_NAMED_MAX:
            scale = SIZE_NAMED_MAX / tot_p
            probs = [(b, math.floor(p * scale * MICRO) / MICRO)
                     for b, p in probs if math.floor(p * scale * MICRO) > 0]
        sizes.extend({"place": k + 1, "bin": b, "p": p} for b, p in probs)

    # top5: by place-1 probability, then by mu, then mint
    p1 = {}
    for row in probabilities:
        if row["place"] == 1:
            p1[row["mint"]] = row["p"]
    top5 = sorted(
        cand_mints,
        key=lambda m: (-p1.get(m, 0.0), -forecast[m][0], m),
    )[:places]
    # pad (only if history had fewer coins than places) with next-ranked
    if len(top5) < places:
        extra = [m for m, _ in ranked if m not in top5]
        top5 = (top5 + extra)[:places]

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
