# make.fun — pumpfun-top5-forecast-01

Entry for the [make.fun bounty](https://make.fun/bounty/pumpfun-top5-forecast-01)
**"Predict the top 5 new Solana coins by market cap, 24 hours out"**.

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads the house's 10-minute market-cap snapshots of pump.fun coins launched
since the opening (`/input/history.ndjson` + `/input/task.json`) and writes
`/output/forecast.json`: for each of the 5 places, a probability for each
candidate coin; the top 5 in order; and per-place market-cap bin probabilities.

## Method

1. **Per-coin forecast.** For every coin ever seen, build its log10 market-cap
   (in SOL) series and fit a time-decayed log-linear trend (recent slope),
   with age, venue (graduated pool vs bonding curve), peak/drawdown and
   volatility. Forecast the log cap at the mean answer-read horizon
   (read from `task.json`, so the same code adapts from the 0.5 h dev
   rehearsal to the ~10 h final): the trend persists ~1 h, then the coin
   mean-reverts toward a low "graveyard" level — pool coins revert slower
   than curve coins, survivors slower than newborns. Predictive spread
   grows with horizon, volatility, youth, curve venue and staleness.
2. **Monte Carlo ranking.** Deterministic (fixed-seed inverse-CDF draws)
   simulation over the top 100 candidates by upper quantile: draw every
   coin's log cap, rank, and tally which coin holds each place and which
   size bin each place's cap falls in.
3. **Honest, newcomer-aware probabilities.** An explicit unnamed share per
   place (growing with horizon) is left for coins born after the cut and
   long-tail movers; only coins whose probability beats what they would get
   left unnamed are named. Size bins mix the simulated distribution with a
   broad newcomer distribution centred near the place-1 level (top-5 caps
   bunch near the top). Every probability is floored at 1e-6; per-place
   sums stay ≤ 1.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~4 s on a final-scale history (67 marks), well inside the
300 s / 1 CPU / 1024 MB limits. Two runs on the same history write the
same `forecast.json`.

## Local verification

On the kit's dev data the program runs and writes a valid `forecast.json`
(checked: exactly the keys `probabilities`/`top5`/`sizes`, per-place sums
≤ 1, 5 distinct top-5 mints). In truncated-history backtests against later
marks as answers it beat the house baseline in all four cuts
(cut 20:10: −45.9 vs −59.2; 20:20: −48.2 vs −50.0; 20:30: −31.0 vs −33.5;
20:40: −26.4 vs −29.9). `grade-local` cannot score this kit locally —
its scorer/reference are server-only ("graded on the server only").

## License

MIT — see `LICENSE`.
