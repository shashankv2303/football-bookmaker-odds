# The Bank is Open: ML vs the Bookmaker (EPL total goals)

UE24CS352A Machine Learning mini-project, Team 6 (PES1UG24CS435, PES1UG24CS450).
Problem: *The Bank is Open: AI in Sports Gambling* (Bucquet & Sarukkai, Stanford CS229, 2018).
The paper predicts total points in NBA games and compares against sportsbook Over/Under lines.
We replicate the pipeline (Random Forest baseline, SVD collaborative filtering, neural net, LSTM) on
**English Premier League football** (free results + odds data).

## Setup
Needs Python 3.10-3.12 (PyTorch 2.2.2 has no wheels for newer Python on Intel Macs).
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/data.py        # download 11 EPL seasons (2014/15-2024/25) from football-data.co.uk -> data/matches.csv
python src/features.py    # sanity check of the feature pipeline
python tests/test_leakage.py   # proves features never see the match's own or future results
python src/train.py       # trains + tunes all models (~10 min on CPU), writes results/predictions.csv
python src/evaluate.py    # metrics table (printed + results/metrics.csv) and figures
streamlit run app/demo.py # live demo
```
`python src/train.py rf nn` trains only the named models (keys: rf svd nn lstm).
Pre-computed results are included in `results/`, so the demo and `evaluate.py` work without retraining.

## Layout
| Path | What it does |
|---|---|
| `src/data.py` | downloads and cleans the season CSVs |
| `src/features.py` | leakage-safe features: last 3/5 matches, season-to-date form, rest days, sequences for the LSTM |
| `src/models/` | `rf`, `svd`, `nn` (MLP + LSTM), shared `common.py` |
| `src/train.py` | trains everything, picks hyperparameters on the validation season |
| `src/evaluate.py` | MSE/MAE, Over/Under 2.5 accuracy, Brier score, betting ROI vs the bookmaker; figures |
| `app/demo.py` | Streamlit demo |
| `tests/test_leakage.py` | no-leakage test |
| `docs/WALKTHROUGH.md` | plain-language explanation of everything (for presentation / Q&A prep) |
| `report/` | 2-page write-up |

## Method in one paragraph
Target: total goals in a match. Split by time (train 2014/15-2022/23, validation 2023/24, test 2024/25), never randomly.
Features use only earlier matches. Bookmaker odds are never inputs; the market-average Over/Under 2.5 odds are the benchmark
(converted to an implied expected-goals value so MSE is comparable). Hyperparameters are chosen on the validation season only.

## Results (test season 2024/25, 374 matches)
See `results/metrics.csv`. Short version: no model clearly beats the bookmaker, as in the paper.
