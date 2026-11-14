# Review Guide: "The Bank is Open" (EPL total goals vs the bookmaker)

## Start here (for Shubashitha)
1. Read this guide first, especially **Part A** (bigger picture) and **Part B** (Shubashitha's part).
2. Set up a fresh environment with Python 3.10-3.12 (Python 3.13 has no PyTorch wheels on an Intel Mac):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the leakage test and train her models:
   ```bash
   python tests/test_leakage.py
   python src/train.py rf
   ```
4. Open `src/features.py` and trace one match by hand (see "Shubashitha's quick-run checklist" in Part B).

---

Three parts: **(A) the bigger picture** (Shashank and Shubashitha must both know this), **(B) Shubashitha's part**, **(C) Shashank's part**.
Each part says what the code does, why, and the questions a panel is likely to ask.

---

## A. Bigger picture (both)

### The question
Bookmakers publish a line, e.g. "Over/Under 2.5 goals". Can a machine learning model predict a match's total goals
better than the bookmaker, well enough to bet profitably? The Stanford paper tried this on NBA points. We repeat it on
English Premier League football because free data with bookmaker odds exists for it.

### The pipeline (one line each)
```
data.py      download 11 EPL seasons                     -> data/matches.csv (4,180 matches)
features.py  turn each match into numbers about the      -> 32 features + 5-game sequences
             teams' recent form (past matches ONLY)
models/*     6 models predict total goals                -> predictions for every match
train.py     trains them, tunes on the validation season -> results/predictions.csv
evaluate.py  compares models to the bookmaker            -> metrics table + figures
demo.py      shows it live                               -> Streamlit app
```

### Four ideas that carry the whole project
1. **Time-ordered split.** Train on 2014/15-2022/23, tune on 2023/24, test once on 2024/25. A random split would let the model
   "see the future" and look better than it is.
2. **No leakage.** A match's features are built only from games played before it (`shift(1)`). `tests/test_leakage.py` proves it.
3. **Odds are the benchmark, not an input.** The bookmaker's price already contains the market's forecast; feeding it in would defeat the comparison.
4. **A fair scoreboard.** We turn the bookmaker's Over/Under 2.5 odds into an "implied expected goals" number (solving a Poisson equation),
   so bookmaker and models are compared on the same MSE scale and the same Over/Under accuracy.

### Result and what it means
| Model | Test MSE | O/U 2.5 accuracy |
|---|---|---|
| Bookmaker | 2.614 | 57.5% |
| Predict the average | 2.645 | 57.0% |
| LSTM | 2.691 | 56.4% |
| Random Forest | 2.689 | 54.5% |
| Neural net | 2.764 | 53.2% |
| SVD | 3.388 | 54.5% |

- Nobody beats the bookmaker, and the models are barely better than "always predict the average".
- That is **not a failed project**; it is the same conclusion as the paper (NN MSE 369.8 vs sportsbook 320.7, 51.5% correct side).
- Why: goals are rare and random, so form statistics carry little signal; the bookmaker also knows injuries, line-ups and market information we don't.
- 374 test matches means accuracy has about +/-2.6 points of noise, so 1-3 point gaps mean nothing.
- Betting ROI is about zero or negative, i.e. no money-making strategy.

### Differences from the paper (say these proactively)
- Football instead of basketball (check this was OK with the faculty); total goals instead of total points.
- Same four models as the paper (Random Forest, SVD, neural net, LSTM); the one thing we add is the fair bookmaker comparison (Part C).
- Our SVD predicts from a rolling window of past matches; the paper's SVD badly overfit (train MSE 2.95, test 459).

### Honest limitations (panels like these)
One league, one test season; odds are a market average; Poisson conversion assumes no over-dispersion; no injuries/line-ups/xG data;
newly promoted teams' first 5 games are dropped.

---

## B. Shubashitha's part: data, features, leakage test, RF, README

Files: `src/data.py`, `src/features.py`, `tests/test_leakage.py`, `src/models/rf.py`, `README.md`.
Theme: **"getting honest data into the models."**

### `src/data.py`
- Downloads one CSV per season from football-data.co.uk (`mmz4281/<yy><yy>/E0.csv`) into `data/raw/`, skipping files already present.
- Keeps date, teams, goals (FTHG/FTAG), shots (HS/AS), shots on target (HST/AST), corners (HC/AC) and Bet365/market-average Over/Under 2.5 odds.
- Older seasons call the market-average odds `BbAv>2.5`; the code renames them to `Avg>2.5` so all 11 seasons line up.
- Sorts by date and creates `TotalGoals = FTHG + FTAG`, the **target** we predict.
- Likely questions: *Why that website? Why these columns? What happens with missing odds?* (free, no login, results + odds together; older seasons use a different column name which we unify.)

### `src/features.py` (the most important file to understand)
- `long_format`: reshapes "one row per match" into "one row per team per match", so each team has one timeline.
- `team_history`: for each team and stat (goals for/against, shots, shots on target, corners) computes
  last-3 mean, last-5 mean and season-to-date mean. Each is computed on `shift(1)` data, so a row **never includes its own match**.
  Also `rest` = days since the team's previous match (the paper's fatigue idea).
- Home and away teams' features are joined back to the match: 16 per team x 2 = **32 tabular features**.
- `sequences`: for the LSTM, the previous 5 matches of each team in order (7 numbers per match: the 5 stats, home flag, rest).
- Rows are dropped when either team has fewer than 5 prior matches (start of data, newly promoted teams): 4,180 -> **4,031** matches.
- Split column: train <= 2022 (3,283), val 2023 (374), test 2024 (374).
- Likely questions:
  - *How do you avoid leakage?* shift(1) before rolling, plus the test.
  - *Why last 3 and 5 games?* the paper used last 3; 5 adds a smoother form signal.
  - *Why season-to-date means?* analogue of the paper's "opponent strength" features.
  - *Why not use odds as a feature?* see Part A.

### `tests/test_leakage.py`
- Adds +7 to the results/stats of match #2000 and every later match, rebuilds the features, and asserts that every match on or before that date has **identical** features.
- If any feature peeked at its own or a future result, the assertion would fail. It passes.

### `src/models/rf.py`: Random Forest (the paper's baseline)
- 300 trees averaged; grid over `max_depth` (4, 8, None) and `min_samples_leaf` (5, 20), scored on validation MSE.
- Chosen: depth 4, leaf 20 (a shallow forest, because the data is noisy).
- Explain simply: many decision trees, each trained on a random sample, predictions averaged.

### `common.py` helper (used by all models)
`pick_best` runs every parameter combination, scores it on the **validation** season, and keeps the best. The test season is never used for choosing anything.

### Shubashitha's quick-run checklist
```bash
python src/data.py && python src/features.py && python tests/test_leakage.py
python src/train.py rf
```
Be able to open `features.py` and walk through one row: "for this match, the home team's `gf_l3` is the average goals it scored in its previous 3 games."

---

## C. Shashank's part: SVD, neural net, LSTM, training, evaluation, demo, report

Files: `src/models/svd.py`, `src/models/nn.py`, `src/train.py`, `src/evaluate.py`, `app/demo.py`, `results/`, `report/`, `docs/`.
Theme: **"the learning models and how we judge them."**

### `src/models/svd.py`: collaborative filtering (paper section 4.1)
- Build a table: rows = home team, columns = away team, cells = average total goals from their past meetings. Most cells are empty (each pair meets about twice a season).
- Fill the table with a **low-rank approximation** using iterative "hard-impute" SVD: guess the empty cells, take the top-k singular components, re-fill, repeat 30 times.
- Prediction for home team i vs away team j = the filled cell (i, j). Same idea as predicting a user's rating of a film from similar users.
- To avoid leakage the table for match t only uses the previous 380 matches. Tuned window (380 vs 760) and rank (1-3); chosen rank 1.
- Why it's worst (MSE 3.39): few past meetings per pair, no form information, clipped to a plausible range but still noisy. The paper's version overfit too.

### `src/models/nn.py`: neural net (paper 4.2)
- MLP: 32 inputs -> 32 -> 16 -> 1, ReLU, dropout 0.2, Adam, MSE loss, weight decay 1e-4.
- Inputs standardised with **training-set** mean/std only.
- **Early stopping:** keep the weights from the epoch with the best validation loss, stop after 25 epochs without improvement.
- Averages 3 random seeds, because small data makes single runs unstable.
- Grid: layer sizes, dropout (0.2/0.4), weight decay (1e-4/1e-2).

### `src/models/nn.py`: LSTM (paper 4.3)
- Two LSTMs (home team, away team) each read that team's last 5 matches **in order** (oldest first), 7 numbers per match.
- Their final hidden states (16 each) are concatenated and a linear layer outputs total goals.
- The idea: order matters (recent form vs older form). In practice with only 5 steps and noisy data it performs about like the other models (MSE 2.691, the best of the learned models on accuracy, 56.4%).

### `src/train.py`
- Builds the dataset once, runs each requested model, writes `results/predictions.csv` (predictions for every match), `params.json` (chosen hyperparameters) and `curves.json` (training curves).
- `python src/train.py rf nn` trains only some; results of other models are kept.
- Full run takes about 10 minutes on CPU (SVD ~2.5 min, NN ~3 min, LSTM ~3 min).

### `src/evaluate.py`: how we judge (know this well, it is the core of the argument)
1. **Bookmaker's probability:** from the Over/Under 2.5 odds, `p = (1/over)/(1/over + 1/under)` removes the bookmaker margin.
2. **Implied expected goals:** solve `P(Poisson(lambda) >= 3) = p` for lambda. Now the bookmaker has a "predicted goals" number, so MSE is comparable.
3. **Models to probability:** the same formula in reverse: `P(over 2.5) = 1 - e^-lambda (1 + lambda + lambda^2/2)`.
4. **Metrics:**
   - MSE / MAE: how far predicted goals are from actual.
   - Over/Under accuracy: did we pick the right side of 2.5 (predict over if probability > 0.5)?
   - Brier score: squared error of the probability, rewards calibrated probabilities.
   - Betting ROI: stake 1 on a side when `our probability x odds > 1.05`; ROI = profit / number of bets.
5. **Figures:** model comparison bars, training curves, calibration plot.
- Likely questions:
  - *Why Over/Under 2.5?* it is the standard goals market and has odds for every match.
  - *Why a Poisson assumption?* simple, standard for goal counts; a limitation we state.
  - *Is 57% accuracy good?* only relative to the baseline: "predict the average" gets 57.0%, the bookmaker 57.5%. Read accuracy together with MSE and Brier.

### `app/demo.py` (live demo script)
1. Start: `streamlit run app/demo.py`.
2. **Tab "Predict a match":** pick a 2024/25 match. Show the bookmaker's P(over 2.5) and odds, every model's predicted goals and Over/Under call, then the actual goals revealed.
3. **Tab "Scoreboard":** the metrics table. Say: "bookmaker is best; our models are close but not better."
4. **Tab "Figures":** comparison, training curves, calibration.
- Backup if Streamlit fails: `python src/evaluate.py` prints the same table.

### `report/Team6_Writeup.pdf`
Two pages: problem, dataset, approach, implementation, results table + figure, conclusions. Source is `report/report.html` (printed to PDF with headless Chrome).

### Shashank's quick-run checklist
```bash
python src/train.py nn lstm svd     # ~8 min
python src/evaluate.py
streamlit run app/demo.py
```
Be able to say in 20 seconds each: what the SVD does, what early stopping does, how the bookmaker is turned into a number.

---

## D. Presentation split (suggested, ~8-10 minutes)
| Part | Who | Content |
|---|---|---|
| 1 | Both | Problem + paper in 1 minute (Part A) |
| 2 | Shubashitha | Dataset, features, no-leakage test, time split, RF |
| 3 | Shashank | SVD, NN, LSTM, how we compare to the bookmaker |
| 4 | Shashank | Live demo |
| 5 | Both | Results, conclusion, limitations, future work (Elo/Dixon-Coles, xG, more seasons) |

Rule of thumb for Q&A: Shashank and Shubashitha each answer questions on their own files first; if stuck, fall back to the four ideas in Part A.
