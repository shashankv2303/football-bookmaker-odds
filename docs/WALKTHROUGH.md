# Walkthrough: what we built and how to explain it

## 1. The paper in 60 seconds
Bucquet & Sarukkai (2018) predict the **total points** scored in an NBA game (the "Over/Under" bet) using game stats
from the previous 3 games, opponent season averages, rest days and travel. Models: Random Forest (baseline),
SVD "collaborative filtering", a fully-connected neural net and an LSTM. They score models by MSE and by how often the
model picks the right side (over/under) of the sportsbook's line. Result: neural net MSE 369.8 vs the sportsbook's 320.7,
correct side 51.5% of the time. So close, but the bookmaker is still better. They note the NBA changes over time
(three-point boom), so older seasons hurt.

## 2. What we changed (and why it's justifiable)
- **Sport:** English Premier League. Same problem (total goals instead of total points), but football-data.co.uk gives free
  results *and* bookmaker odds for every match, so the "beat the house" comparison is direct. (Check with faculty that a sport swap is fine.)
- **Fairer benchmark:** we convert the bookmaker's Over/Under 2.5 odds to an implied expected-goals number so MSE is comparable.

## 3. Pipeline, file by file
1. `src/data.py` downloads 11 seasons (4,180 matches), merges them, keeps results, shots, corners and odds.
2. `src/features.py` builds features **only from earlier matches**: rolling means of the last 3 and 5 games (goals for/against, shots,
   shots on target, corners), season-to-date means, days of rest, for both teams (32 features). It also builds a (2 teams x 5 games x 7 stats) sequence for the LSTM.
   Matches where a team has fewer than 5 prior games (start of the data, newly promoted teams) are dropped: 4,031 remain.
3. `src/models/*` four models, all predicting total goals, hyperparameters chosen on the validation season.
4. `src/train.py` runs them and saves predictions for every match.
5. `src/evaluate.py` computes the metrics, bets, and figures.
6. `app/demo.py` shows a match, each model's prediction, and the bookmaker's line.

Split: train 2014/15-2022/23 (3,283 matches), validation 2023/24 (374), test 2024/25 (374). Time-ordered, like the paper.

## 4. The models in plain words
- **Random Forest:** many decision trees averaged. Baseline.
- **SVD / collaborative filtering:** a home-team x away-team table of past total goals; most cells are empty (each pair meets twice a season),
  so we fill it with a low-rank approximation (like Netflix recommendations: "teams like these usually produce this many goals"). Rebuilt for every match from the previous 380 matches only.
- **Neural net:** 32 inputs -> 32 -> 16 -> 1, ReLU, dropout, weight decay, early stopping on validation, average of 3 seeds.
- **LSTM:** one LSTM reads the home team's last 5 matches in order, another the away team's; their final states feed a linear layer.

## 5. Results (test season, 374 matches)
| Model | MSE | O/U 2.5 accuracy |
|---|---|---|
| Bookmaker | 2.614 | 57.5% |
| Train mean (predict the average) | 2.645 | 57.0% |
| Random Forest | 2.689 | 54.5% |
| LSTM | 2.691 | 56.4% |
| Neural net | 2.764 | 53.2% |
| SVD | 3.388 | 54.5% |

Takeaways: nobody beats the bookmaker; our models are barely better than predicting the average; the bookmaker's edge is small but real.
Football totals are very noisy (a goal is rare and random), so there is little signal in form stats. Betting ROI is around zero or negative.
The accuracy standard error with 374 matches is about 2.6 points, so differences of 1-3 points are noise. This matches the paper's finding.

## 6. Likely questions and answers
- **Why split by time, not randomly?** Random splits let the model see the future (a team's later form), inflating scores. Real betting only has the past.
- **How do you know there's no leakage?** `tests/test_leakage.py` corrupts a match and all later ones and checks that earlier matches' features don't change. Rolling stats are shifted by one match.
- **Why are the odds not features?** They'd contain the market's own forecast; the point is to compare against it.
- **Why is the SVD so bad?** Few past meetings per pair, so the matrix is sparse and noisy. In the paper it also overfit (train MSE 2.95 vs test 459).
- **Why does the train-mean baseline have such high accuracy?** In 2024/25 the Under 2.5 side happened a bit more often and the mean sits near the 2.5 line; with 374 matches this is within noise. It's a reminder to read accuracy next to MSE and Brier score.
- **How do you pick hyperparameters?** Small grids scored on the validation season only; test is touched once at the end.
- **What's MSE vs accuracy?** MSE measures how far predicted goals are from actual; accuracy only checks which side of 2.5 we picked.
- **Why do you use a Poisson formula at all?** Only to turn the bookmaker's odds into an "expected goals" number so it can be compared with our models. Goals are small counts, and Poisson is the standard simple model for them. It is part of the scoring, not one of our models.
- **What would improve it?** Team-strength ratings (Elo/Dixon-Coles), injuries, lineups, xG data; more careful probability models than a single Poisson mean.
- **Does it make money?** No. Flat-stake ROI is about zero/negative, which is the honest answer and also what the paper found.
- **Limitations?** One league, one test season, single-bookmaker-average odds, Poisson conversion assumes no over-dispersion.

## 7. How to run it in front of the panel
`streamlit run app/demo.py` opens the demo. Tab 1: pick a match, show predictions vs bookmaker, reveal the actual result. Tab 2: scoreboard. Tab 3: figures.
Have `python src/evaluate.py` ready in a terminal as a backup (takes seconds).
