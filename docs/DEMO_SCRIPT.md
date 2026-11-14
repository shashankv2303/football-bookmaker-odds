# How to demo it (3-4 minutes)

## Before the review (do once, 10 minutes early)
```bash
cd <project folder>
source .venv/bin/activate
python src/evaluate.py          # sanity check: prints the scoreboard
streamlit run app/demo.py       # opens http://localhost:8501
```
- Pre-computed results are in `results/`, so **no training is needed and no internet is needed**.
- Open the app once and click through the tabs so it is warm. Keep a terminal visible with `python src/evaluate.py` ready as a backup.
- If Streamlit fails: run `python src/evaluate.py` in the terminal and show `results/model_comparison.png`.

## Script
| Time | Do | Say |
|---|---|---|
| 0:00 | Tab **Predict a match**, pick a high-scoring match | "The bookmaker thinks P(over 2.5) is X%. These are the models' predictions. None of them saw this match." |
| 0:45 | Compare model rows to the bookmaker line | "Models disagree a bit, mostly sitting near the average, around 2.7 goals. Now the real result:" (point at Actual goals) |
| 1:15 | Pick a low-scoring match | "One match proves nothing; football totals are noisy. That's why we evaluate on all 374." |
| 1:45 | Tab **Scoreboard** | "Bookmaker MSE 2.614 is lowest. Our best are about 2.69. Accuracy differences are within noise, plus or minus 2.6 points." |
| 2:30 | Tab **Figures** | "Bar chart, training curves (early stopping on validation), calibration: both are close to the diagonal." |
| 3:00 | Terminal: `python tests/test_leakage.py` | "This proves no feature sees its own or future results." (prints OK) |

## Likely demo-time questions
- *Is this live?* The predictions come from models trained on earlier seasons; the demo loads saved predictions. Retraining takes about 10 minutes (`python src/train.py`).
- *Can it predict a future match?* Not in the demo; it needs that team's last 5 results, which the pipeline can build. Listed as future work.
- *Why does the bookmaker win?* Odds also encode injuries, line-ups and market information that public form statistics do not.
