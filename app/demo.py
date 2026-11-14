"""Live demo.  Run:  streamlit run app/demo.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import LABELS, evaluate, p_over  # noqa: E402

st.set_page_config(page_title="The Bank is Open", layout="wide")
st.title("The Bank is Open: can ML beat the bookmaker? (EPL total goals)")

df = pd.read_csv(ROOT / "results" / "predictions.csv", parse_dates=["Date"])
models = [c[5:] for c in df.columns if c.startswith("pred_") and c != "pred_book"]
test = df[df["split"] == "test"].dropna(subset=["Avg>2.5"]).reset_index(drop=True)

tab1, tab2, tab3 = st.tabs(["Predict a match", "Scoreboard", "Figures"])

with tab1:
    test["label"] = test["Date"].dt.strftime("%Y-%m-%d") + "  " + test["HomeTeam"] + " vs " + test["AwayTeam"]
    row = test[test["label"] == st.selectbox("Pick a match from the held-out 2024/25 season", test["label"])].iloc[0]
    pb = (1 / row["Avg>2.5"]) / (1 / row["Avg>2.5"] + 1 / row["Avg<2.5"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Bookmaker P(over 2.5)", f"{pb:.1%}")
    c2.metric("Over odds / Under odds", f"{row['Avg>2.5']:.2f} / {row['Avg<2.5']:.2f}")
    c3.metric("Actual total goals (revealed)", int(row["TotalGoals"]))
    tbl = pd.DataFrame({"Model": [LABELS[m] for m in models],
                        "Predicted goals": [row[f"pred_{m}"] for m in models],
                        "P(over 2.5)": [p_over(row[f"pred_{m}"]) for m in models]})
    tbl["Call"] = ["OVER" if p > 0.5 else "UNDER" for p in tbl["P(over 2.5)"]]
    st.dataframe(tbl.style.format({"Predicted goals": "{:.2f}", "P(over 2.5)": "{:.1%}"}), hide_index=True)
    st.caption("Models only saw matches played before this one. Odds are the benchmark, never an input.")

with tab2:
    res, _ = evaluate(df[df["split"] == "test"], models)
    st.dataframe(res.style.format({"MSE": "{:.3f}", "MAE": "{:.3f}", "O/U acc": "{:.1%}",
                                   "Brier(P_over)": "{:.3f}", "ROI": "{:.1%}"}), hide_index=True)
    st.caption("Test = 2024/25 season (374 matches). Accuracy has a standard error of about +/-2.6 points.")

with tab3:
    for f in ("model_comparison", "training_curves", "calibration"):
        st.image(str(ROOT / "results" / f"{f}.png"))
