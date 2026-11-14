// Builds slides/Team6_Presentation.pptx   (run: NODE_PATH=<dir with pptxgenjs> node slides/build_deck.js)
const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "The Bank is Open: Can ML Beat the Bookmaker?";
pres.author = "Team 6";

const DARK = "0B2E23", GREEN = "1F7A4D", LIME = "C6F25A", PAPER = "FFFFFF", MIST = "EEF4F0", INK = "1B1B1B", MUTE = "5C6B63", RED = "D55E00";
const HEAD = "Cambria", BODY = "Calibri";
const RES = path.join(__dirname, "..", "results");

function titleOf(slide, text) {
  slide.addText(text, { x: 0.6, y: 0.35, w: 8.8, h: 0.8, fontFace: HEAD, fontSize: 28, bold: true, color: DARK, margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, head, body, fill) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || MIST }, line: { color: fill || MIST }, objectName: "card " + head });
  slide.addText(head, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: GREEN, margin: 0, isTextBox: true });
  slide.addText(body, { x: x + 0.2, y: y + 0.55, w: w - 0.4, h: h - 0.65, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true });
}
function stat(slide, x, y, w, big, label, dark) {
  slide.addText(big, { x, y, w, h: 0.8, fontFace: HEAD, fontSize: 40, bold: true, color: dark ? LIME : GREEN, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.8, w, h: 0.6, fontFace: BODY, fontSize: 14, color: dark ? "DDE8E1" : MUTE, margin: 0, valign: "top", isTextBox: true });
}

// 1. Title -------------------------------------------------------------
let s = pres.addSlide(); s.background = { color: DARK };
s.addText("The Bank is Open", { x: 0.7, y: 1.3, w: 8.6, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true, color: PAPER, margin: 0, isTextBox: true });
s.addText("Can machine learning beat the bookmaker? Predicting Premier League total goals", { x: 0.7, y: 2.4, w: 8.2, h: 0.9, fontFace: BODY, fontSize: 22, color: LIME, margin: 0, valign: "top", isTextBox: true });
s.addText("UE24CS352A Machine Learning Mini-Project  |  Team 6\nShashank Vinayak (PES1UG24CS435)  ·  Shubashitha G (PES1UG24CS450)", { x: 0.7, y: 4.2, w: 8.6, h: 0.8, fontFace: BODY, fontSize: 14, color: "DDE8E1", margin: 0, valign: "top", isTextBox: true });
s.addNotes("Intro. Our problem is #6: the Stanford CS229 paper 'The Bank is Open: AI in Sports Gambling'. We replicate it and extend it, on English Premier League football.");

// 2. Problem ------------------------------------------------------------
s = pres.addSlide(); s.background = { color: PAPER };
titleOf(s, "Can a model out-predict the bookmaker?");
card(s, 0.6, 1.4, 2.8, 2.6, "The bet", "Bookmakers set an Over/Under line, e.g. 2.5 goals. You bet whether the match total is above or below it.");
card(s, 3.6, 1.4, 2.8, 2.6, "The paper", "Bucquet & Sarukkai (2018) predicted NBA total points with RF, SVD, a neural net and an LSTM, and compared with sportsbook lines.");
card(s, 6.6, 1.4, 2.8, 2.6, "Our goal", "Repeat it on EPL football, add modern models, and test honestly against the bookmaker's own odds.");
s.addText("Success = lower error than the bookmaker, on a season the models have never seen.", { x: 0.6, y: 4.4, w: 8.8, h: 0.5, fontFace: BODY, fontSize: 16, italic: true, color: MUTE, margin: 0, isTextBox: true });
s.addNotes("Explain Over/Under with one example. The paper's best NN got MSE 369.8 vs 320.7 for the sportsbook and picked the right side 51.5% of the time.");

// 3. Data ---------------------------------------------------------------
s = pres.addSlide(); s.background = { color: DARK };
s.addText("Data: 11 seasons of EPL results and odds", { x: 0.6, y: 0.35, w: 8.8, h: 0.8, fontFace: HEAD, fontSize: 28, bold: true, color: PAPER, margin: 0, isTextBox: true });
stat(s, 0.6, 1.4, 2.7, "4,180", "matches, 2014/15 to 2024/25 (football-data.co.uk)", true);
stat(s, 3.6, 1.4, 2.7, "4,031", "usable after requiring 5 prior games per team", true);
stat(s, 6.6, 1.4, 2.7, "32", "features per match, built from past games only", true);
s.addText("Time-ordered split, never random", { x: 0.6, y: 3.3, w: 8.8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: PAPER, margin: 0, isTextBox: true });
const segs = [["Train 2014/15 - 2022/23  (3,283)", 5.6, GREEN], ["Validation 23/24 (374)", 1.6, "3FA56E"], ["Test 24/25 (374)", 1.6, LIME]];
let x = 0.6;
segs.forEach(([t, w, c]) => {
  s.addShape(pres.shapes.RECTANGLE, { x, y: 3.85, w, h: 0.7, fill: { color: c }, line: { color: DARK }, objectName: "split " + t });
  s.addText(t, { x, y: 3.85, w, h: 0.7, fontFace: BODY, fontSize: 14, bold: true, color: c === LIME ? DARK : PAPER, align: "center", valign: "middle", margin: 2, isTextBox: true });
  x += w;
});
s.addText("Bookmaker odds are the benchmark, never an input.", { x: 0.6, y: 4.8, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 14, italic: true, color: "DDE8E1", margin: 0, isTextBox: true });
s.addNotes("Why time split: a random split would let the model see future form. Odds are held out so the comparison is fair.");

// 4. Features ------------------------------------------------------------
s = pres.addSlide(); s.background = { color: PAPER };
titleOf(s, "Features: only what was known before kickoff");
card(s, 0.6, 1.4, 2.8, 2.5, "Recent form", "Each team's last 3 and 5 matches: goals for/against, shots, shots on target, corners.");
card(s, 3.6, 1.4, 2.8, 2.5, "Season so far", "Season-to-date averages for both teams, the paper's opponent-strength idea.");
card(s, 6.6, 1.4, 2.8, 2.5, "Fatigue", "Days since each team's last match, the paper's rest idea. Plus home/away.");
s.addText("Leakage test: corrupt one match and everything after it; every earlier feature stays identical. It passes.", { x: 0.6, y: 4.25, w: 8.8, h: 0.7, fontFace: BODY, fontSize: 16, bold: true, color: GREEN, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Show features.py: each stat is shifted by one match before the rolling mean, so a match never sees its own result. Then mention tests/test_leakage.py.");

// 5. Models --------------------------------------------------------------
s = pres.addSlide(); s.background = { color: PAPER };
titleOf(s, "Four models, straight from the paper");
const models = [
  ["Random Forest", "Paper baseline: many decision trees, predictions averaged."],
  ["SVD", "Collaborative filtering: fill in a home x away goals table."],
  ["Neural net", "32-32-16-1 layers, dropout, early stopping, 3 seeds."],
  ["LSTM", "Reads each team's last 5 games in order."],
];
models.forEach(([h, b], i) => {
  const cx = 0.6 + (i % 2) * 4.5, cy = 1.4 + Math.floor(i / 2) * 1.75;
  card(s, cx, cy, 4.3, 1.55, h, b, MIST);
});
s.addText("Hyperparameters tuned on the validation season only; test season touched once.", { x: 0.6, y: 4.95, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 14, italic: true, color: MUTE, margin: 0, isTextBox: true });
s.addNotes("SVD: fill the empty cells of the team-vs-team table with a low-rank approximation, like recommending films. LSTM: ordered form. Random Forest = many trees voting.");

// 6. Fair comparison -------------------------------------------------------
s = pres.addSlide(); s.background = { color: PAPER };
titleOf(s, "A fair scoreboard against the bookmaker");
const steps = [["1", "Odds to probability", "Remove the margin from Over/Under 2.5 odds."], ["2", "Probability to goals", "Solve for the Poisson mean that gives that probability."], ["3", "Same metrics", "MSE, Over/Under accuracy, Brier score, betting ROI."]];
steps.forEach(([n, h, b], i) => {
  const cx = 0.6 + i * 3.0;
  s.addShape(pres.shapes.OVAL, { x: cx, y: 1.5, w: 0.7, h: 0.7, fill: { color: GREEN }, line: { color: GREEN }, objectName: "step " + n });
  s.addText(n, { x: cx, y: 1.5, w: 0.7, h: 0.7, fontFace: HEAD, fontSize: 22, bold: true, color: PAPER, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(h, { x: cx, y: 2.4, w: 2.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: DARK, margin: 0, isTextBox: true });
  s.addText(b, { x: cx, y: 2.85, w: 2.7, h: 1.2, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true });
});
s.addText("Now the bookmaker has a 'predicted goals' number, so every model is compared on the same scale.", { x: 0.6, y: 4.4, w: 8.8, h: 0.6, fontFace: BODY, fontSize: 16, italic: true, color: MUTE, margin: 0, valign: "top", isTextBox: true });
s.addNotes("This is the key evaluation idea. p = (1/over)/(1/over+1/under); then solve P(Poisson(lambda)>=3)=p.");

// 7. Results chart -------------------------------------------------------
s = pres.addSlide(); s.background = { color: PAPER };
titleOf(s, "Result: nobody beats the bookmaker");
const names = ["Bookmaker", "Predict avg", "Random Forest", "LSTM", "Neural net", "SVD"];
const mse = [2.614, 2.645, 2.689, 2.691, 2.764, 3.388];
// drawn with shapes (renders identically in PowerPoint and Keynote)
names.forEach((n, i) => {
  const y = 1.35 + i * 0.42, len = (mse[i] - 2.4) / (3.5 - 2.4) * 3.3, isBook = i === 0;
  s.addText(n, { x: 0.6, y, w: 1.6, h: 0.34, fontFace: BODY, fontSize: 14, color: INK, align: "right", valign: "middle", margin: 0, isTextBox: true });
  s.addShape(pres.shapes.RECTANGLE, { x: 2.35, y, w: len, h: 0.34, fill: { color: isBook ? RED : "4C72B0" }, line: { color: isBook ? RED : "4C72B0" }, objectName: "bar " + n });
  s.addText(mse[i].toFixed(3), { x: 2.35 + len + 0.08, y, w: 0.8, h: 0.34, fontFace: BODY, fontSize: 14, bold: isBook, color: INK, valign: "middle", margin: 0, isTextBox: true });
});
s.addText("Test MSE of total goals, lower is better (axis starts at 2.4)", { x: 0.6, y: 4.85, w: 5.8, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0, isTextBox: true });
stat(s, 6.8, 1.4, 2.7, "2.614", "bookmaker MSE, the best on the board");
stat(s, 6.8, 2.9, 2.7, "2.689", "best learned model (Random Forest): within 3% of it");
s.addText("2024/25 season, 374 matches", { x: 6.8, y: 4.5, w: 2.7, h: 0.4, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0, isTextBox: true });
s.addNotes("Bookmaker 2.614. Predicting the training average is 2.645, so our models add little. SVD is worst: few past meetings per team pair, no form information.");

// 8. Accuracy & betting -----------------------------------------------------
s = pres.addSlide(); s.background = { color: DARK };
s.addText("Accuracy and betting tell the same story", { x: 0.6, y: 0.35, w: 8.8, h: 0.8, fontFace: HEAD, fontSize: 28, bold: true, color: PAPER, margin: 0, isTextBox: true });
stat(s, 0.6, 1.4, 2.7, "57.5%", "bookmaker picks the right side of 2.5", true);
stat(s, 3.6, 1.4, 2.7, "57.0%", "just predicting the average", true);
stat(s, 6.6, 1.4, 2.7, "53-56%", "our models; noise is about +/-2.6 points", true);
s.addText("Flat-stake value betting returns about zero or negative ROI for every model. There is no profitable strategy here, which is the same conclusion as the paper.", { x: 0.6, y: 3.4, w: 8.8, h: 1.2, fontFace: BODY, fontSize: 18, color: PAPER, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Be upfront: with 374 matches, accuracy differences of 1 to 3 points are noise. Read accuracy together with MSE and Brier score.");

// 9. Why & conclusion -------------------------------------------------------
s = pres.addSlide(); s.background = { color: PAPER };
titleOf(s, "Why, limits, and what next");
card(s, 0.6, 1.4, 2.8, 3.3, "Why it's hard", "Goals are rare and random. Form stats carry little signal, and bookmakers also know injuries and line-ups.");
card(s, 3.6, 1.4, 2.8, 3.3, "Limitations", "One league, one test season, averaged odds, Poisson assumes no over-dispersion, promoted teams' first games dropped.");
card(s, 6.6, 1.4, 2.8, 3.3, "Next steps", "Elo / Dixon-Coles team ratings, expected-goals data, line-ups and injuries, several test seasons.");
s.addNotes("Conclusion: a clean, leak-free pipeline reproduces the paper's finding. The honest result is that the bookmaker is hard to beat with public form statistics.");

// 10. Demo -------------------------------------------------------------------
s = pres.addSlide(); s.background = { color: DARK };
s.addText("Live demo", { x: 0.7, y: 0.5, w: 8.6, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: PAPER, margin: 0, isTextBox: true });
const demo = ["Pick a 2024/25 match", "Bookmaker's probability and odds", "Every model's prediction vs the actual goals", "Scoreboard and figures"];
s.addText(demo.map((t, i) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < demo.length - 1 } })),
  { x: 0.7, y: 1.7, w: 8.2, h: 2.2, fontFace: BODY, fontSize: 20, color: PAPER, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
s.addText("streamlit run app/demo.py", { x: 0.7, y: 4.4, w: 6, h: 0.5, fontFace: "Courier New", fontSize: 18, color: LIME, margin: 0, isTextBox: true });
s.addNotes("Switch to the terminal / browser. Backup: python src/evaluate.py prints the same table.");

pres.writeFile({ fileName: path.join(__dirname, "Team6_Presentation.pptx") }).then(f => console.log("wrote", f));
