// Group 5 — Model milestone deck (11/22). 16:9, Calibri, Northwestern purple.
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/6307a341-66a7-4080-859d-c202e9a11dd6_03facb7c-deb0-4209-b94d-9d8c6ed3f7ad/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Group5 Purple",
  headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: {
    dk1: "1F1F1F", lt1: "FFFFFF", dk2: "4E2A84", lt2: "EFEBF5",
    accent1: "4E2A84", accent2: "ED7D31", accent3: "7A5AA6", accent4: "B6A6D1", accent5: "5B9BD5", accent6: "70AD47",
    hlink: "4E2A84", folHlink: "7A5AA6",
  },
};
const PURPLE = "4E2A84", LILAC = "EFEBF5", ORANGE = "ED7D31", INK = "1F1F1F", GREY = "595959", LIGHTGREY = "D9D9D9", MID = "7A5AA6", PALE = "B6A6D1";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";          // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Group 5";
pres.title = "Predicting Bail and Remand Decisions at NYC Arraignments";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE", background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0, y: 0, w: 10, h: 0.12, fill: { color: PURPLE } } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  margin: [0.4, 0.4, 0.5, 0.4],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.4, y: 0.22, w: 7.6, h: 0.6, fontSize: 26, bold: true, color: C.text2, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "SE_POL 408 · Group 5 · Model milestone", options: { x: 0.4, y: 5.28, w: 5, h: 0.25, fontSize: 9, color: GREY, margin: 0 } } },
  ],
  slideNumber: { x: 9.3, y: 5.28, w: 0.4, h: 0.25, fontSize: 9, color: GREY },
});

// ---------- helpers ----------
function corner(slide) {   // purple geometric corner, as in the course slides (top-right, small)
  slide.addShape(pres.ShapeType.rtTriangle, { x: 8.8, y: 0, w: 1.2, h: 0.95, fill: { color: PURPLE }, line: { color: PURPLE }, flipH: true, flipV: true, objectName: "corner" });
  slide.addShape(pres.ShapeType.rtTriangle, { x: 9.4, y: 0, w: 0.6, h: 0.48, fill: { color: PALE }, line: { color: PALE }, flipH: true, flipV: true, objectName: "corner2" });
}
function guide(slide, question, answer) {
  // Two-line dialogue, as in the course slides: the client asks, the team answers in plain words.
  const rows = [[question, "?", ORANGE, "FDF1E8", "Court administrator: "], [answer, "A", PURPLE, "F5F2F9", "Our answer: "]];
  rows.forEach(([text, mark, col, fill, who], i) => {
    const y = 0.92 + i * 0.5;
    slide.addShape(pres.ShapeType.ellipse, { x: 0.5, y: y + 0.02, w: 0.38, h: 0.38, fill: { color: col }, line: { color: col }, objectName: "speaker " + mark });
    slide.addText(mark, { x: 0.5, y: y + 0.02, w: 0.38, h: 0.38, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "speaker mark " + mark });
    slide.addShape(pres.ShapeType.roundRect, { x: 1.0, y, w: 8.6, h: 0.42, fill: { color: fill }, line: { color: col, width: 0.75 }, rectRadius: 0.08, objectName: "bubble " + mark });
    slide.addText([{ text: who, options: { bold: true, color: col } }, { text, options: { italic: i === 0, color: INK } }],
      { x: 1.1, y, w: 8.4, h: 0.42, fontSize: 11.5, valign: "middle", margin: 0, isTextBox: true, objectName: "bubble text " + mark });
  });
}
function h3(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.3, fontSize: 13, bold: true, color: PURPLE, margin: 0, isTextBox: true, objectName: "heading " + text });
}
function bullets(slide, items, x, y, w, h, size = 11.5) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 10 }, breakLine: i < items.length - 1, paraSpaceAfter: 4 } })),
    { x, y, w, h, fontSize: size, color: INK, valign: "top", margin: 0.02, isTextBox: true, objectName: "bullets " + items[0].slice(0, 20) });
}
function table(slide, rows, x, y, w, colW, opts = {}) {
  const head = rows[0].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: PURPLE }, fontSize: opts.fs ?? 10.5, align: "left", valign: "middle" } }));
  const body = rows.slice(1).map((r, i) => r.map((t, j) => {
    const o = { fontSize: opts.fs ?? 10.5, color: INK, fill: { color: i % 2 ? "FFFFFF" : "F5F2F9" }, align: "left", valign: "middle" };
    if (opts.boldRow !== undefined && i === opts.boldRow) o.bold = true;
    if (opts.rightCols && opts.rightCols.includes(j)) o.align = "right";
    return typeof t === "object" ? { text: t.text, options: { ...o, ...t.options } } : { text: String(t), options: o };
  }));
  slide.addTable([head, ...body], { x, y, w, colW, border: { type: "solid", pt: 0.5, color: LIGHTGREY }, rowH: opts.rowH ?? 0.26, margin: 0.04, autoPage: false, objectName: opts.name ?? "table" });
}

// =====================================================================
// Slide 0 — Title
// =====================================================================
pres.addSection({ title: "Milestone" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Milestone" });
  s.addShape(pres.ShapeType.rtTriangle, { x: 0, y: 0.12, w: 1.8, h: 1.1, fill: { color: PURPLE }, line: { color: PURPLE }, flipH: true, flipV: true, objectName: "corner purple" });
  s.addShape(pres.ShapeType.rtTriangle, { x: 0, y: 0.12, w: 1.0, h: 0.6, fill: { color: PALE }, line: { color: PALE }, flipH: true, flipV: true, objectName: "corner lilac" });
  s.addText("Predicting Bail and Remand Decisions at NYC Arraignments", { x: 0.7, y: 1.25, w: 8.6, h: 0.95, fontSize: 30, bold: true, color: PURPLE, margin: 0, valign: "bottom", isTextBox: true, objectName: "title" });
  s.addText("Model milestone: Models → Tune → Select → Evaluate", { x: 0.7, y: 2.25, w: 8.6, h: 0.32, fontSize: 14, color: GREY, margin: 0, isTextBox: true, objectName: "subtitle" });
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 2.68, w: 8.6, h: 1.5, fill: { color: LILAC }, line: { color: PALE, width: 0.75 }, rectRadius: 0.08, objectName: "about box" });
  s.addText([
    { text: "What this project is about", options: { bold: true, color: PURPLE, fontSize: 12, breakLine: true } },
    { text: "When a person arrested in New York City first sees a judge, the judge either releases them or sets bail / sends them to jail (remand). We predict that decision using only what the judge knows at that moment.", options: { bullet: { indent: 10 }, breakLine: true } },
    { text: "Since the 2019 bail reform, bail is allowed mainly for serious charges. We ask how much of the decision the charge and record explain, and how much the borough and the individual judge add: a question about how consistent court decisions are.", options: { bullet: { indent: 10 }, breakLine: true } },
    { text: "Data: public pretrial records from the New York State court system, 2021–2024 (449,474 cases).", options: { bullet: { indent: 10 } } },
  ], { x: 0.85, y: 2.71, w: 8.3, h: 1.44, fontSize: 11, color: INK, valign: "middle", margin: 0, paraSpaceAfter: 3, isTextBox: true, objectName: "about text" });
  s.addText([
    { text: "Group 5 — ", options: { bold: true, color: INK } },
    { text: "Yiling Long · Xinhui Zong · Shuyi Niu · Yixuan Wen · Yanqi Wu", options: { color: INK } },
  ], { x: 0.7, y: 4.35, w: 8.6, h: 0.3, fontSize: 13, margin: 0, isTextBox: true, objectName: "team" });
  s.addText("SE_POL 408 Applied Data Science and Application in Social Policy · Dr. Ruhan Circi · Northwestern University · November 22, 2026",
    { x: 0.7, y: 4.7, w: 8.6, h: 0.28, fontSize: 10, color: GREY, margin: 0, isTextBox: true, objectName: "course" });
  s.addText("github.com/POL408-Team5/ny-pretrial-detention", { x: 0.7, y: 4.98, w: 6, h: 0.28, fontSize: 10, color: PURPLE, margin: 0, isTextBox: true, objectName: "repo" });
  s.addNotes(
`AI Use Statement (Group 5)

Tools used: Claude (Anthropic) — Yiling; Codex (OpenAI) — Shuyi, Yanqi, Yixuan, Xinhui.

What we used them for: writing and reviewing the Python code (sample construction, feature pipeline, model training, evaluation, fairness functions, the locked final-test workflow), re-checking numbers against the result files, drafting documentation, and verifying legal background and citations against primary sources. Every model was trained and every number in this deck was produced by our own code in the shared repository (github.com/POL408-Team5/ny-pretrial-detention); a full per-task log is in AI_Use_Statement.md.

What we did ourselves: defined the question and the target, chose the sample rules, the feature sets, the metric and the validation design, reviewed each other's pull requests, decided the final model, and froze the specification before the one-time 2024 test.

One example of a key prompt (Yiling to Claude): "Write the final-test step: refit the frozen XGBoost and logit specifications on 2021–2023 using judge_id_final, score 2024 once, and refuse to run unless the frozen spec is committed and no test has been run before."`);
}

// =====================================================================
// Slide 1 — Candidate models and baseline
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Milestone" });
  corner(s);
  s.addText("Candidate Models and Baseline", { placeholder: "title" });
  guide(s, "Can we tell, from what the judge sees at arraignment, which cases will end in bail or remand?",
    "We test a simple rule and five models, scored on how well they rank bail cases above release cases.");

  h3(s, "The prediction task", 0.5, 2.05, 4.3);
  bullets(s, [
    "Target y_detain: 1 = bail set or remanded, 0 = released (on recognizance or with non-money conditions). Binary classification.",
    "One row per case in NYC criminal courts; 449,474 cases. Train on 2021–22, choose the model on 2023, test once on 2024.",
    "Inputs known at arraignment: charge, record and pending cases, supervision, arrest type, age; then county; then an anonymised judge code.",
  ], 0.5, 2.33, 4.3, 1.75, 10.5);
  h3(s, "Metric: ROC-AUC", 0.5, 3.85, 4.3);
  bullets(s, [
    "Only 12.8% of cases end in bail or remand, so accuracy misleads: “always release” is 87% accurate and useless.",
    "AUC = the chance that a random bail case is ranked above a random release case (0.5 = coin flip, 1 = perfect). AP and Brier are reported too.",
  ], 0.5, 4.13, 4.3, 1.0, 10.5);

  h3(s, "Six candidates, each on three nested feature sets", 5.2, 2.05, 4.4);
  table(s, [
    ["Model", "In plain words"],
    ["Rule baseline", "charge class × violent-felony flag"],
    ["Logistic regression", "weighted sum of inputs (L2)"],
    ["LASSO logistic", "same, drops weak inputs (L1)"],
    ["Decision tree", "one set of yes/no questions"],
    ["Random forest", "300 trees voting"],
    ["XGBoost", "trees that fix each other’s errors"],
  ], 5.2, 2.33, 4.4, [1.6, 2.8], { fs: 10.5, rowH: 0.27, name: "candidates" });
  s.addText("Feature sets: CASE ⊂ CASE_COURT (+ county) ⊂ CASE_COURT_JUDGE (+ judge). Race, ethnicity and gender are never inputs; we use them only to check errors by group.",
    { x: 5.2, y: 4.37, w: 4.4, h: 0.7, fontSize: 9.5, color: GREY, margin: 0, isTextBox: true, objectName: "feature note" });
  s.addNotes("Target carried over from the Data Exploration milestone. The judge code is built on the training years; judges not seen in training are grouped as 'unseen' (30% of 2023 cases, 28% of 2024 after the final refit on 2021–2023). Nominal ($1) bail is excluded from the main sample because the data dictionary says it is usually set to credit time served on another case; keeping it, or skipping arrest-cycle de-duplication, changes no conclusion (sensitivity checks in the repository).");
}

// =====================================================================
// Slide 2 — Hyperparameter tuning
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Milestone" });
  corner(s);
  s.addText("Hyperparameter Tuning", { placeholder: "title" });
  guide(s, "How did you choose each model’s settings without peeking at 2023 or 2024?",
    "All tuning stayed inside 2021–22: each setting was scored five times on data it had not been fit on.");

  h3(s, "Method", 0.5, 2.05, 4.1);
  bullets(s, [
    "Grid search on 2021–22, scored by ROC-AUC with 5-fold stratified cross-validation.",
    "Encoding and scaling are refit inside every fold, so held-out folds never leak in.",
    "XGBoost: the number of trees is set by early stopping on a 20% slice of the training fold (up to 2,000; stop after 50 rounds without gain).",
  ], 0.5, 2.33, 4.1, 1.45, 10.5);

  const fy = 4.08, fx = 0.5, fw = 0.66;
  s.addText("5-fold cross-validation, inside 2021–22 only", { x: fx, y: fy - 0.28, w: 4.1, h: 0.24, fontSize: 9.5, bold: true, color: GREY, margin: 0, isTextBox: true, objectName: "fold caption" });
  for (let r = 0; r < 2; r++) for (let k = 0; k < 5; k++) {
    const held = (r === 0 && k === 0) || (r === 1 && k === 1);
    s.addShape(pres.ShapeType.rect, { x: fx + k * (fw + 0.05), y: fy + r * 0.27, w: fw, h: 0.21, fill: { color: held ? ORANGE : PALE }, line: { color: "FFFFFF", width: 0.5 }, objectName: `fold ${r}-${k}` });
  }
  s.addText([{ text: "■ ", options: { color: ORANGE } }, { text: "held-out fold (scored)    ", options: { color: GREY } }, { text: "■ ", options: { color: PALE } }, { text: "fitting folds   … five rounds in total", options: { color: GREY } }],
    { x: fx, y: fy + 0.56, w: 4.1, h: 0.22, fontSize: 9, margin: 0, isTextBox: true, objectName: "fold legend" });

  h3(s, "Grids searched and best values (full feature set)", 5.0, 2.05, 4.6);
  table(s, [
    ["Model", "Grid", "Best"],
    ["Logistic (L2)", "C: 0.01, 0.1, 1, 10", "C = 1"],
    ["LASSO (L1)", "C: 0.001, 0.01, 0.1, 1", "C = 1"],
    ["Decision tree", "max_depth: 3, 5, 8, 12\nmin_samples_leaf: 50, 200", "12\n200"],
    ["Random forest", "max_depth: 8, 16, None\nmin_samples_leaf: 20, 100", "None\n20"],
    ["XGBoost", "max_depth: 3, 5, 7\nlearning_rate: 0.05, 0.1\nsubsample: 0.8; trees by early stopping", "7\n0.05\n1,018 trees"],
  ], 5.0, 2.33, 4.6, [1.3, 2.45, 0.85], { fs: 10, rowH: 0.27, name: "grid table" });
  s.addNotes("Best values for the smaller feature sets differ only for logistic on CASE_COURT (C = 10) and XGBoost max_depth (5 on CASE and CASE_COURT). The linear models and the tree select the edge of their grid; we did not widen it because the gaps between model families (next slide) are far larger than anything a finer grid moves. Tree counts from the final refit on 2021–2023: 574 / 854 / 1,018 for the three feature sets. During development the same rule chose 470–966 trees depending on the machine, with validation AUC moving by less than 0.0001; that is why the count is set by the rule rather than fixed. Random seed 408 throughout.");
}

// =====================================================================
// Slide 3 — Model comparison and selection
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Milestone" });
  corner(s);
  s.addText("Model Comparison and Selection", { placeholder: "title" });
  guide(s, "Which model should we trust, and how much better is it than a simple rule?",
    "XGBoost. Every model beats the rule by a wide margin; XGBoost is best both in tuning and on the next year (2023).");

  h3(s, "Tuned models on the full feature set", 0.5, 2.05, 5.6);
  table(s, [
    ["Model", "5-fold CV AUC, 2021–22\n(mean ± SD across folds)", "2023 validation\nAUC", "2023\nAP"],
    ["Rule baseline", "—", "0.847", "0.478"],
    ["Logistic (L2)", "0.940 ± 0.002", "0.924", "0.677"],
    ["LASSO (L1)", "0.940 ± 0.002", "0.924", "0.677"],
    ["Decision tree", "0.936 ± 0.002", "0.922", "0.656"],
    ["Random forest", "0.940 ± 0.002", "0.924", "0.668"],
    ["XGBoost", "0.949 ± 0.002", "0.935", "0.703"],
  ], 0.5, 2.33, 5.5, [1.5, 1.9, 1.2, 0.9], { fs: 10.5, rowH: 0.29, boldRow: 5, rightCols: [1, 2, 3], name: "comparison" });
  s.addText("Validation = fit on all of 2021–22, scored on 2023, a later year not used for tuning.",
    { x: 0.5, y: 4.48, w: 5.5, h: 0.3, fontSize: 9.5, color: GREY, margin: 0, isTextBox: true, objectName: "table note" });

  h3(s, "Why XGBoost", 6.3, 2.05, 3.3);
  bullets(s, [
    "Best on both cross-validation and the 2023 validation year, on all three feature sets: +0.011 AUC over logistic on the same inputs (95% CI 0.010–0.012).",
    "It does not over-fit more than the others: every model drops 0.013–0.016 AUC from tuning to 2023; XGBoost drops 0.014. Logistic stays as the transparent reference.",
  ], 6.3, 2.33, 3.3, 2.7, 10.5);
  s.addNotes("Course process: tune within each family by CV, then select across families on data not used for tuning. We select on the later year because the final use is also a later year, so the choice reflects drift (new judges; the 2022 and 2023 bail-law amendments). The out-of-fold CV scores are a development diagnostic, not a nested-CV estimate. The judge code helps XGBoost and the linear models (+0.003 AUC) but not the random forest (−0.002); random forest is best on CASE_COURT (0.926). Frozen before the test: XGBoost (max_depth 7, learning rate 0.05, subsample 0.8, early stopping) × three feature sets, with the threshold fixed on 2023 so that the predicted share of positives equals the actual share (12.8%).");
}

// =====================================================================
// Slide 4 — Test results and what they mean
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Milestone" });
  corner(s);
  s.addText("Test Data Results", { placeholder: "title" });
  guide(s, "Does it hold up on a year the model never saw?",
    "Yes. Refit on 2021–23 and run once on all 135,821 cases of 2024, it does as well as on 2023.");

  h3(s, "XGBoost, full feature set: validation vs. test", 0.5, 2.05, 4.4);
  table(s, [
    ["Metric", "2023", "2024 test"],
    ["ROC-AUC", "0.935", "0.937 (CI 0.935–0.938)"],
    ["Average precision", "0.703", "0.689"],
    ["Brier score", "0.063", "0.061"],
    ["Actual / mean predicted rate", "12.8% / 11.9%", "12.0% / 11.4%"],
  ], 0.5, 2.33, 4.4, [1.75, 1.05, 1.6], { fs: 10, rowH: 0.26, rightCols: [1, 2], name: "val vs test" });

  h3(s, "2024 decisions at the frozen threshold (0.351)", 5.2, 2.05, 4.4);
  table(s, [
    ["", "Predicted release", "Predicted bail / remand"],
    ["Released (119,477)", "113,371", "6,106  (5.1%)"],
    ["Bail / remand (16,344)", "6,047  (37.0%)", "10,297"],
  ], 5.2, 2.33, 4.4, [1.6, 1.25, 1.55], { fs: 10, rowH: 0.29, rightCols: [1, 2], name: "confusion" });
  s.addText("Of the cases it flags, 63% did get bail or remand. Errors are measured against what judges decided, not against a “right” decision.",
    { x: 5.2, y: 3.28, w: 4.4, h: 0.45, fontSize: 9.5, color: GREY, margin: 0, isTextBox: true, objectName: "confusion note" });

  // policy takeaways
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 3.85, w: 9.1, h: 1.33, fill: { color: LILAC }, line: { color: PALE, width: 0.75 }, rectRadius: 0.06, objectName: "takeaway box" });
  s.addText([
    { text: "What this means", options: { bold: true, color: PURPLE, fontSize: 11.5, breakLine: true } },
    { text: "Bail decisions mostly follow the case. ", options: { bold: true, bullet: { indent: 10 } } },
    { text: "Charge and record alone reach AUC 0.930, consistent with a law that ties bail to the charge.", options: { breakLine: true } },
    { text: "Where the case is heard still matters a little. ", options: { bold: true, bullet: { indent: 10 } } },
    { text: "Borough adds +0.004 and the judge +0.003 AUC: similar cases get somewhat different odds. A signal for consistency review, not proof of unfair treatment.", options: { breakLine: true } },
    { text: "About 1 in 3 bail decisions is not anticipated from the case. ", options: { bold: true, bullet: { indent: 10 } } },
    { text: "That is judgment or facts outside the data, so a model like this could help courts monitor patterns, not replace the judge.", options: {} },
  ], { x: 0.65, y: 3.88, w: 8.85, h: 1.27, fontSize: 10, color: INK, valign: "middle", margin: 0, paraSpaceAfter: 2, isTextBox: true, objectName: "takeaway text" });

  s.addNotes("The 2024 run was executed once on 2026-11-15 from a frozen, committed specification (reports/bd07_frozen_spec.json; run log reports/bd07_test_run_log.json). Increments on 2024 with paired-bootstrap 95% CIs: + county 0.930 → 0.934 (+0.004, CI 0.004–0.004); + judge 0.934 → 0.937 (+0.003, CI 0.003–0.004); 72% of 2024 cases had a judge seen in 2021–2023. Permutation importance on 2024: shuffling the charge inputs costs 0.28 AUC, the record counts 0.09, the judge code 0.008. Group audit (case-only inputs, same threshold): false positive rate 6.4% / false negative rate 34% for Black defendants vs. 4.2% / 47% for White defendants; adding county and judge leaves the gaps essentially unchanged, so a tool built on past decisions would carry those differences forward unless audited. Limitations: six criminal-record counts have an undocumented reference date (dropping them lowers AUC to 0.900 but changes no ranking); the 2022 and 2023 bail-law amendments fall inside the data window. The judge increment is predictive information, not a causal effect.");
}

(async () => {
  await pres.writeFile({ fileName: "/home/claude/deck/Group5_Model_Milestone.pptx" });
  await applyTheme("/home/claude/deck/Group5_Model_Milestone.pptx", THEME);
  console.log("written");
})();
