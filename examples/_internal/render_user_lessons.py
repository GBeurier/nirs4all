"""Render annotated, deterministic SVG lessons for the nirs4all user guide.

All numbers are explicit teaching examples, not model-performance benchmarks.
Run with ``python examples/_internal/render_user_lessons.py`` from the repository.
"""

from __future__ import annotations

import html
import math
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs/source/assets/guide"
INK, MUTED = "#162C46", "#4C6278"
BLUE, TEAL, PURPLE, ORANGE = "#1463A0", "#007B71", "#7150A5", "#A64C12"
PAPER = "#F3F6FA"


class Lesson:
    """A deliberately laid-out lesson, with readable text and precise arrows."""

    def __init__(self, name: str, title: str, subtitle: str, height: int = 720):
        self.name, self.height = name, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{html.escape(title)}</title><desc id="description">{html.escape(subtitle)}</desc>',
            '<defs><filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">'
            '<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#243B53" flood-opacity=".07"/>'
            '</filter><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" '
            'orient="auto-start-reverse"><path d="M0,0 L8,4.5 L0,9" fill="none" '
            'stroke="#64788C" stroke-width="1.4"/></marker></defs>',
            '<style>text{font-family:Inter,DejaVu Sans,Arial,sans-serif;fill:#162C46}</style>',
            f'<rect width="1200" height="{height}" rx="24" fill="{PAPER}"/>',
            '<rect x="32" y="32" width="6" height="66" rx="3" fill="#1463A0"/>',
        ]
        self.text(56, 59, title, size=30, weight=700)
        for i, line in enumerate(textwrap.wrap(subtitle, 105)):
            self.text(56, 94 + 25 * i, line, size=19, fill=MUTED)

    def text(self, x: float, y: float, value: str, size: int = 21, fill: str = INK,
             weight: int = 400, anchor: str = "start") -> None:
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
                          f'text-anchor="{anchor}" style="fill:{fill}">{html.escape(value)}</text>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "white",
             stroke: str = "#D9E2EC", radius: int = 16, shadow: bool = False) -> None:
        effect = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
                          f'rx="{radius}" fill="{fill}" stroke="{stroke}"{effect}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, arrow: bool = False,
             color: str = "#64788C", dashed: bool = False, width: float = 2) -> None:
        end = ' marker-end="url(#arrow)"' if arrow else ""
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(f'<path d="M{x1:g},{y1:g} L{x2:g},{y2:g}" fill="none" '
                          f'stroke="{color}" stroke-width="{width}"{end}{dash}/>')

    def card(self, x: float, y: float, w: float, h: float, label: str,
             lines: list[str], color: str = BLUE, number: str = "") -> None:
        self.rect(x, y, w, h, shadow=True)
        self.rect(x, y, 7, h, fill=color, stroke=color, radius=3)
        offset = 24
        if number:
            self.rect(x + 21, y + 18, 35, 32, fill=PAPER, radius=8)
            self.text(x + 38, y + 41, number, 19, color, 700, "middle")
            offset = 70
        size = min(23, max(17, int((w - offset - 17) / max(1, len(label)) / .55)))
        self.text(x + offset, y + 42, label, size, color, 700)
        yy = y + 79
        for value in lines:
            for row in textwrap.wrap(value, max(15, int((w - 48) / 10))):
                self.text(x + 24, yy, row, 20)
                yy += 29

    def chip(self, x: float, y: float, w: float, value: str, color: str = BLUE) -> None:
        self.rect(x, y, w, 40, stroke=color, radius=10)
        size = min(19, max(16, int((w - 20) / max(1, len(value)) / .55)))
        self.text(x + w / 2, y + 27, value, size, color, 600, "middle")

    def note(self, value: str) -> None:
        y = self.height - 116
        self.rect(40, y, 1120, 88, radius=12)
        self.text(60, y + 27, "EXPECTED RESULT", 17, TEAL, 700)
        for i, line in enumerate(textwrap.wrap(value, 110)):
            self.text(60, y + 53 + i * 25, line, 20)

    def matrix(self, x: float, y: float, rows: list[list[str]], headers: list[str],
               widths: list[int], highlight: int | None = None) -> None:
        total = sum(widths)
        self.rect(x, y, total, (len(rows) + 1) * 44, radius=10)
        self.rect(x, y, total, 44, fill="#EAF0F7", radius=10)
        for r, row in enumerate([headers, *rows]):
            if r and r - 1 == highlight:
                self.rect(x + 1, y + r * 44, total - 2, 44, fill="#FFF1E5", stroke="none", radius=0)
            left = x
            for i, value in enumerate(row):
                self.text(left + 16, y + r * 44 + 29, value, 19, weight=600 if r == 0 else 400)
                left += widths[i]

    def dot(self, x: float, y: float, color: str = BLUE, radius: float = 5) -> None:
        self.parts.append(f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" fill="{color}"/>')

    def plot(self, x: float, y: float, w: float, h: float,
             series: list[tuple[list[float], str, bool]], minimum: float, maximum: float,
             xlabel: str, ylabel: str) -> None:
        for fraction in (0., .5, 1.):
            yy = y + h * (1 - fraction)
            self.line(x, yy, x + w, yy, color="#DDE5EE", width=1)
            self.text(x - 12, yy + 6, f"{minimum + fraction * (maximum - minimum):.2g}", 16, MUTED, anchor="end")
        self.line(x, y, x, y + h, color=MUTED)
        self.line(x, y + h, x + w, y + h, color=MUTED)
        self.text(x + w / 2, y + h + 36, xlabel, 18, MUTED, anchor="middle")
        self.text(x, y - 18, ylabel, 18, MUTED)
        for values, color, dashed in series:
            points = [(x + i * w / max(1, len(values) - 1), y + h * (1 - (v - minimum) / (maximum - minimum)))
                      for i, v in enumerate(values)]
            coords = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in points)
            dash = ' stroke-dasharray="8 5"' if dashed else ""
            self.parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                              f'stroke-width="3" stroke-linejoin="round"{dash}/>')
            if len(points) <= 12:
                for xx, yy in points:
                    if dashed:
                        self.rect(xx - 4, yy - 4, 8, 8, fill="white", stroke=color, radius=1)
                    else:
                        self.dot(xx, yy, color, 4)

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{self.name}.svg").write_text("\n".join([*self.parts, "</svg>"]) + "\n", encoding="utf-8")


def generators() -> None:
    f = Lesson("generators", "A generator turns choices into complete recipes",
               "Two preprocessing choices × three PLS component counts = six separate experiments", 840)
    f.card(40, 135, 332, 165, "Preprocessing choices", ["SNV", "StandardScaler"], number="A")
    f.card(432, 135, 330, 165, "Component choices", ["PLS: 2, 3 or 4 components", "Try every value for each choice"], PURPLE, "B")
    f.line(380, 218, 420, 218, True)
    f.card(822, 135, 338, 165, "Expand", ["2 × 3 = 6 complete recipes", "Keep data and folds fixed"], TEAL, "C")
    f.line(770, 218, 810, 218, True)
    f.text(44, 348, "Each card below is one pipeline. Fit and compare all six.", 23, weight=600)
    for row, preprocessing in enumerate(["SNV", "StandardScaler"]):
        for col, components in enumerate([2, 3, 4]):
            f.card(40 + col * 380, 380 + row * 155, 360, 131, f"Recipe {row * 3 + col + 1}",
                   [f"{preprocessing} → PLS({components})", "Fit on the same training folds"], BLUE if row == 0 else PURPLE)
    f.note("Six evaluated recipes. Compare validation error, choose one recipe, then refit that recipe.")
    f.save()


def workflow() -> None:
    f = Lesson("workflow", "From measurements to a prediction you can use",
               "Follow one complete experiment before adding tuning, branches or more modalities")
    labels = ["Load", "Split", "Compare", "Refit", "Predict"]
    details = [["Spectra + lab targets", "Same sample IDs"], ["Train / validate", "Keep groups together"],
               ["Try recipes", "Measure held-out error"], ["Fit chosen recipe", "Save learned state"], ["Fresh spectra", "Return target units"]]
    for i, (label, lines) in enumerate(zip(labels, details, strict=True)):
        x = 40 + i * 228
        f.card(x, 178, 208, 195, label, lines, BLUE if i < 3 else TEAL, str(i + 1))
        if i < 4:
            f.line(x + 212, 263, x + 221, 263, True)
    f.card(40, 420, 520, 130, "Laboratory question", ["What is the concentration in this sample?"], PURPLE)
    f.card(600, 420, 560, 130, "Statistical question", ["How wrong will I be on genuinely new samples?"], ORANGE)
    f.note("The fitted predictor contains the preprocessing and feature order needed to reproduce predictions.")
    f.save()


def datasets() -> None:
    f = Lesson("datasets", "Join measurements by sample ID",
               "Files may have different row orders. IDs tell you which observations belong together.")
    f.text(40, 150, "Two input files", 23, weight=700)
    f.matrix(40, 172, [["A", "0.42"], ["B", "0.57"], ["C", "0.31"]], ["Spectrum ID", "Peak"], [170, 120])
    f.matrix(360, 172, [["C", "12"], ["A", "18"], ["B", "25"]], ["Lab ID", "Target"], [145, 125])
    f.line(650, 265, 742, 265, True)
    f.text(668, 235, "join by ID", 18, TEAL)
    f.matrix(770, 172, [["A", "0.42", "18"], ["B", "0.57", "25"], ["C", "0.31", "12"]],
             ["ID", "Peak", "Target"], [100, 145, 145])
    f.card(40, 407, 536, 145, "Position alone gives wrong labels", ["First row A is not first row C.", "A positional join would give A target 12."], ORANGE)
    f.card(614, 407, 546, 145, "Several modalities, one sample", ["A spectrum, image and metadata describe A.", "Keep IDs and presence information together."], TEAL)
    f.note("A has target 18, B has target 25 and C has target 12, regardless of file order.")
    f.save()


def branching() -> None:
    for name, second, width, title in [
        ("branch", "Savitzky–Golay smoothing", 31, "Keep two processing paths side by side"),
        ("concat_transform", "PCA: keep 3 components", 3, "Join different representations of the same samples"),
    ]:
        f = Lesson(name, title, "Sample rows stay aligned. Each path creates its own feature columns.")
        f.card(40, 251, 250, 157, "Raw input", ["48 samples", "31 spectral features"])
        f.card(380, 155, 366, 135, "Path 1 · SNV", ["48 samples × 31 features"])
        f.card(380, 360, 366, 145, "Path 2", [second, f"48 samples × {width} features"], PURPLE)
        f.line(294, 326, 370, 230, True)
        f.line(294, 332, 370, 430, True)
        f.card(858, 251, 302, 157, "Join columns", [f"31 + {width} = {31 + width} features", f"Output: 48 × {31 + width}"], TEAL)
        f.line(752, 230, 847, 318, True)
        f.line(752, 430, 847, 342, True)
        f.note(f"48 rows are preserved. The final model receives {31 + width} columns per sample.")
        f.save()


def fusion() -> None:
    f = Lesson("fusion", "Two ways to combine modalities",
               "Early fusion combines measurements. Late fusion combines source-specific model predictions.", 820)
    f.text(44, 157, "EARLY · join features before fitting", 21, BLUE, 700)
    f.card(40, 180, 330, 126, "Spectral features", ["48 samples × 31 columns"])
    f.card(40, 329, 330, 126, "Marker features", ["48 samples × 3 columns"], PURPLE)
    f.card(447, 246, 303, 151, "Concatenate", ["31 + 3 = 34 features", "48 aligned rows"], TEAL)
    f.card(830, 246, 330, 151, "Fit one model", ["Use both measurements", "Predict one target per row"], TEAL)
    f.line(376, 243, 438, 302, True)
    f.line(376, 390, 438, 334, True)
    f.line(756, 321, 820, 321, True)
    f.text(44, 505, "LATE · learn from held-out source predictions", 21, PURPLE, 700)
    f.chip(40, 535, 330, "Spectral model → OOF prediction")
    f.chip(40, 598, 330, "Marker model → OOF prediction", PURPLE)
    f.card(447, 529, 303, 121, "Join predictions", ["48 × 2 prediction columns"], PURPLE)
    f.card(830, 529, 330, 121, "Fit meta-model", ["Learn a prediction combination"], TEAL)
    f.line(376, 555, 438, 575, True)
    f.line(376, 617, 438, 599, True)
    f.line(756, 589, 820, 589, True)
    f.note("Early: 34 feature columns. Late: 2 held-out prediction columns for the combining model.")
    f.save()

    f = Lesson("merge", "Choose a merge that matches your branch outputs",
               "Features, predictions, sources and routed rows describe different kinds of joins.", 820)
    entries = [("Features", "Same rows; different transformed features", "48 × 31 + 48 × 31 → 48 × 62", BLUE),
               ("Predictions", "One held-out prediction per base model", "48 × 1 + 48 × 1 → 48 × 2", PURPLE),
               ("Sources", "Different sensors for the same samples", "48 × 31 + 48 × 3 → 48 × 34", TEAL),
               ("Reassembly", "Subsets routed through different branches", "20 rows + 28 rows → 48 original IDs", ORANGE)]
    for i, (label, meaning, result, color) in enumerate(entries):
        y = 143 + i * 133
        f.card(40, y, 247, 111, label, [], color, str(i + 1))
        f.text(320, y + 34, meaning, 21)
        f.line(323, y + 60, 390, y + 60, True)
        f.text(422, y + 67, result, 24, color, 600)
    f.note("Feature/source joins add columns. Reassembly restores rows. Prediction joins create meta-features.")
    f.save()


def preprocessing() -> None:
    f = Lesson("preprocessing_node", "SNV removes each spectrum's offset and scale",
               "Two spectra with the same shape become identical after row-wise normalization.")
    f.card(40, 153, 350, 210, "Before · two spectra", ["A: [1, 2, 3]", "B: [3, 5, 7]", "Different offset and amplitude"])
    f.card(445, 175, 280, 166, "For each row", ["Subtract its mean", "Divide by std (ddof=0)"], PURPLE)
    f.card(785, 153, 375, 210, "After · same shape", ["A: [−1.225, 0, 1.225]", "B: [−1.225, 0, 1.225]", "Still 2 samples × 3 features"], TEAL)
    f.line(396, 260, 435, 260, True)
    f.line(731, 260, 775, 260, True)
    f.plot(91, 430, 390, 104, [([1, 2, 3], BLUE, False), ([3, 5, 7], PURPLE, True)], 0, 8, "Feature 1 → feature 3", "Input signal")
    values = [-math.sqrt(1.5), 0., math.sqrt(1.5)]
    f.plot(744, 430, 390, 104, [(values, BLUE, False), (values, PURPLE, True)], -1.5, 1.5, "Feature 1 → feature 3", "Normalized signal")
    f.note("The curves overlap. SNV changes feature values; it does not add samples or feature columns.")
    f.save()

    f = Lesson("y_processing", "Fit in transformed target units; predict in original units",
               "The fitted target transformation is inverted after the model makes a prediction.")
    f.card(40, 180, 342, 174, "Original target", ["Concentrations: 10, 20, 30", "Training range: 10 to 30"])
    f.card(434, 180, 326, 174, "Fit MinMaxScaler", ["Map range to [0, 1]", "Model sees: 0, 0.5, 1"], PURPLE)
    f.card(812, 180, 348, 174, "Model output", ["Predicts 0.6", "Still in transformed units"], TEAL)
    f.line(389, 265, 424, 265, True)
    f.line(767, 265, 802, 265, True)
    f.rect(174, 406, 852, 133, shadow=True)
    f.text(600, 449, "Inverse transform the prediction", 23, TEAL, 700, "middle")
    f.text(600, 496, "10 + 0.6 × (30 − 10) = 22", 32, weight=600, anchor="middle")
    f.note("The user receives 22 in original target units. Reuse the fitted training transformation.")
    f.save()


def splitting() -> None:
    f = Lesson("split", "Three folds: every sample is held out once",
               "12 independent samples: each model trains on 8 and predicts the other 4.")
    f.text(43, 161, "Sample", 19, MUTED)
    for i in range(12):
        f.text(225 + i * 76, 161, str(i + 1), 19, anchor="middle")
    for fold in range(3):
        y = 195 + fold * 106
        f.text(44, y + 39, f"Fold {fold + 1}", 24, weight=600)
        for i in range(12):
            held = i // 4 == fold
            f.rect(191 + i * 76, y, 68, 60, fill="#FFF0E4" if held else "#E4F1EF", stroke=ORANGE if held else TEAL, radius=8)
            f.text(225 + i * 76, y + 37, "V" if held else "T", 21, ORANGE if held else TEAL, 700, "middle")
    f.chip(191, 532, 330, "T = train / fit this fold", TEAL)
    f.chip(553, 532, 350, "V = held out / score this fold", ORANGE)
    f.note("12 out-of-fold predictions. Each prediction comes from a model that did not fit that sample.")
    f.save()

    f = Lesson("evaluation", "Split specimens, not their repeated scans",
               "Repeated measurements of the same physical specimen stay together during validation.")
    for x, good in [(40, False), (627, True)]:
        f.card(x, 155, 533, 126, "Whole specimens held out" if good else "Same specimens on both sides", [], TEAL if good else ORANGE)
        f.text(x + 25, 323, "TRAIN", 21, TEAL, 700)
        f.text(x + 300, 323, "VALIDATE", 21, ORANGE, 700)
        train = ["A · scan 1", "B · scan 1", "C · scan 1"] if not good else ["A · scans 1 + 2", "B · scans 1 + 2"]
        valid = ["A · scan 2", "B · scan 2", "C · scan 2"] if not good else ["C · scans 1 + 2"]
        for i, label in enumerate(train):
            f.chip(x + 20, 345 + i * 57, 232, label, TEAL)
        for i, label in enumerate(valid):
            f.chip(x + 293, 345 + i * 57, 224, label, ORANGE)
        f.line(x + 272, 330, x + 272, 524, color="#A8B6C5", dashed=True)
    f.note("On the right, validation measures predictions for specimen C, which the model has never seen.")
    f.save()


def flags() -> None:
    rows = [["A", "10", "keep"], ["B", "11", "keep"], ["C", "12", "keep"], ["D", "13", "keep"], ["E", "14", "keep"], ["F", "50", "flag"]]
    for name, exclude in [("tag", False), ("exclude", True)]:
        f = Lesson(name, "Exclude a flagged training sample" if exclude else "Tag a sample without removing it",
                   "An IQR criterion flags one unusually high target value in this teaching dataset.")
        f.matrix(40, 167, rows, ["Sample", "Target", "IQR flag"], [125, 120, 125], highlight=5)
        f.card(475, 177, 685, 163, "Criterion", ["Q1 = 11.25 · Q3 = 13.75 · IQR = 2.5", "Bounds: 7.5 to 17.5; sample F lies above the upper bound"], PURPLE)
        if exclude:
            f.card(475, 375, 685, 165, "After exclude", ["Fit uses A, B, C, D, E: 5 training rows", "Original IDs remain available for reporting", "Future prediction rows are not removed"], TEAL)
            f.note("Training shrinks from 6 to 5 rows. Record why F was excluded before comparing models.")
        else:
            f.card(475, 375, 685, 165, "After tag", ["Fit still uses A, B, C, D, E, F: 6 training rows", "F carries a flag for analysis or routing", "The tag is information, not an exclusion"], TEAL)
            f.note("All 6 rows remain. F receives an outlier tag so its behavior can be inspected separately.")
        f.save()


def augmentation() -> None:
    f = Lesson("sample_augmentation", "Add training variations; keep validation unchanged",
               "Two generated versions of each training observation share its original sample identity.")
    f.card(40, 175, 333, 157, "Before", ["8 original training rows", "4 held-out validation rows"])
    f.card(432, 175, 335, 157, "Create two variations", ["8 × 2 = 16 generated rows", "Keep the 8 original rows"], PURPLE)
    f.card(826, 175, 334, 157, "After", ["24 rows used for fitting", "Still 4 validation rows"], TEAL)
    f.line(381, 253, 422, 253, True)
    f.line(775, 253, 816, 253, True)
    f.text(44, 399, "One original sample and its copies stay in the same training fold", 23, weight=600)
    for i, label in enumerate(["A · original", "A · noisy version 1", "A · noisy version 2"]):
        f.chip(40 + i * 380, 432, 348, label, BLUE if i == 0 else PURPLE)
    f.text(44, 520, "Three rows; one independent origin. The label relationship must remain valid.", 21, MUTED)
    f.note("More training rows, not more independent evidence. No synthetic copies appear during prediction.")
    f.save()

    f = Lesson("feature_augmentation", "Keep several feature views of every spectrum",
               "Three views describe the same 48 samples. This does not create new samples.")
    f.card(40, 255, 265, 142, "Input spectra", ["48 samples × 31 features"])
    for i, (label, color) in enumerate([("Original", BLUE), ("SNV", TEAL), ("First derivative", PURPLE)]):
        y = 150 + i * 145
        f.card(391, y, 353, 115, label, ["48 samples × 31 features"], color)
        f.line(312, 325, 381, y + 55, True)
        f.line(750, y + 55, 822, 325, True)
    f.card(833, 255, 327, 142, "Tracked views", ["Shape: 48 × 3 × 31", "sample × view × feature"], TEAL)
    f.note("48 samples remain. Flattening the three views would produce 93 feature columns per sample.")
    f.save()


def repetitions() -> None:
    f = Lesson("repetitions", "Four repeated scans belong to one physical sample",
               "Reshape 120 scans into 30 sample rows while preserving their repetition relation.")
    f.card(40, 248, 283, 186, "Input", ["30 physical samples", "4 scans per sample", "120 scans × 31 features"])
    f.card(422, 162, 738, 156, "rep_to_sources · one source for each repeat", ["Sources 0, 1, 2, 3 each have shape 30 × 31", "Useful for source-local processing or fusion"], TEAL)
    f.card(422, 377, 738, 156, "rep_to_pp · one source with four views", ["Shape: 30 samples × 4 repetitions × 31 features", "Useful for tracking repeats as feature views"], PURPLE)
    f.line(331, 315, 411, 237, True)
    f.line(331, 366, 411, 451, True)
    f.note("The sample axis has 30 physical samples. Do not evaluate their repeated scans as independent observations.")
    f.save()


def models() -> None:
    f = Lesson("model", "A model learns to predict the laboratory target",
               "Compare predictions with held-out measurements to see error magnitude and direction.")
    for i, (label, lines, color) in enumerate([
        ("Training", ["Input features X + target y", "Fit learned model state"], BLUE),
        ("Prediction", ["New X → fitted model", "One prediction per sample"], PURPLE),
        ("Evaluation", ["Prediction minus observation", "Inspect the signed residual"], TEAL),
    ]):
        x = 40 + i * 399
        f.card(x, 155, 321, 165, label, lines, color)
        if i < 2:
            f.line(x + 328, 239, x + 388, 239, True)
    f.matrix(220, 378, [["A", "5.0", "4.8", "−0.2"], ["B", "8.0", "8.2", "+0.2"], ["C", "11.0", "10.9", "−0.1"]],
             ["Sample", "Observed", "Predicted", "Residual"], [160, 200, 200, 200])
    f.note("Three predictions in target units. Illustrative RMSE = √[(0.04 + 0.04 + 0.01) / 3] ≈ 0.173.")
    f.save()

    f = Lesson("results", "Read the errors, not just the best score",
               "Illustrative observations: 5, 8, 11. Predictions: 4.8, 8.2, 10.9.")
    f.card(40, 147, 540, 386, "Observed versus predicted", [])
    f.card(619, 147, 541, 386, "Signed residuals", [], PURPLE)
    left, top, width, height = 110., 228., 403., 210.
    f.line(left, top, left, top + height, color=MUTED)
    f.line(left, top + height, left + width, top + height, color=MUTED)
    f.line(left, top + height, left + width, top, color=MUTED, dashed=True)
    for value in [4, 8, 12]:
        tick_x = left + (value - 4) / 8 * width
        tick_y = top + height - (value - 4) / 8 * height
        f.text(tick_x, top + height + 24, str(value), 16, MUTED, anchor="middle")
        f.text(left - 12, tick_y + 6, str(value), 16, MUTED, anchor="end")
    for sample, observed, predicted in zip(["A", "B", "C"], [5., 8., 11.], [4.8, 8.2, 10.9], strict=True):
        point_x = left + (observed - 4) / 8 * width
        point_y = top + height - (predicted - 4) / 8 * height
        f.dot(point_x, point_y, radius=7)
        f.text(point_x + 13, point_y - 8, sample, 21, weight=700)
    f.text(left + width / 2, 483, "Observed concentration", 20, MUTED, anchor="middle")
    f.text(left, 211, "Predicted concentration", 20, MUTED)
    f.text(320, 510, "Dashed line: predicted = observed", 17, MUTED, anchor="middle")
    zero = 341
    f.line(680, zero, 1112, zero, color=MUTED, dashed=True)
    f.text(668, zero + 6, "0", 18, MUTED, anchor="end")
    for i, (sample, residual) in enumerate(zip(["A", "B", "C"], [-.2, .2, -.1], strict=True)):
        x, endpoint = 740 + i * 145, zero - residual * 440
        f.rect(x - 22, min(zero, endpoint), 44, abs(endpoint - zero), fill=PURPLE, stroke=PURPLE, radius=4)
        f.text(x, endpoint - 12 if residual > 0 else endpoint + 27, f"{residual:+.1f}", 21, anchor="middle")
        f.text(x, 483, sample, 21, weight=600, anchor="middle")
    f.note("A and C are slightly underestimated; B is slightly overestimated. Aggregate RMSE is about 0.173.")
    f.save()

    f = Lesson("residual", "A residual learner adds a correction to a base prediction",
               "Use held-out base predictions to learn what the first model has not explained.")
    f.card(40, 172, 328, 180, "Base model", ["Base prediction: 10", "Its OOF errors become", "the learner's targets"])
    f.card(436, 172, 328, 180, "Residual learner", ["Predicted correction: +2", "Weight λ = 0.5", "Gate = 1 in this illustration"], PURPLE)
    f.card(832, 172, 328, 180, "Combined prediction", ["10 + 0.5 × 1 × 2", "Final prediction: 11"], TEAL)
    f.line(375, 255, 426, 255, True)
    f.line(771, 255, 822, 255, True)
    f.card(132, 421, 936, 116, "Why the correction has its own weight", ["A perfect training correction may be too strong on new samples."], PURPLE)
    f.note("The full predictor returns 11, not the base output 10 or the learner output 2 by itself.")
    f.save()


def deployment_and_operators() -> None:
    f = Lesson("deployment", "Save fitted state, then predict fresh samples",
               "A recipe describes what to fit. An artifact contains the learned values needed for prediction.")
    f.card(40, 156, 530, 171, "Recipe · instructions", ["Use MSC, then a Ridge model", "Parameters describe how training should run", "No fitted reference or model coefficients"])
    f.card(630, 156, 530, 171, "Artifact · learned state", ["The training MSC reference spectrum", "The fitted model coefficients", "Feature order, source schema and target units"], TEAL)
    f.line(578, 237, 620, 237, True)
    f.text(584, 210, "FIT", 18, MUTED)
    for i, (label, lines, color) in enumerate([
        ("Fresh measurements", ["Same source and feature schema", "No target labels needed"], BLUE),
        ("LOAD → PREDICT", ["Reuse saved preprocessing", "Reuse saved model state"], PURPLE),
        ("Predictions", ["One output per sample ID", "In original target units"], TEAL),
    ]):
        x = 40 + i * 395
        f.card(x, 396, 331, 154, label, lines, color)
        if i < 2:
            f.line(x + 339, 468, x + 384, 468, True)
    f.note("Loading the predictor does not run a new experiment or choose a new preprocessing recipe.")
    f.save()

    f = Lesson("operator_step", "Each operator performs one operation",
               "The class identifies the operation; parameters control what it does to the data.")
    for i, (label, lines, color) in enumerate([
        ("StandardScaler", ["48 × 31 → 48 × 31", "Learn column means and scales"], BLUE),
        ("PCA(3)", ["48 × 31 → 48 × 3", "Learn three feature directions"], PURPLE),
        ("PLS model", ["48 × 3 → 48 × 1", "Learn the target relationship"], TEAL),
    ]):
        x = 40 + i * 396
        f.card(x, 198, 328, 178, label, lines, color, str(i + 1))
        if i < 2:
            f.line(x + 337, 285, x + 382, 285, True)
    f.card(40, 433, 1120, 114, "Fit learned operations using training rows only", ["Prediction reuses the saved means, feature directions and model."], ORANGE)
    f.note("48 input rows remain. Feature width changes from 31 to 3 before one target prediction per row.")
    f.save()

    f = Lesson("auto_transfer_preproc", "Choose preprocessing for instrument transfer",
               "Compare source measurements with an explicit target-instrument adaptation cohort.")
    f.card(40, 158, 371, 146, "Source instrument", ["40 observations × 31 features", "Original calibration cohort"])
    f.card(40, 365, 371, 146, "Target instrument", ["8 observations × 31 features", "Explicit adaptation cohort"], PURPLE)
    f.card(483, 254, 306, 179, "Selector", ["Compare candidate recipes", "Measure transfer criterion", "Choose a recommendation"], TEAL)
    f.card(865, 254, 295, 179, "Selected transform", ["One recommendation", "Then fit the predictor", "Validate unseen target rows"], TEAL)
    f.line(418, 237, 473, 314, True)
    f.line(418, 436, 473, 367, True)
    f.line(796, 335, 854, 335, True)
    f.note("One preprocessing recommendation. It does not guarantee better target-instrument predictions.")
    f.save()


def diagnostics() -> None:
    f = Lesson("charts", "Inspect the data before and after preprocessing",
               "Expected output types: two PCA projections plus a fold diagram. Coordinates here are schematic.", 840)
    for offset, after in [(40, False), (628, True)]:
        f.card(offset, 155, 532, 364, "PCA after SNV" if after else "PCA of the raw features", [], PURPLE if after else BLUE)
        left, top = offset + 70, 255
        f.line(left, top, left, top + 173, color=MUTED)
        f.line(left, top + 173, left + 390, top + 173, color=MUTED)
        f.text(left + 195, 474, "Principal component 1", 19, MUTED, anchor="middle")
        f.text(left, 230, "Principal component 2", 19, MUTED)
        coords = [(-.5, -.8), (-.3, .3), (.2, .8), (.7, -.2), (.1, .1), (-.5, -.1)] if after else [(-1.8, -.3), (-1.3, .1), (-.4, .4), (.5, -.5), (1.5, .2), (2., 0.)]
        for sample, (px, py) in zip("ABCDEF", coords, strict=True):
            point_x, point_y = left + (px + 2.3) / 4.8 * 390, top + (1 - (py + 1.2) / 2.4) * 173
            f.dot(point_x, point_y, PURPLE if after else BLUE, 6)
            f.text(point_x + 10, point_y - 10, sample, 18, weight=600)
    f.text(43, 565, "The same sample IDs remain. Preprocessing changes their projected coordinates.", 22, weight=600)
    f.text(43, 624, "Fold 1", 22, weight=600)
    for i in range(12):
        held = i < 4
        f.chip(185 + i * 79, 594, 68, "V" if held else "T", ORANGE if held else TEAL)
    f.text(185, 668, "12 independent rows: 4 validation (V), 8 training (T). Repeat for folds 2 and 3.", 20, MUTED)
    f.note("Saved diagnostic figures show the named processing stage and actual folds; they do not add model features.")
    f.save()

    f = Lesson("model_selection", "Choose model complexity using validation error",
               "Illustrative capacity curve: more components help training here, but not new observations.")
    train = [.9 * math.exp(-n / 3) + .08 for n in range(1, 13)]
    valid = [.6 * math.exp(-n / 2) + .035 * max(n - 4, 0) + .18 for n in range(1, 13)]
    f.plot(98, 193, 642, 327, [(train, BLUE, False), (valid, ORANGE, True)], 0, .8,
           "PLS components: 1 → 12", "RMSE · synthetic target units")
    f.card(813, 202, 347, 170, "Validation minimum", ["4 components", "Keep the independent test", "population untouched"], TEAL)
    f.chip(813, 407, 347, "Solid + circles: training")
    f.chip(813, 466, 347, "Dashed + squares: validation", ORANGE)
    f.note("Select 4 components from validation, even though training error continues to decrease.")
    f.save()

    f = Lesson("pooled_rmse", "Pooling errors differs from averaging fold RMSE",
               "Explicit data: fold 1 residuals [1, 1]; fold 2 residuals [3, 3].", 690)
    f.matrix(40, 156, [["Fold 1", "1, 1", "1"], ["Fold 2", "3, 3", "3"]], ["Held-out fold", "Residuals", "RMSE"], [210, 190, 145])
    f.card(40, 344, 545, 164, "Average fold RMSE", ["(1 + 3) / 2 = 2.000", "Each fold RMSE receives equal weight"])
    f.card(637, 156, 523, 352, "Pool all squared residuals", ["Residuals: 1, 1, 3, 3", "Squared errors: 1, 1, 9, 9", "Mean squared error: 20 / 4 = 5", "Pooled RMSE: √5 ≈ 2.236"], PURPLE)
    f.note("Mean fold RMSE = 2.000. Pooled RMSE = 2.236. State which aggregation is reported.")
    f.save()

    f = Lesson("search_budget", "Count the experiments before starting a search",
               "Independent choices multiply the number of complete candidate recipes.")
    for i, (label, count, color) in enumerate([("Preprocessing", "3 choices", BLUE), ("Components", "4 values", PURPLE), ("Seeds", "2 values", TEAL)]):
        f.card(40 + i * 396, 167, 328, 154, label, [count], color)
        if i < 2:
            f.text(402 + i * 396, 255, "×", 38, MUTED, anchor="middle")
    f.card(40, 376, 530, 157, "24 candidate recipes", ["3 × 4 × 2 = 24", "Each has its own fitted state"])
    f.card(630, 376, 530, 157, "120 model fits · five folds", ["24 × 5 = 120 evaluation fits", "Winner refit and inner tuning add work"], TEAL)
    f.note("Even this small search needs 120 evaluation fits. Start with a small space and a baseline.")
    f.save()


def ragged_series() -> None:
    """Demonstrate exact population summary statistics for variable-length rows."""
    f = Lesson("ragged_series", "Summarize different-length series into the same columns",
               "SequenceSummary keeps one row per sample; statistics are computed separately within each series.")
    f.matrix(40, 164, [["A", "[1, 3]", "2"], ["B", "[2, 4, 6]", "3"],
                       ["C", "[0, 2, 4, 6]", "4"]], ["Sample", "Channel values", "Length"], [120, 260, 120])
    f.card(642, 164, 518, 199, "One channel → five summary columns", ["Mean, std (ddof=0), min, max, length", "Time coordinates are not used", "Order and dynamics are not retained"], PURPLE)
    f.line(548, 256, 631, 256, True)
    f.matrix(132, 410, [["A", "2", "1", "1", "3", "2"],
                       ["B", "4", "1.633", "2", "6", "3"],
                       ["C", "3", "2.236", "0", "6", "4"]],
             ["Sample", "Mean", "Std", "Min", "Max", "Length"], [145, 155, 155, 145, 145, 180])
    f.note("Three variable-length series become a numeric 3 × 5 matrix. Channels must keep their identity.")
    f.save()


def main() -> None:
    """Regenerate all illustrations, preserving legacy figure filenames."""
    for render in [generators, workflow, datasets, branching, fusion, preprocessing,
                   splitting, flags, augmentation, repetitions, models, deployment_and_operators, diagnostics, ragged_series]:
        render()
    for old, new in [("preprocessing", "preprocessing_node"), ("augmentation", "sample_augmentation")]:
        (OUT / f"{old}.svg").write_bytes((OUT / f"{new}.svg").read_bytes())
    print(f"Rendered {len(list(OUT.glob('*.svg')))} annotated SVG lessons in {OUT}")


if __name__ == "__main__":
    main()
