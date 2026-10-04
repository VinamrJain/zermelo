"""One launch written for the paper: a CSV per curve per acquisition, a pgfplots figure reading them, and a booktabs table

Every curve is a median over seeds inside its interquartile band, at every x; every table cell is mean +- standard error over seeds.
"""

import shutil
import subprocess
import sys
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

STYLE = Path("../paper/manuscript/figures/configs/style.tex")
"""The paper's shared pgfplots style, copied beside the figure so it compiles where it is written"""

FAMILY_COLOURS = {
    "EMI": "0173B2",
    "ESI": "D55E00",
    "EVI": "029E73",
    "EI": "DE8F05",
    "MC-EI": "56B4E9",
    "UCB": "7F3C8D",
    "TS": "CC78BC",
    "MaxVar": "CA9161",
    "RandTarget": "555555",
    "RandAct": "A0A0A0",
    "Oracle-EMI": "000000",
    "Forecast": "000000",
}
"""A hex colour per acquisition family, defined in every figure under the family's own name; a table lists the families in this order"""

OURS = ("EMI", "ESI", "EVI")
"""The families a table rules off above the rest"""

FALLBACK_COLOURS = (
    "0173B2",
    "DE8F05",
    "029E73",
    "D55E00",
    "CC78BC",
    "CA9161",
    "949494",
    "ECE133",
    "56B4E9",
    "222222",
    "7F3C8D",
    "11A579",
    "E73F74",
    "80BA5A",
    "E68310",
    "008695",
    "CF1C90",
    "F2B701",
)
"""Hex colours given one per acquisition where an acquisition's family has no colour of its own"""

VARIANT_STYLES = (
    "mark=*, mark size=1.8pt, mark repeat=12, mark phase=1",
    "mark=square*, mark size=1.8pt, mark repeat=12, mark phase=4",
    "mark=triangle*, mark size=2.2pt, mark repeat=12, mark phase=7",
    "mark=diamond*, mark size=2.2pt, mark repeat=12, mark phase=10",
    "dashed, mark=o, mark size=1.8pt, mark repeat=12, mark phase=2, mark options={solid}",
    "dashed, mark=square, mark size=1.8pt, mark repeat=12, mark phase=5, mark options={solid}",
    "dashed, mark=triangle, mark size=2.2pt, mark repeat=12, mark phase=8, mark options={solid}",
    "dashed, mark=x, mark size=2.4pt, mark repeat=12, mark phase=11, mark options={solid}",
)
"""Line styles given one per acquisition in the order drawn; a family with more members than styles is coloured one by one instead"""

ROWS_PER_CURVE = 100
"""Most x values one CSV holds; a longer curve is thinned to every k-th x with its last x kept"""


def family(label: str) -> str:
    """The acquisition family of a paper label: `EMI-L25` -> `EMI`, `ESI-L8 (wind)` -> `ESI`, `MC-EI charged` -> `MC-EI`"""
    known = [f for f in FAMILY_COLOURS if label == f or label.startswith((f + "-", f + " "))]
    return max(known, key=len) if known else label.split("-")[0].split(" ")[0]


def _slug(label: str) -> str:
    """`label` as a file name: lower case, every run of other characters one dash"""
    return "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")


def _thinned(x: np.ndarray) -> np.ndarray:
    """The x values a CSV keeps: every k-th of `x` and its last, at most `ROWS_PER_CURVE` of them"""
    step = max(1, int(np.ceil(x.size / ROWS_PER_CURVE)))
    return np.union1d(x[::step], x[-1:])


def write_curves(directory: Path, curves: pd.DataFrame, panels: Sequence[str], labels: dict[str, str]) -> dict[tuple[str, str], Path]:
    """CSVs `x,med,q25,q75` under `directory`, one per (panel, acquisition), from `curves` holding `acquisition, seed, x` and the panels"""
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[tuple[str, str], Path] = {}
    for acquisition, held in curves.groupby("acquisition"):
        kept = held[held["x"].isin(_thinned(np.sort(held["x"].unique())))]
        by_x = kept.groupby("x")
        for panel in panels:
            table = by_x[panel].quantile(np.array([0.5, 0.25, 0.75])).unstack().set_axis(["med", "q25", "q75"], axis=1).reset_index()
            path = directory / f"{panel}_{_slug(labels[str(acquisition)])}.csv"
            table.to_csv(path, index=False, float_format="%.6g")
            written[(panel, str(acquisition))] = path
    return written


def write_figure(
    path: Path,
    csvs: dict[tuple[str, str], Path],
    panels: Sequence[tuple[str, str, str]],
    acquisitions: Sequence[str],
    labels: dict[str, str],
    x_label: str,
    opening_x: float | None,
    omitted: Collection[tuple[str, str]],
) -> None:
    """A standalone pgfplots group over `panels` as (column, y label, `linear` or `log`), one band per acquisition, legend below

    Acquisitions of one family share its colour, every acquisition has its own line style, and a pair (column, acquisition) in `omitted` is not drawn.
    A dotted rule at `opening_x` marks the shared opening.
    """
    families = [family(labels[a]) for a in acquisitions]
    coloured = all(f in FAMILY_COLOURS for f in families) and max(families.count(f) for f in families) <= len(VARIANT_STYLES)
    colour = {a: families[i].replace("-", "") if coloured else f"acq{chr(65 + i)}" for i, a in enumerate(acquisitions)}
    # one colour per family and a style per acquisition; past the styles, a colour per acquisition and a style per round of colours
    style = {a: VARIANT_STYLES[(i if coloured else i // len(FALLBACK_COLOURS)) % len(VARIANT_STYLES)] for i, a in enumerate(acquisitions)}
    defined = (
        [rf"\definecolor{{{f.replace('-', '')}}}{{HTML}}{{{FAMILY_COLOURS[f]}}}" for f in dict.fromkeys(families)]
        if coloured
        else [rf"\definecolor{{{colour[a]}}}{{HTML}}{{{FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)]}}}" for i, a in enumerate(acquisitions)]
    )
    columns = 2 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels) / columns))
    lines = [
        r"\documentclass[tikz, 11pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\input{configs/style}",
        *defined,
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        rf"    group style={{group size={columns} by {rows}, horizontal sep=54pt, vertical sep=36pt}},",
        r"    panel,",
        r"  ]",
    ]
    for i, (column, y_label, scale) in enumerate(panels):
        bottom = i >= len(panels) - columns
        options = [f"ylabel={{{y_label}}}"] + ([f"xlabel={{{x_label}}}"] if bottom else []) + (["ymode=log"] if scale == "log" else [])
        if i == 0:  # the legend is collected from the first panel alone, every panel holding the same acquisitions
            legend_rows = int(np.ceil(len(acquisitions) / 5))
            options.append(f"legend to name=legend:{path.stem}, legend columns={int(np.ceil(len(acquisitions) / legend_rows))}")
        lines.append(rf"  \nextgroupplot[{', '.join(options)}]")
        for a in acquisitions:
            if (column, a) in omitted:
                continue
            lines.append(
                rf"    \band[every axis plot post/.style={{}}, {style[a]}]{{{colour[a]}}}{{{csvs[(column, a)].relative_to(path.parent).as_posix()}}}"
            )
        if opening_x is not None:
            lines.append(rf"    \opening{{{opening_x:g}}}")
        if i == 0:
            lines.append(r"    \legend{" + ", ".join(f"{{{labels[a]}}}" for a in acquisitions) + "}")
    lines += [
        r"\end{groupplot}",
        rf"\node[anchor=north, yshift=-28pt] at ($(group c1r{rows}.south)!0.5!(group c{columns}r{rows}.south)$) {{\pgfplotslegendfromname{{legend:{path.stem}}}}};",
        r"\end{tikzpicture}",
        r"\end{document}",
        "",
    ]
    path.write_text("\n".join(lines))


def _cell(values: pd.Series, digits: int, best: bool) -> str:
    """`mean +- standard error` over seeds, the mean bold where `best`, `--` where nothing is finite"""
    finite = values[np.isfinite(values)]
    if finite.empty:
        return "--"
    mean, error = f"{finite.mean():.{digits}f}", f"{finite.std(ddof=1) / np.sqrt(finite.size) if finite.size > 1 else 0.0:.{digits}f}"
    return (rf"\textbf{{{mean}}}" if best else mean) + rf" {{\scriptsize $\pm$ {error}}}"


def write_table(path: Path, scalars: pd.DataFrame, columns: Sequence[tuple[str, str, int, str]], labels: dict[str, str]) -> Path:
    """A booktabs tabular at `path`, a row per acquisition of `scalars` (`acquisition, seed`, and the columns), as (column, heading, digits, `min` | `max` | `none`)

    Rows follow the order of `FAMILY_COLOURS` with a rule under the last of `OURS`; a heading carries an arrow toward its better end.
    Gives back the standalone document beside it that inputs the tabular.
    """
    grouped = scalars.groupby("acquisition")
    means = grouped[[c for c, *_ in columns]].mean()
    best = {
        c: ("" if better == "none" or means[c].dropna().empty else (means[c].idxmin() if better == "min" else means[c].idxmax()))
        for c, _, _, better in columns
    }
    order = list(FAMILY_COLOURS)
    ranked = sorted(
        grouped, key=lambda pair: (order.index(f) if (f := family(labels[str(pair[0])])) in order else len(order), labels[str(pair[0])])
    )
    rows: list[str] = []
    for i, (a, held) in enumerate(ranked):
        if i > 0 and family(labels[str(ranked[i - 1][0])]) in OURS and family(labels[str(a)]) not in OURS:
            rows.append(r"\midrule")
        rows.append(" & ".join([labels[str(a)], *(_cell(held[c], digits, a == best[c]) for c, _, digits, _ in columns)]) + r" \\")
    arrow = {"min": r" $\downarrow$", "max": r" $\uparrow$", "none": ""}
    path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{l" + "c" * len(columns) + "}",
                r"\toprule",
                " & ".join(["method", *(heading + arrow[better] for _, heading, _, better in columns)]) + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        )
    )
    standalone = path.with_name(f"{path.stem}_standalone.tex")
    standalone.write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{standalone}",
                r"\usepackage{booktabs}",
                r"\begin{document}",
                r"\footnotesize",
                rf"\input{{{path.stem}}}",
                r"\end{document}",
                "",
            ]
        )
    )
    return standalone


def compile_document(path: Path) -> None:
    """`path`'s PDF alone under `build/` beside it through tectonic, the style copied beside it first; a missing style or binary is reported, not raised"""
    if not STYLE.exists():
        print(f"no style at {STYLE}, so {path.name} was written but not compiled")
        return
    (path.parent / "configs").mkdir(exist_ok=True)
    shutil.copy(STYLE, path.parent / "configs" / "style.tex")
    binary = shutil.which("tectonic") or shutil.which("tectonic", path=str(Path(sys.executable).parent))
    if binary is None:
        print(f"no tectonic on PATH, so {path.name} was written but not compiled")
        return
    build = path.parent / "build"
    build.mkdir(exist_ok=True)
    compiled = subprocess.run([binary, "-X", "compile", "--outdir", str(build), str(path)], capture_output=True, text=True, check=False)
    if compiled.returncode:
        print(f"tectonic exited {compiled.returncode} on {path.name}:\n" + "\n".join(compiled.stderr.splitlines()[-12:]))
