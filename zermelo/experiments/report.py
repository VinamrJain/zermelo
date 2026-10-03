"""One launch written for the paper: a CSV per curve per acquisition, a pgfplots figure reading them, and a booktabs table

Every curve is a median over seeds inside its interquartile band, at every x; every table cell is median [q25, q75] at the last x.
"""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

STYLE = Path("../paper/manuscript/figures/configs/style.tex")
"""The paper's shared pgfplots style, copied beside the figure so it compiles where it is written"""

COLOURED = ("RandAct", "RandWP", "MaxVar", "UCB", "EI", "TS", "Forecast", "EMI", "EVI", "ESI")
"""Acquisition families the paper's style defines a colour for, under the family's own name"""

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
    "",
    "mark=*, mark size=1.4pt, mark repeat=12, mark phase=1",
    "dashed, mark=square*, mark size=1.4pt, mark repeat=12, mark phase=5",
    "densely dotted, mark=triangle*, mark size=1.7pt, mark repeat=12, mark phase=9",
    "dash dot, mark=diamond*, mark size=1.7pt, mark repeat=12, mark phase=3",
)
"""Line styles telling apart the acquisitions of one family, in the order they are drawn"""

ROWS_PER_CURVE = 100
"""Most x values one CSV holds; a longer curve is thinned to every k-th x with its last x kept"""


def family(label: str) -> str:
    """The acquisition family of a paper label: `EMI-L25` -> `EMI`, `ESI-L8 (wind)` -> `ESI`"""
    return label.split("-")[0].split(" ")[0]


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
) -> None:
    """A standalone pgfplots group over `panels` as (column, y label, `linear` or `log`), one band per acquisition, legend below

    Acquisitions of one family share its colour and differ by line style; a dotted rule at `opening_x` marks the shared opening.
    """
    families = [family(labels[a]) for a in acquisitions]
    coloured = all(f in COLOURED for f in families)
    colour = {a: families[i] if coloured else f"acq{chr(65 + i)}" for i, a in enumerate(acquisitions)}
    style = {
        a: VARIANT_STYLES[families[:i].count(families[i]) % len(VARIANT_STYLES)] if coloured else "" for i, a in enumerate(acquisitions)
    }
    defined = (
        []
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
        rf"    group style={{group size={columns} by {rows}, horizontal sep=42pt, vertical sep=36pt}},",
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
            lines.append(rf"    \band[{style[a]}]{{{colour[a]}}}{{{csvs[(column, a)].relative_to(path.parent).as_posix()}}}")
        if opening_x is not None:
            lines.append(rf"    \opening{{{opening_x:g}}}")
        if i == 0:
            lines.append(r"    \legend{" + ", ".join(labels[a] for a in acquisitions) + "}")
    lines += [
        r"\end{groupplot}",
        rf"\node[anchor=north, yshift=-28pt] at ($(group c1r{rows}.south)!0.5!(group c{columns}r{rows}.south)$) {{\pgfplotslegendfromname{{legend:{path.stem}}}}};",
        r"\end{tikzpicture}",
        r"\end{document}",
        "",
    ]
    path.write_text("\n".join(lines))


def _cell(values: pd.Series, digits: int, best: bool) -> str:
    """`med [q25, q75]` over seeds, the median bold where `best`, `--` where nothing is finite"""
    finite = values[np.isfinite(values)]
    if finite.empty:
        return "--"
    med, q25, q75 = (f"{q:.{digits}f}" for q in finite.quantile([0.5, 0.25, 0.75]))
    return (rf"\textbf{{{med}}}" if best else med) + rf" {{\scriptsize [{q25}, {q75}]}}"


def write_table(path: Path, scalars: pd.DataFrame, columns: Sequence[tuple[str, str, int, str]], labels: dict[str, str]) -> Path:
    """A booktabs tabular at `path`, a row per acquisition of `scalars` (`acquisition, seed`, and the columns), as (column, heading, digits, `min` | `max` | `none`)

    Gives back the standalone document beside it that inputs the tabular.
    """
    grouped = scalars.groupby("acquisition")
    medians = grouped[[c for c, *_ in columns]].median()
    best = {
        c: ("" if better == "none" or medians[c].dropna().empty else (medians[c].idxmin() if better == "min" else medians[c].idxmax()))
        for c, _, _, better in columns
    }
    rows = [
        " & ".join([labels[str(a)], *(_cell(held[c], digits, a == best[c]) for c, _, digits, _ in columns)]) + r" \\" for a, held in grouped
    ]
    path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{l" + "c" * len(columns) + "}",
                r"\toprule",
                " & ".join(["acquisition", *(heading for _, heading, _, _ in columns)]) + r" \\",
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
