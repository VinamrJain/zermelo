"""The sheets one sweep run is drawn as"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from zermelo.experiments.balloon_waypoint.metrics.data import CURVES, DIAGNOSTICS
from zermelo.experiments.balloon_waypoint.render.style import Style, plain_numbers, text_width

AXES = {**CURVES, **DIAGNOSTICS}
"""Every series a curve panel can draw, as the words on its axis and the scale it takes"""

DRAWN_AS_QUANTILES = ("simple_regret", "cumulative_regret")
"""Curves drawn as a median inside an interquartile band, the rest as a mean plus or minus a standard error"""


def _band(curves: Sequence[str], seeds: int, opening: int) -> str:
    """What the shading behind a line means and what the dotted line marks, written on the figure"""
    quantile = [name for name in curves if name in DRAWN_AS_QUANTILES]
    if not quantile:
        said = "mean and standard error"
    elif len(quantile) == len(curves):
        said = "median and interquartile band"
    else:
        said = "regret as median and interquartile band, the rest as mean and standard error"
    return f"{said}, over {seeds} seed{'s' if seeds != 1 else ''}; the dotted line closes the shared opening at move {opening}"


def _rules(panel: Axes, style: Style) -> None:
    """The grid, the tick colours and the two spines every panel here is read against"""
    panel.tick_params(labelsize=style.tick_size, colors=style.faint, labelcolor=style.ink)
    panel.grid(True, color=style.faint, alpha=0.18, lw=0.7)
    panel.set_axisbelow(True)
    for edge, shown in (("top", False), ("right", False), ("left", True), ("bottom", True)):
        panel.spines[edge].set_visible(shown)
        panel.spines[edge].set_color(style.faint)


def _dress(panel: Axes, curve: str, opening: int, style: Style) -> None:
    """The scale, the two axis names, the rules and the opening mark one curve panel is read against"""
    panel.set_yscale(AXES[curve][1])
    panel.set_xlabel("move $t$", fontsize=style.label_size, color=style.ink)
    panel.set_ylabel(AXES[curve][0], fontsize=style.label_size, color=style.ink)
    panel.axvline(opening, ls=":", lw=1.4, color=style.faint, alpha=0.8, zorder=1)  # every arm walks alike left of this
    _rules(panel, style)
    if AXES[curve][1] == "log":
        plain_numbers(panel)


def _lines(panel: Axes, data: pd.DataFrame, curve: str, colours: dict[str, str], names: dict[str, str], opening: int, style: Style) -> None:
    """One curve against move: a line for each arm, with the spread of its seeds as a band behind it"""
    for arm, rows in data.groupby("arm"):
        by_move = rows.groupby("step")[curve]
        if curve in DRAWN_AS_QUANTILES:
            middle, low, high = by_move.median(), by_move.quantile(0.25), by_move.quantile(0.75)
        else:
            middle = by_move.mean()
            error = by_move.std().div(np.sqrt(by_move.count()))  # standard error of the mean over seeds
            low, high = middle - error, middle + error
        arm = str(arm)
        panel.plot(middle.index, middle, color=colours[arm], lw=style.curve_width, label=names[arm], zorder=3)
        panel.fill_between(middle.index, low, high, color=colours[arm], alpha=style.band_alpha, lw=0.0, zorder=2)
    _dress(panel, curve, opening, style)


def _key(figure: Figure, panel: Axes, said: str, style: Style) -> None:
    """The arms named under the panels, the key titled by what the shading means, the sheet widened to hold it"""
    marks, named = panel.get_legend_handles_labels()
    handle = (style.key_handle + style.key_spacing) * style.curve_key_size / 72.0  # the mark and the space after it
    column = max(text_width(figure, name, style.curve_key_size) for name in named) + handle
    wide, tall = figure.get_size_inches()
    across = min(len(named), max(2, int((wide - 2 * style.margin) // column)))  # two at least, or eight arms make eight rows
    line = style.key_line * style.curve_key_size / 72.0
    # the sheet is never narrower than its widest key row or its title
    floor = max(across * column, text_width(figure, said, style.curve_key_size))
    figure.set_size_inches(max(wide, 2 * style.margin + floor), tall + (1 + np.ceil(len(named) / across)) * line)
    key = figure.legend(
        marks,
        named,
        loc="outside lower center",
        ncol=across,
        fontsize=style.curve_key_size,
        title=said,
        labelcolor=style.ink,
        frameon=False,
        handlelength=style.key_handle,
        columnspacing=style.key_spacing,
    )
    key.get_title().set(fontsize=style.curve_key_size, color=style.faint)


def _sheet(rows: int, columns: int, style: Style) -> tuple[Figure, Any]:
    """A grid of empty panels, each at the size a page places one panel at"""
    figure, panels = plt.subplots(
        rows, columns, figsize=(columns * style.panel_width, rows * style.panel_height), layout="constrained", facecolor=style.paper
    )
    figure.set_layout_engine("constrained", w_pad=style.margin / 2, h_pad=style.margin / 2)
    return figure, panels


def draw_curves(
    data: pd.DataFrame,
    drawn: Mapping[str, tuple[str, str]],
    colours: dict[str, str],
    names: dict[str, str],
    opening: int,
    style: Style,
    title: str,
    into: Path,
) -> None:
    """Every curve of `drawn` on one sheet under one key, two panels across"""
    columns = 2
    figure, panels = _sheet(int(np.ceil(len(drawn) / columns)), columns, style)
    for panel in panels.flat[len(drawn) :]:  # an odd count leaves a trailing panel with nothing to draw
        panel.set_axis_off()
    for panel, curve in zip(panels.flat, drawn, strict=False):
        _lines(panel, data, curve, colours, names, opening, style)
    figure.suptitle(title, fontsize=style.title_size, color=style.ink, x=0.012, ha="left")
    _key(figure, panels.flat[0], _band(tuple(drawn), data["seed"].nunique(), opening), style)
    figure.savefig(into, dpi=style.dpi, facecolor=style.paper)
    plt.close(figure)


def draw_legs(legs: pd.DataFrame, budget: int, colour: str, style: Style, title: str, into: Path) -> None:
    """One episode's finished waypoint legs: steps walked against the steps priced, and against the move each ended at"""
    figure, panels = _sheet(1, 2, style)
    priced, over = panels.flat[0], panels.flat[1]

    priced.plot(legs["predicted"], legs["walked"], "o", ms=7.0, mew=0.0, alpha=0.7, color=colour, zorder=3)
    priced.plot([0.0, float(budget)], [0.0, float(budget)], "--", lw=1.2, color=style.faint, zorder=2)  # k_j = H_j
    priced.set_xlabel("steps predicted $H_j$", fontsize=style.label_size, color=style.ink)

    over.plot(legs["ended"], legs["walked"], "o", ms=7.0, mew=0.0, alpha=0.7, color=colour, zorder=3)
    over.axhline(budget, ls="--", lw=1.2, color=style.faint, zorder=2)  # k_j = L, where a waypoint leg is given up on
    over.set_xlabel("move the leg ended at", fontsize=style.label_size, color=style.ink)

    for panel in (priced, over):
        panel.set_ylabel("steps walked $k_j$", fontsize=style.label_size, color=style.ink)
        panel.set_ylim(0.0, budget * 1.06)
        _rules(panel, style)
    arrived = float(legs["arrived"].mean())
    figure.suptitle(title, fontsize=style.subtitle_size, color=style.ink, x=0.012, ha="left")
    figure.supxlabel(
        f"{len(legs)} legs, {arrived:.0%} of them arriving; the dashes are $k_j = H_j$ and $k_j = L = {budget}$",
        fontsize=style.curve_key_size,
        color=style.faint,
    )
    figure.savefig(into, dpi=style.dpi, facecolor=style.paper)
    plt.close(figure)


def draw_cost(
    spent: pd.DataFrame, data: pd.DataFrame, colours: dict[str, str], names: dict[str, str], style: Style, title: str, into: Path
) -> None:
    """What each arm bought with what it spent: where every curve ended, against the seconds one move of planning took"""
    columns = 2
    figure, panels = _sheet(int(np.ceil(len(CURVES) / columns)), columns, style)
    ended = data.sort_values("step").groupby(["arm", "seed"]).last()
    for panel in panels.flat[len(CURVES) :]:  # an odd count leaves a trailing panel with nothing to draw
        panel.set_axis_off()
    for panel, curve in zip(panels.flat, CURVES, strict=False):
        for arm in sorted(names):
            reached, seconds = ended[curve].loc[arm], spent.loc[spent["arm"] == arm, "seconds_per_move"]
            middle, sideways = float(reached.median()), float(seconds.mean())
            panel.errorbar(
                sideways,
                middle,
                xerr=float(seconds.std()) if len(seconds) > 1 else 0.0,
                yerr=[[middle - float(reached.quantile(0.25))], [float(reached.quantile(0.75)) - middle]],
                fmt="o",
                ms=9.0,
                color=colours[arm],
                ecolor=colours[arm],
                elinewidth=1.4,
                capsize=3.0,
                label=names[arm],
                zorder=3,
            )
        panel.set_yscale(CURVES[curve][1])
        panel.set_xlabel("seconds of planning per move", fontsize=style.label_size, color=style.ink)
        panel.set_ylabel(f"final {CURVES[curve][0]}", fontsize=style.label_size, color=style.ink)
        panel.margins(x=0.10, y=0.14)
        _rules(panel, style)
    figure.suptitle(title, fontsize=style.title_size, color=style.ink, x=0.012, ha="left")
    said = f"median and interquartile band up, mean and standard deviation across, over {data['seed'].nunique()} seeds"
    _key(figure, panels.flat[0], said, style)
    figure.savefig(into, dpi=style.dpi, facecolor=style.paper)
    plt.close(figure)


def draw_altitude(
    wind: Float[np.ndarray, "alt lat lon"],
    flow: Float[np.ndarray, "alt lat lon uv"],
    extent: tuple[float, float, float, float],
    altitude_km: Sequence[float],
    stood: Float[np.ndarray, "moves 2"],
    altitude: Int[np.ndarray, " moves"],
    colour: str,
    style: Style,
    title: str,
    into: Path,
) -> None:
    """The wind at each altitude across, with the cells the episode stood at that altitude marked on it"""
    figure, panels = plt.subplots(
        1,
        len(altitude_km),
        figsize=(len(altitude_km) * style.panel_width * 0.72, style.panel_height * 1.15),
        layout="constrained",
        facecolor=style.paper,
        sharex=True,
        sharey=True,
    )
    fastest = np.unravel_index(int(np.argmax(wind)), wind.shape)  # the one fastest point of the whole volume
    span = float(np.max(wind))
    lon = np.linspace(extent[0], extent[1], wind.shape[2])
    lat = np.linspace(extent[2], extent[3], wind.shape[1])
    every = max(1, wind.shape[2] // 24)  # arrows thinned to a readable count across
    for level, panel in enumerate(panels.flat):
        panel.pcolormesh(lon, lat, wind[level], cmap=style.speed_colours, vmin=0.0, vmax=span, shading="auto", rasterized=True)
        panel.quiver(
            lon[::every],
            lat[::every],
            flow[level, ::every, ::every, 0],
            flow[level, ::every, ::every, 1],
            color=style.arrow_colour,
            width=style.arrow_width,
            scale=span / style.arrow_span,
            scale_units="width",
        )
        here = stood[altitude == level]
        if here.size:
            panel.plot(here[:, 1], here[:, 0], "o", ms=2.6, mew=0.0, alpha=0.55, color=colour, zorder=3)
        if level == fastest[0]:
            panel.plot(lon[fastest[2]], lat[fastest[1]], "*", ms=15.0, mew=0.8, color=style.paper, mec=style.ink, zorder=4)
        panel.set_title(f"{altitude_km[level]:.1f} km   {len(here)} moves", fontsize=style.panel_title_size, color=style.ink)
        panel.set_aspect(1.0 / np.cos(np.radians(float(np.mean(lat)))))
        panel.set_xlim(extent[0], extent[1])  # the field's own edges: a move outside the grid cannot stretch the sheet
        panel.set_ylim(min(extent[2], extent[3]), max(extent[2], extent[3]))  # north up, whichever way the rows run
        _rules(panel, style)
    figure.suptitle(title, fontsize=style.subtitle_size, color=style.ink, x=0.012, ha="left")
    figure.supxlabel(
        f"colour is wind speed to {span:.0f} m/s; the star marks the fastest point of the whole volume",
        fontsize=style.curve_key_size,
        color=style.faint,
    )
    figure.savefig(into, dpi=style.dpi, facecolor=style.paper)
    plt.close(figure)
