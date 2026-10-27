"""
charts.py
---------
Generates Matplotlib charts and returns them as Base64-encoded PNG strings.
All chart functions accept pandas Series or DataFrames and return a dict
with a 'chart' key containing the base64 string.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Colour palette ───────────────────────────────────────────────────────────
PALETTE = ["#6366f1", "#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#fb923c"]
BG_COLOR = "#0f172a"
PANEL_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"


def _apply_dark_style(ax, fig):
    """Apply a dark, clean style consistent with the Spatial UI."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


def _figure_to_base64(fig):
    """Convert a Matplotlib figure to a Base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # Free memory
    buf.close()
    return encoded


# ── Bar chart ────────────────────────────────────────────────────────────────
def make_bar_chart(labels, values, title="Bar Chart",
                   xlabel="Category", ylabel="Count"):
    """Vertical bar chart for frequency distributions."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, edgecolor=PANEL_COLOR,
                  linewidth=0.8, zorder=3)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(values, default=1),
                f"{val}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.xticks(rotation=30, ha="right")
    _apply_dark_style(ax, fig)
    return _figure_to_base64(fig)


# ── Pie chart ────────────────────────────────────────────────────────────────
def make_pie_chart(labels, values, title="Pie Chart"):
    """Pie chart for percentage breakdowns."""
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=PALETTE[:len(labels)],
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor=PANEL_COLOR, linewidth=1.4),
    )
    for text in texts:
        text.set_color(TEXT_COLOR)
        text.set_fontsize(9)
    for atext in autotexts:
        atext.set_color(TEXT_COLOR)
        atext.set_fontsize(8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=TEXT_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    return _figure_to_base64(fig)


# ── Histogram ────────────────────────────────────────────────────────────────
def make_histogram(values, title="Histogram", xlabel="Value",
                   ylabel="Frequency", mean_val=None, median_val=None):
    """Histogram with optional mean/median lines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins="auto", color=PALETTE[0], edgecolor=PANEL_COLOR,
            linewidth=0.8, zorder=3)

    if mean_val is not None:
        ax.axvline(mean_val, color="#f472b6", linestyle="--",
                   linewidth=1.8, label=f"Mean: {mean_val:.2f}", zorder=4)
    if median_val is not None:
        ax.axvline(median_val, color="#34d399", linestyle=":",
                   linewidth=1.8, label=f"Median: {median_val:.2f}", zorder=4)

    if mean_val is not None or median_val is not None:
        legend = ax.legend(fontsize=9)
        legend.get_frame().set_facecolor(PANEL_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    _apply_dark_style(ax, fig)
    return _figure_to_base64(fig)


# ── Scatter plot ─────────────────────────────────────────────────────────────
def make_scatter(x_values, y_values, title="Scatter Plot",
                 xlabel="X", ylabel="Y",
                 regression_line=None):
    """
    Scatter plot, with an optional regression line.
    regression_line: dict with keys 'slope', 'intercept' if provided.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, color=PALETTE[1], alpha=0.75,
               edgecolors=PANEL_COLOR, linewidth=0.6, zorder=3, s=60)

    if regression_line:
        slope = regression_line["slope"]
        intercept = regression_line["intercept"]
        x_arr = np.array(x_values)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#f472b6", linewidth=2,
                label=f"y = {slope:.4f}x + {intercept:.4f}", zorder=4)
        legend = ax.legend(fontsize=9)
        legend.get_frame().set_facecolor(PANEL_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    _apply_dark_style(ax, fig)
    return _figure_to_base64(fig)


# ── Grouped bar (observed vs expected) ───────────────────────────────────────
def make_grouped_bar(categories, observed, expected, title="Observed vs Expected"):
    """Grouped bar chart comparing observed and expected frequencies."""
    x = np.arange(len(categories))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.4), 5))
    ax.bar(x - width / 2, observed, width, label="Observed",
           color=PALETTE[0], edgecolor=PANEL_COLOR, linewidth=0.8, zorder=3)
    ax.bar(x + width / 2, expected, width, label="Expected",
           color=PALETTE[1], edgecolor=PANEL_COLOR, linewidth=0.8,
           alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    legend = ax.legend(fontsize=9)
    legend.get_frame().set_facecolor(PANEL_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    _apply_dark_style(ax, fig)
    return _figure_to_base64(fig)
