"""Shared plot formatting helpers."""

from __future__ import annotations


def apply_three_decimal_ticks(ax, *, x_axis: bool = True, y_axis: bool = True) -> None:
    from matplotlib.ticker import FormatStrFormatter

    if x_axis:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    if y_axis:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def annotate_first_last_frame(ax, rows, x_values, index_column: str) -> None:
    valid = [
        (row, x_value)
        for row, x_value in zip(rows, x_values)
        if x_value is not None
    ]
    if not valid:
        return

    first_row, first_x = valid[0]
    last_row, last_x = valid[-1]
    first_label = first_row.get(index_column, "0") or "0"
    last_label = last_row.get(index_column, str(len(valid) - 1)) or str(len(valid) - 1)

    ax.text(first_x, -0.48, first_label, fontsize=8, ha="left", va="top", color="#222222")
    ax.text(last_x, -0.48, last_label, fontsize=8, ha="right", va="top", color="#222222")
