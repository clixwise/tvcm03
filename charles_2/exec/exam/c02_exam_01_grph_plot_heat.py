    
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c02_exam_01_grph_plot_ import PlotTranEXAM_01_scat
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker

# ----
# Timepoint outcomes
# ----
# https://chatgpt.com/c/69878a74-0900-8331-ac82-014c76b9ea24
def exec_plot_heat(plot_tran_pure: PlotTranEXAM_01_scat) -> None:
    # from vcss.c02_vcss_01_grph_plot_ import PlotTranEXAM_01_scat

    # =====================
    # USER OPTIONS
    # =====================
    SHOW_PERCENT = True   # True = %, False = absolute counts
    EXPORT_EXCEL = True

    trac = True

    # -----------------
    # Data
    # -----------------
    df_fram = plot_tran_pure.fram
    if trac:
        print_yes(df_fram, labl="df_fram")

    colu_list = plot_tran_pure.colu_list  # ['CEAP_R', 'CEAP_L']

    ceap_categories = list(df_fram[colu_list[0]].cat.categories)

    # -----------------
    # Crosstab
    # -----------------
    tab = pd.crosstab(
        df_fram[colu_list[1]],  # rows = L
        df_fram[colu_list[0]],  # cols = R
        dropna=False
    )

    tab = tab.reindex(index=ceap_categories, columns=ceap_categories, fill_value=0)

    # Save original counts for stats
    tab_counts = tab.copy()

    # -----------------
    # Percent option
    # -----------------
    if SHOW_PERCENT:
        total = tab.values.sum()
        if total > 0:
            tab = tab / total * 100

    # -----------------
    # Figure
    # -----------------
    fig = plot_tran_pure.figu
    ax = plot_tran_pure.axis
    ax.clear()

    # -----------------
    # Zero mask → grey
    # -----------------
    data = tab.values
    masked = np.ma.masked_where(tab_counts.values == 0, data)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(masked, cmap=cmap)

    # -----------------
    # Colorbar
    # -----------------
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% of patients" if SHOW_PERCENT else "Number of patients")

    # -----------------
    # Ticks
    # -----------------
    ax.set_xticks(range(len(ceap_categories)))
    ax.set_yticks(range(len(ceap_categories)))

    ax.set_xticklabels(ceap_categories)
    ax.set_yticklabels(ceap_categories)

    ax.set_xlabel(plot_tran_pure.xlab)
    ax.set_ylabel(plot_tran_pure.ylab)

    # -----------------
    # Invert Y axis
    # -----------------
    ax.invert_yaxis()

    # -----------------
    # Annotate cells
    # -----------------
    max_val = data.max() if data.size else 0

    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):

            count = tab_counts.iloc[i, j]
            if count == 0:
                continue

            value = data[i, j]

            if SHOW_PERCENT:
                text = f"{value:.1f}"
            else:
                text = f"{int(value)}"

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > max_val / 2 else "black",
                fontsize=9
            )

    # -----------------
    # Diagonal
    # -----------------
    ax.plot([-0.5, len(ceap_categories) - 0.5],
            [-0.5, len(ceap_categories) - 0.5],
            linestyle='--', linewidth=2, color='red')

    # -----------------
    # Symmetry stats
    # -----------------
    codes_R = df_fram[colu_list[0]].cat.codes
    codes_L = df_fram[colu_list[1]].cat.codes
    mask = (codes_R >= 0) & (codes_L >= 0)

    diff = codes_L[mask] - codes_R[mask]

    ax.text(
        0.98, 0.02,
        f'L > R: {(diff > 0).sum()}\n'
        f'R > L: {(diff < 0).sum()}\n'
        f'L = R: {(diff == 0).sum()}',
        transform=ax.transAxes,
        va='bottom', ha='right',
        fontsize=10,
        bbox=dict(boxstyle='square',
                  facecolor='white',
                  edgecolor='none',
                  alpha=0.9)
    )

    # -----------------
    # Title
    # -----------------
    ax.set_title(plot_tran_pure.titl)

    fig.tight_layout()
    #fig.show()
    
def print_yes(df, labl=None):
    print (f"\n----\nFram labl : {labl}\n----")
    with pd.option_context(
            'display.max_columns', None,       # Show all columns
            # 'display.max_rows', None,          # Show more rows before truncating
            'display.max_colwidth', None,      # Don't cut off long text in 'info'
            'display.width', 1000,             # Prevent the table from wrapping to a new line
            'display.precision', 2,            # Round floats to 2 decimal places
            'display.colheader_justify', 'left' # Align headers for better readability
        ):
        print(f"df:{len(df)} type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        print(df.info())
    pass