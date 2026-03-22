    
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
def exec_plot_scat(plot_tran_pure: PlotTranEXAM_01_scat) -> None:
    # from vcss.c02_vcss_01_grph_plot_ import PlotTranEXAM_01_scat

    trac = True

    # -----------------
    # Data
    # -----------------
    df_fram = plot_tran_pure.fram
    if trac:
        print_yes(df_fram, labl="df_fram")

    plt.style.use('default')

    colu_list = plot_tran_pure.colu_list   # e.g. ['CEAP_R', 'CEAP_L']

    # -----------------
    # Convert categories → numeric codes
    # -----------------
    # Use the order defined in pandas category
    ceap_categories = list(df_fram[colu_list[0]].cat.categories)

    data_ser1 = df_fram[colu_list[0]].cat.codes
    data_ser2 = df_fram[colu_list[1]].cat.codes

    # remove NaN (coded as -1)
    mask = (data_ser1 >= 0) & (data_ser2 >= 0)
    data_ser1 = data_ser1[mask]
    data_ser2 = data_ser2[mask]

    max_score = len(ceap_categories) - 1

    # -----------------
    # Figure
    # -----------------
    fig = plot_tran_pure.figu
    ax = plot_tran_pure.axis
    ax.clear()

    scat_size = plot_tran_pure.scat_size
    scat_edge = plot_tran_pure.scat_edge
    scat_colo = plot_tran_pure.scat_colo
    scat_alph = plot_tran_pure.scat_alph

    # -----------------
    # Scatter strategies
    # -----------------
    match (plot_tran_pure.stra):

        case 'a':  # default
            ax.scatter(data_ser1, data_ser2,
                       alpha=scat_alph, s=scat_size,
                       c=scat_colo, edgecolors=scat_edge)

        case 'b':  # jitter
            jitter = 0.15
            x = data_ser1 + np.random.uniform(-jitter, jitter, size=len(data_ser1))
            y = data_ser2 + np.random.uniform(-jitter, jitter, size=len(data_ser2))
            ax.scatter(x, y,
                       alpha=scat_alph, s=scat_size,
                       c=scat_colo, edgecolors=scat_edge)

        case 'c1' | 'c2' | 'c3':  # ties as bubble size / color

            counts = (
                df_fram[colu_list]
                .dropna()
                .apply(lambda s: s.cat.codes)
                .value_counts()
                .reset_index(name='count')
            )

            x = counts[colu_list[0]]
            y = counts[colu_list[1]]
            sizes = counts['count'] * 40

            if plot_tran_pure.stra == 'c3':
                bounds = np.arange(counts['count'].min() - 0.5,
                                   counts['count'].max() + 1.5)
                norm = BoundaryNorm(bounds, ncolors=plt.cm.viridis.N)
                sc = ax.scatter(x, y, s=sizes, c=counts['count'],
                                cmap='viridis', edgecolors=scat_edge,
                                alpha=scat_alph, norm=norm)
                cbar = plt.colorbar(sc, ax=ax,
                                    ticks=np.arange(counts['count'].min(),
                                                    counts['count'].max() + 1))
            else:
                sc = ax.scatter(x, y, s=sizes, c=counts['count'],
                                cmap='viridis', edgecolors=scat_edge,
                                alpha=scat_alph)
                cbar = plt.colorbar(sc, ax=ax)

                if plot_tran_pure.stra == 'c2':
                    cbar.locator = MaxNLocator(integer=True)
                    cbar.update_ticks()

            cbar.set_label('Number of patients')

        case 'd':  # table look
            ax.scatter(data_ser1, data_ser2,
                       alpha=scat_alph, s=scat_size,
                       c=scat_colo, edgecolors=scat_edge)
            ax.set_aspect('equal')

        case _:
            raise Exception()

    # -----------------
    # Diagonal
    # -----------------
    ax.plot([0, max_score], [0, max_score],
            color='blue', linestyle='--', lw=2,
            label='L,R limb symmetry')

    # -----------------
    # Axis labels
    # -----------------
    ax.set_xlabel(plot_tran_pure.xlab)
    ax.set_ylabel(plot_tran_pure.ylab)

    ax.set_xlim(-0.5, max_score + 0.5)
    ax.set_ylim(-0.5, max_score + 0.5)

    ax.set_xticks(range(len(ceap_categories)))
    ax.set_yticks(range(len(ceap_categories)))

    ax.set_xticklabels(ceap_categories)
    ax.set_yticklabels(ceap_categories)

    # -----------------
    # Grid
    # -----------------
    grid_alph = plot_tran_pure.grid_alph
    if grid_alph is not None:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=grid_alph)

    # -----------------
    # Legend
    # -----------------
    ax.legend()

    # -----------------
    # Symmetry text
    # -----------------
    diff = data_ser1 - data_ser2
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
                  alpha=0.8)
    )

    # -----------------
    # Title
    # -----------------
    ax.set_title(plot_tran_pure.titl)

    fig.tight_layout()
    # fig.show()
    pass
    
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