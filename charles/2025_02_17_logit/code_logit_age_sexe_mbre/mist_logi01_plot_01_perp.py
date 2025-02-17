import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from scipy.stats import norm

from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from util_file_mngr import write
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import scipy.stats as scs

from scipy.interpolate import make_interp_spline

# ====
# DETA plots
# ====
def mist_logi01_anal11x(df, ylab, ysca, yzer, titl, foot, omit_const=False, link_coef=False, line_alpha=0.5, line_width=1, area_alpha=0.1):
 
        # Data
        # ----
        if omit_const:
            df = df[df['coef=x(independant)'] != 'const']
        ceap_categories = df['ceap=y(dependant)'].unique()
        coef_variables = df['coef=x(independant)'].unique()
        max_coef_by_ceap = df[df['coef=x(independant)'] != 'const'].groupby('ceap=y(dependant)').apply(
                        lambda x: x.loc[x['coef'].abs().idxmax(), 'coef=x(independant)']).to_dict()
 
        # Grph
        # ----
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'const': 'red', 'sexe': 'blue', 'mbre': 'green', 'age': 'purple'}
        offset = 0.30
        positions = np.arange(len(ceap_categories))

        # Swimlanes
        # ---------
        for pos in positions[0:]:
            ax.axvline(x=pos - 0.5, color='gray', linestyle='-', alpha=0.5, zorder=0, linewidth=0.5)
        for i, pos in enumerate(positions):
            if i % 2 == 0:
                ax.axvspan(pos - 0.5, pos + 0.5, facecolor='lightgray', alpha=area_alpha, zorder=0)
        
        # Plot
        # ----
        for i, coef in enumerate(coef_variables):
            
            coef_data = df[df['coef=x(independant)'] == coef]
            x = positions + (i - len(coef_variables)/2 + 0.5) * offset
            y = coef_data['circ']  
                    
            # B-Spline
            spli_b = False
            if spli_b:
                x_smooth = np.linspace(x.min(), x.max(), 300)
                spl = make_interp_spline(x, y, k=3) # a bspline
                y_smooth = spl(x_smooth)   
                ax.plot(x_smooth, y_smooth, color=colors[coef], linewidth=1, alpha=0.7, zorder=1)
            
            # C-Spline
            spli_c = True
            def plot_spli_c(ax, x, y, color, shift=0):
                # Use the actual x values instead of bin centers
                cs = CubicSpline(x, y, bc_type='natural')
                x_smooth = np.linspace(x.min(), x.max(), 300)
                smoothed = cs(x_smooth)
                ax.plot(x_smooth + shift, smoothed, color=color, linewidth=1, linestyle='-', alpha=0.7, zorder=1)
            if spli_c:
                plot_spli_c(ax, x, y, color=colors[coef], shift=0)
    
            # Dots (odds_ratio, ...) and Segments (CI,...)
            ax.scatter(x, coef_data['circ'], color=colors[coef], label=coef, s=50, zorder=3)
            ax.errorbar(x, coef_data['circ'], 
                        yerr=np.array(coef_data['segm'].tolist()).T, # yerr=coef_data['segm']
                    fmt='none', ecolor=colors[coef], elinewidth=line_width, capsize=5, zorder=2)

            # Link lines
            if link_coef:
                ax.plot(x, coef_data['circ'], color=colors[coef], linestyle=':', alpha=line_alpha, linewidth=line_width, zorder=1)

            # Annotations
            for j, (idx, row) in enumerate(coef_data.iterrows()):
                weight = 'bold' if (coef != 'const' and coef == max_coef_by_ceap[row['ceap=y(dependant)']]) else 'normal'      
                if True:
                    annotation_text = f"{coef}\n[{row['circ']:.2f}]\n({row['P>|z|']:.2f})"
                    # if Latex installed
                    # annotation_text = f"{coef}\n({row['odds_ratio']:.2f})\n"
                    # annotation_text += r"$\mathit{(" + f"{row['P>|z|']:.3f}" + r")}$"
                    ax.annotate(annotation_text,
                                (x[j], row['anno']),
                                xytext=(0, -8), textcoords='offset points', 
                                ha='center', va='top', fontsize=9, weight=weight)
                else: 
                    p_value = 99
                    annotation_text = f"{coef}\nOR: {row['odds_ratio']:.2f}\n"
                    annotation_text += r"$\textcolor{gray}{\fontsize{7pt}{8pt}\selectfont p: " + f"{p_value:.3f}" + r"}$"
                    annotation_text = f"{coef}\nOR: {row['odds_ratio']:.2f}\n"
                    annotation_text += r"$\it{p}$" + f": {p_value:.3f}"
                    ax.annotate(annotation_text,
                                (x[j], row['anno']),
                                xytext=(0, -15), textcoords='offset points', 
                                ha='center', va='top', fontsize=9, weight=weight)
        if False:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "font.size": 9,
                "text.latex.preamble": r"\usepackage{relsize}"
            })

        # Axis & Title
        # ----
        ax.set_yscale(ysca)
        ax.set_ylabel(ylab)
        ax.set_xlabel('ceap=y(dependant)')
        ax.set_xticks(positions)
        ax.set_xticklabels(ceap_categories)
        if not omit_const:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top') 
          
        ax.axhline(y=yzer, color='black', linestyle='--', linewidth=1, zorder=1)
        ax.text(1.02, yzer + 0.10, "M=1,F=1", rotation=90, verticalalignment='bottom', transform=ax.get_yaxis_transform())
        ax.text(1.02, yzer - 0.10, "M=0,F=0", rotation=90, verticalalignment='top', transform=ax.get_yaxis_transform())

        ax.set_title(titl)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Legend
        # ------
        mark_size = 9
        legend_elements = []
        if not omit_const:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='const', markerfacecolor=colors['const'], markersize=mark_size))
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', label='sexe [M=0;F=1]', markerfacecolor=colors['sexe'], markersize=mark_size),
            Line2D([0], [0], marker='o', color='w', label='mbre [G=0;D=1]', markerfacecolor=colors['mbre'], markersize=mark_size),
            Line2D([0], [0], marker='o', color='w', label='age', markerfacecolor=colors['age'], markersize=mark_size)])
        #
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title='coef=x(independant)')
             
        # Plot saving
        # -----------
        # Right-aligned footnote
        plt.figtext(0.98, 0.02, foot, horizontalalignment='right', fontsize=8)
        # Adjust the layout to make room for the legend
        # plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.tight_layout()
        plt.show()
        pass
    
def mist_logi01_plot_01_perp_a(what, df_logit):
    
    # ====
    # AA : "Forest Plot of Odds Ratios"
    # ====
    
    # Data
    # ----
    df = df_logit.copy()
    print (df)
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'coef', '[0.025', '0.975]', 'P>|z|']]
    df['coef'] = pd.to_numeric(df['coef'], errors='coerce')
    df['[0.025'] = pd.to_numeric(df['[0.025'], errors='coerce')
    df['0.975]'] = pd.to_numeric(df['0.975]'], errors='coerce')
    df['P>|z|'] = pd.to_numeric(df['P>|z|'], errors='coerce')
    
    # Exec
    # ----   
    df['odds_ratio'] = np.exp(df['coef'])
    df['ci_lower'] = np.exp(df['[0.025'])
    df['ci_upper'] = np.exp(df['0.975]'])
    #
    df['circ'] = df['odds_ratio'] # the dots
    df['segm'] = df.apply(lambda row: (row['odds_ratio'] - row['ci_lower'], row['ci_upper'] - row['odds_ratio']), axis=1) # the segments
    df['anno'] = df['ci_lower'] # the annotation location
    foot = "footnote A"
    ylab = 'Odds Ratio'
    ysca = 'log'
    yzer = 1
    titl = "Forest Plot of Odds Ratios"
    print(df)
    mist_logi01_anal11x(df, ylab, ysca, yzer, titl, foot, omit_const=True, link_coef=True, line_alpha=0.5, line_width=2, area_alpha=0.1)
    pass
    # mist_logi01_anal11a(df, omit_const=True, link_coef=True, line_alpha=0.5, line_width=2, area_alpha=0.1)

def mist_logi01_plot_01_perp_b1(what, df_logit):
    # ====
    # B1 : "Coefficient Plot with Standard Errors"
    # ====
    
    # Data
    # ----
    df = df_logit.copy()
    print (df)
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'coef', 'std err', 'P>|z|']]
    df['coef'] = pd.to_numeric(df['coef'], errors='coerce')
    df['std err'] = pd.to_numeric(df['std err'], errors='coerce')
    df['P>|z|'] = pd.to_numeric(df['P>|z|'], errors='coerce')
    # mist_logi01_anal11b(df, omit_const=True, link_coef=True, line_alpha=0.5, line_width=2, area_alpha=0.1)
    df['circ'] = df['coef']
    df['segm'] = df['std err']
    df['anno'] = df['coef'] - df['std err']
    foot = "AB"
    ylab = 'Coefficient'
    ysca = 'linear'
    yzer = 0
    titl = "Coefficient Plot with Standard Errors"
    mist_logi01_anal11x(df, ylab, ysca, yzer, titl, foot, omit_const=True, link_coef=True, line_alpha=0.5, line_width=2, area_alpha=0.1)
    pass

def mist_logi01_plot_01_perp_b2(what, df_logit):
    # ==============
    # 2
    # ==============
    def zplot_logit(df, area=0.95):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        norm = scs.norm()
        x = np.linspace(-5, 5, 1000)
        y = norm.pdf(x)
        ax.plot(x, y, color='black', alpha=0.5)
        
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines([left, right], 0, norm.pdf([left, right]), color='grey', linestyle='--')
        ax.fill_between(x, 0, y, color='grey', alpha=0.25, where=(x > left) & (x < right))
        
        colors = {'const': 'red', 'sexe': 'blue', 'mbre': 'green', 'age': 'purple'}
        
        for _, row in df.iterrows():
            z_score = row['z']
            color = colors.get(row['coef=x(independant)'], 'black')
            ax.vlines(z_score, 0, norm.pdf(z_score), color=color, alpha=0.7)
            ax.scatter(z_score, norm.pdf(z_score), color=color, s=50, zorder=3)
            ax.annotate(f"{row['ceap=y(dependant)']}_{row['coef=x(independant)']}\nz={z_score:.2f}", 
                        (z_score, norm.pdf(z_score)), 
                        xytext=(0, 5), textcoords='offset points', 
                        ha='center', va='bottom', rotation=90, fontsize=8)
        
        ax.text(left, norm.pdf(left), f"z = {left:.3f}", fontsize=12, rotation=90, va="bottom", ha="right")
        ax.text(right, norm.pdf(right), f"z = {right:.3f}", fontsize=12, rotation=90, va="bottom", ha="left")
        ax.text(0, 0.1, f"shaded area = {area:.3f}", fontsize=12, ha='center')
        
        ax.set_xlabel('Z-Score')
        ax.set_ylabel('Probability Density')
        ax.set_title('Z-Score Plot of Logistic Regression Results')
        plt.show()

    if False:
        df = pd.DataFrame({
            'ceap=y(dependant)': ['NA', 'NA', 'NA', 'NA', 'C0', 'C0', 'C0', 'C0'],
            'coef=x(independant)': ['const', 'sexe', 'mbre', 'age', 'const', 'sexe', 'mbre', 'age'],
            'z': [-4.008, -1.453, 2.571, -1.187, -4.861, -0.570, 2.218, -0.407]
        })

    # Data
    # ----
    df = df_logit.copy()
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'z']]
    df['z'] = pd.to_numeric(df['z'], errors='coerce')

    # Exec
    # ----
    zplot_logit(df)
    pass

    # =============
    # B3
    # =============

def mist_logi01_plot_01_perp_b3(what, df_logit):
    def pvalue_heatmap(df):
        # Pivot the dataframe to create a matrix of p-values
        pivot_df = df.pivot(index='ceap=y(dependant)', columns='coef=x(independant)', values='P>|z|')
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd_r', vmin=0, vmax=0.05, center=0.025)
        
        # Customize the plot
        plt.title('P-value Heatmap of Logistic Regression Results')
        plt.xlabel('Coefficients')
        plt.ylabel('CEAP Categories')
        
        # Add color bar
        cbar = plt.gcf().axes[-1]
        cbar.set_ylabel('P-value', rotation=270, labelpad=15)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    if False:
        df = pd.DataFrame({
            'ceap=y(dependant)': ['NA', 'NA', 'NA', 'NA', 'C0', 'C0', 'C0', 'C0'],
            'coef=x(independant)': ['const', 'sexe', 'mbre', 'age', 'const', 'sexe', 'mbre', 'age'],
            'P>|z|': [0.000, 0.146, 0.010, 0.235, 0.000, 0.568, 0.027, 0.684]
        })

    # Data
    # ----
    df = df_logit.copy()
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'P>|z|']]
    df['P>|z|'] = pd.to_numeric(df['P>|z|'], errors='coerce')
    pvalue_heatmap(df)
    pass

def mist_logi01_plot_01_perp_b4(what, df_logit):
    # ====
    # B4 : "Coefficient Range Plot"
    # ====
    df = df_logit.copy()
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'coef', '[0.025', '0.975]', 'P>|z|']]
    df['coef'] = pd.to_numeric(df['coef'], errors='coerce')
    df['[0.025'] = pd.to_numeric(df['[0.025'], errors='coerce')
    df['0.975]'] = pd.to_numeric(df['0.975]'], errors='coerce')
    df['P>|z|'] = pd.to_numeric(df['P>|z|'], errors='coerce')

    # Exec
    # ----   
    df['circ'] = df['coef']  # the dots
    df['segm'] = df.apply(lambda row: (row['coef'] - row['[0.025'], row['0.975]'] - row['coef']), axis=1)  # the segments
    df['anno'] = df['[0.025']  # the annotation location

    foot = "footnote B4"
    ylab = 'Coefficient Value'
    ysca = 'linear'  # Coefficients are typically plotted on a linear scale
    yzer = 0  # Coefficient of 0 indicates no effect
    titl = "Coefficient Range Plot"
    print(df)
    mist_logi01_anal11x(df, ylab, ysca, yzer, titl, foot, omit_const=True, link_coef=True, line_alpha=0.5, line_width=2, area_alpha=0.1)
    pass

    # ====
    # B5 : "Forest Plot of Z-scores"
    # ====
def mist_logi01_plot_01_perp_b5(what, df_logit):
    
    df = df_logit.copy()
    print(df)
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'coef', 'std err', 'P>|z|']]
    df['coef'] = pd.to_numeric(df['coef'], errors='coerce')
    df['std err'] = pd.to_numeric(df['std err'], errors='coerce')
    df['P>|z|'] = pd.to_numeric(df['P>|z|'], errors='coerce')

    # Exec
    # ----   
    df['z_score'] = df['coef'] / df['std err']
    df['ci_lower'] = df['z_score'] - 1.96  # 95% CI
    df['ci_upper'] = df['z_score'] + 1.96  # 95% CI

    df['circ'] = df['z_score']  # the dots
    df['segm'] = df.apply(lambda row: (1.96, 1.96), axis=1)  # the segments (symmetric for z-scores)
    df['anno'] = df['ci_lower']  # the annotation location

    foot = "footnote B5"
    ylab = 'Z-score'
    ysca = 'linear'  # Z-scores are typically plotted on a linear scale
    yzer = 0  # Z-score of 0 indicates no effect
    titl = "Forest Plot of Z-scores"
    print(df)
    mist_logi01_anal11x(df, ylab, ysca, yzer, titl, foot, omit_const=True, link_coef=True, line_alpha=0.5, line_width=2, area_alpha=0.1)
    pass

    # ======
    # B6
    # ======

def mist_logi01_plot_01_perp_b6(what, df_logit):
    
    df = df_logit.copy()
    print(df)
    df = df.reset_index(drop=True)
    df = df[~df['coef=x(independant)'].str.contains('-', na=False)]
    df = df[['ceap=y(dependant)', 'coef=x(independant)', 'coef', 'std err', 'P>|z|']]
    df['coef'] = pd.to_numeric(df['coef'], errors='coerce')
    df['std err'] = pd.to_numeric(df['std err'], errors='coerce')

    # Exec
    # ----    
    def plot_standardized_coefficients(df):
        # Assuming df has columns: 'coef=x(independant)', 'ceap=y(dependant)', 'coef', 'std err'
        
        # Calculate standardized coefficients (using X-standardization for simplicity)
        df['std_coef'] = df['coef'] * df['std err']
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.pointplot(x='std_coef', y='coef=x(independant)', hue='ceap=y(dependant)', data=df, join=False)
        
        plt.title('Standardized Coefficient Plot')
        plt.xlabel('Standardized Coefficient')
        plt.ylabel('Predictors')
        plt.axvline(x=0, color='black', linestyle='--')
        
        plt.tight_layout()
        plt.show()
        pass

    # Example usage
    plot_standardized_coefficients(df)
    pass

def mist_logi01_plot_01_perp_deta(what, df_logit):
    if True:
        mist_logi01_plot_01_perp_a(what, df_logit)
        mist_logi01_plot_01_perp_b1(what, df_logit)
        mist_logi01_plot_01_perp_b2(what, df_logit)
        mist_logi01_plot_01_perp_b3(what, df_logit)
        mist_logi01_plot_01_perp_b4(what, df_logit)
        mist_logi01_plot_01_perp_b5(what, df_logit)
    mist_logi01_plot_01_perp_b6(what, df_logit)

    pass

# ====
# GLOB plots
# ====
def mist_logi01_plot_01_perp_glob(what, df_logit):

    # Data
    categories = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    llr = [0.016699, 0.133920, 0.926443, 0.411061, 0.155396, 0.213902, 0.382461, 0.008646]
    r_squared = [0.015927, 0.011792, 0.003938, 0.004685, 0.005005, 0.007350, 0.007649, 0.012518]
    aic = [640.143006, 475.629733, 125.700333, 619.044794, 1048.568190, 613.284256, 405.004502, 927.714927]
    bic = [659.244471, 494.731197, 144.801797, 638.146259, 1067.669655, 632.385721, 424.105967, 946.816392]

    # ----
    # Grph
    # ----
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7.5)) # x, y
    
    # -------------------------
    # Plot 1: LLR and R-squared
    # -------------------------
    ax1.bar(categories, llr, alpha=0.7, label='LLR p-value', zorder=2)
    ax1.set_ylabel('LLR p-value', fontsize=10)
    ax1.set_title('LLR p-value and R-squared', fontsize=10)
    ax1.axhline(y=0.05, color='r', linestyle='--', label='Significance level (0.05)', linewidth=1, zorder=3)
    ax1.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    #
    ax1_twin = ax1.twinx()
    ax1_twin.plot(categories, r_squared, 'go-', label='R-squared', zorder=3)
    ax1_twin.set_ylabel('R-squared', fontsize=10)
    ax1_twin.grid(False)
    #
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # -------------------
    # Plot 2: AIC and BIC
    # -------------------
    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width/2, aic, width, label='AIC', alpha=0.7)
    ax2.bar(x + width/2, bic, width, label='BIC', alpha=0.7)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('AIC and BIC Scores', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.tick_params(axis='both', labelsize=10)
    # 
    ax2.legend()

    # ----------------------------
    # Plot 3: R-squared comparison
    # ----------------------------
    ax3.bar(categories, r_squared, alpha=0.7)
    ax3.set_ylabel('R-squared', fontsize=10)
    ax3.set_title('R-squared Values', fontsize=10)
    ax3.set_ylim(0, max(r_squared) * 1.1)  # Set y-limit to 110% of max value
    #
    for i, v in enumerate(r_squared):
        ax3.text(i, v, f'{v:.6f}', ha='center', va='bottom')
        ax3.tick_params(axis='both', labelsize=10)
    
    # ------
    # Common
    # ------
    fig.suptitle('Analysis of CEAP Classifications\nStatistical Measures and Model Fit\nGlobal Results Summary', fontsize=11)
    fig.text(0.98, 0.02, 'Data collected from venous disease patients', ha='right', fontsize=8)
    # Layout margin variables
    left_margin = 0
    bottom_margin = 0.03
    right_margin = 1
    top_margin = 0.99
    vertical_padding = 1
    plt.tight_layout(rect=[left_margin, bottom_margin, right_margin, top_margin], h_pad=vertical_padding)
    plt.show()
    pass
'''
This script creates three plots:

1. A combined plot showing LLR p-values as bars and R-squared values as a line graph. This allows for easy comparison of statistical significance (with a red dashed line at 0.05) and model fit across CEAP classifications.
2. A grouped bar chart comparing AIC and BIC scores for each CEAP classification. This helps in model comparison, with lower scores indicating better models.
3. A bar chart focusing on R-squared values, with exact values displayed above each bar. This provides a clear visualization of the explanatory power of each model.
'''