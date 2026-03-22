import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Define patients and limbs
num_patients = 40
num_limbs = 60
timepoints = ['T0', 'T1', 'T2']

# Create patient IDs and assign limbs to them
patient_ids = np.arange(num_patients)
limb_patient_map = list(np.repeat(patient_ids[:20], 2)) + list(patient_ids[20:])
limb_ids = np.arange(num_limbs)

# --- VCSS Mock Data ---
vcss_data = []
for l_idx, p_id in enumerate(limb_patient_map):
    base_vcss = np.random.normal(8.5, 2.0)
    t1_vcss = base_vcss - np.random.normal(5.0, 1.0)
    t2_vcss = t1_vcss - np.random.normal(1.0, 0.5)
    vcss_scores = np.clip([base_vcss, t1_vcss, t2_vcss], 0, 30).round().astype(int)
    for i, tp in enumerate(timepoints):
        vcss_data.append({
            'patient_id': p_id,
            'limb_id': l_idx,
            'timepoint': tp,
            'vcss_score': vcss_scores[i]
        })

df_vcss = pd.DataFrame(vcss_data)

sns = False
if sns:
    # --- PLOT VCSS ---
    plt.figure(figsize=(5, 5))
    # Use classic matplotlib style
    plt.style.use('classic')

    # Boxplot with matplotlib
    boxes = plt.boxplot(
        [df_vcss[df_vcss['timepoint'] == tp]['vcss_score'] for tp in timepoints],
        widths=0.4,
        patch_artist=True,   # Fill boxes with color
        labels=timepoints,   # Label x-axis
        showmeans=True,      # Show mean as a marker
        meanline=False,      # Do not show mean as a line
        showfliers=True,     # Show outliers
        whiskerprops=dict(linestyle='--', linewidth=0.5, color='gray'),  # Whisker style
        boxprops=dict(linestyle='-', linewidth=0.5, color='black'),       # Box edge style
        medianprops=dict(linestyle='-', linewidth=1, color='orange'),        # Median line style
        meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='firebrick', markersize=8),  # Mean marker style
        capprops=dict(linestyle='-', linewidth=0.5, color='gray'),        # Cap style
        flierprops=dict(marker='o', markeredgecolor='gray', markerfacecolor='lightgray', markersize=5, alpha=0.9)  # Outlier style
    )

    # Customize box colors
    colors = ['lightblue', 'cornflowerblue', 'royalblue']
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)

    # Scatter plot for individual points
    for i, tp in enumerate(timepoints):
        y = df_vcss[df_vcss['timepoint'] == tp]['vcss_score']
        x = np.random.normal(i + 1, 0.04, size=len(y))
        plt.scatter(x, y, color='gray', alpha=0.6, s=20, edgecolors='none')

    # Customize plot
    plt.title('Evolution of Venous Clinical Severity Score (VCSS)', fontsize=14, fontweight='bold')
    plt.xlabel('Timepoint (T0 = Surgery)', fontsize=12)
    plt.ylabel('Total VCSS Score', fontsize=12)
    plt.xticks([1, 2, 3], timepoints)

    # Customize grid
    plt.grid(
        axis='y',          # Only show grid lines for the y-axis
        linestyle='--',    # Dashed lines
        alpha=0.4,         # Transparency (0 = invisible, 1 = opaque)
        color='gray',      # Grid line color
        linewidth=0.7      # Grid line width
    )

    # Optional: Customize x-axis grid (if needed)
    grid_x = False
    if grid_x:
        plt.grid(
            axis='x',
            linestyle=':',
            alpha=0.2,
            color='lightgray',
            linewidth=0.5
        )

    # Save and show
    plt.savefig('vcss_evolution_matplotlib.png', dpi=200, bbox_inches='tight')
    plt.show()
else:

    # --- PLOT VCSS ---
    plt.figure(figsize=(5, 5))
    plt.style.use('classic')  # Style classique matplotlib

    # Boxplot
    boxes = plt.boxplot(
        [df_vcss[df_vcss['timepoint'] == tp]['vcss_score'] for tp in timepoints],
        widths=0.4,
        patch_artist=True,  # Remplir les boîtes avec une couleur
        labels=timepoints,  # Étiquettes de l'axe x
        showmeans=True,     # Afficher la moyenne
        meanline=False,     # Ne pas afficher la moyenne comme une ligne
        showfliers=True,    # Afficher les valeurs aberrantes
        whiskerprops=dict(linestyle='--', linewidth=0.7, color='gray'),  # Style des moustaches
        boxprops=dict(linestyle='-', linewidth=0.7, color='black'),      # Style des boîtes
        medianprops=dict(linestyle='-', linewidth=1.5, color='darkorange'),  # Style de la médiane
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick', markersize=6),  # Style du marqueur de la moyenne
        capprops=dict(linestyle='-', linewidth=0.7, color='gray'),        # Style des caps
        flierprops=dict(marker='o', markeredgecolor='gray', markerfacecolor='lightgray', markersize=4, alpha=0.7)  # Style des valeurs aberrantes
    )

    # Couleurs des boîtes
    colors = ['lightblue', 'cornflowerblue', 'royalblue']
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)

    # Nuage de points pour les valeurs individuelles
    for i, tp in enumerate(timepoints):
        y = df_vcss[df_vcss['timepoint'] == tp]['vcss_score']
        x = np.random.normal(i + 1, 0.03, size=len(y))  # Réduction du "jitter" pour plus de clarté
        plt.scatter(x, y, color='darkgray', alpha=0.5, s=15, edgecolors='none')

    # Personnalisation du graphique
    plt.title('Evolution of Venous Clinical Severity Score (VCSS)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Timepoint (T0 = Baseline)', fontsize=10)
    plt.ylabel('Total VCSS Score', fontsize=10)
    plt.xticks([1, 2, 3], timepoints, fontsize=10)

    # Grille
    plt.grid(
        axis='y',          # Grille uniquement pour l'axe y
        linestyle='--',    # Lignes en pointillés
        alpha=0.3,         # Transparence
        color='lightgray', # Couleur de la grille
        linewidth=0.6      # Épaisseur des lignes
    )

    # Sauvegarde et affichage
    plt.tight_layout()  # Ajuste automatiquement les marges
    plt.savefig('vcss_evolution_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.show()

