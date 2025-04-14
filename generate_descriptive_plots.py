import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
DATA_FILE = 'results/responses_with_scores.csv'
OUTPUT_DIR = 'plots'
SCORE_COLUMNS = {
    'score_risque_chaleur': 'Risque Chaleur\nPerçu', # Added newline for better spacing if needed
    'score_efficacite': 'Efficacité\nRéponse Attendue',
    'score_attitude_env': 'Attitude\nEnvironnementale',
    'score_intention_env': 'Intention\nEnvironnementale',
    'score_comportements': 'Comportements\nPro-Env.'
}
THEME_NAMES_FRENCH = list(SCORE_COLUMNS.values()) # Will use renamed columns later
COLUMN_NAMES = list(SCORE_COLUMNS.keys())

# --- Apply Professional Style & Font Sizes ---
plt.style.use('seaborn-v0_8-ticks') # Changed style to 'ticks' for a cleaner academic look
plt.rcParams.update({
    'font.size': 10,             # Base font size (Reduced)
    'axes.titlesize': 14,        # Title font size (Reduced)
    'axes.labelsize': 14,        # Axis label font size (Increased)
    'xtick.labelsize': 12,       # X-tick label size (Increased)
    'ytick.labelsize': 12,       # Y-tick label size (Increased)
    'figure.titlesize': 16,      # Figure title font size (Reduced)
    'legend.fontsize': 10,       # Legend font size (Reduced)
})

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Erreur: Le fichier {DATA_FILE} n'a pas été trouvé.")
    exit()
except Exception as e:
    print(f"Erreur lors de la lecture du fichier CSV: {e}")
    exit()

# Select only the score columns
scores_df = df[COLUMN_NAMES].copy()
scores_df.columns = THEME_NAMES_FRENCH # Rename columns for plotting

# --- Calculate Statistics ---
means = scores_df.mean()
std_devs = scores_df.std()

# --- Plot 1: Bar Chart with Error Bars ---
plt.figure(figsize=(12, 8)) # Increased figure size
bars = plt.bar(means.index, means.values, yerr=std_devs.values, capsize=7, color='darkgrey', alpha=0.9, width=0.6) # Changed color for ticks style
plt.ylabel('Score Moyen (Échelle Likert 1-5)')
plt.title('Moyenne et Écart-Type par Thème', pad=20) # Removed "des Scores"
plt.xticks(rotation=0, ha='center') # Adjusted rotation for potentially longer labels
plt.ylim(0, 6) # Adjusted ylim slightly for text
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.axhline(3, color='grey', linestyle=':', linewidth=1.5, zorder=0, label='Point Neutre (3)') # Added horizontal line at y=3
# Add mean values on top of bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    theme_name = means.index[i] # Get the theme name corresponding to the bar index
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + std_devs[theme_name] + 0.15, f'{yval:.2f}', va='bottom', ha='center', fontsize=11) # Increased font size for text on bars

plt.tight_layout(pad=1.5) # Added padding
plt.savefig(os.path.join(OUTPUT_DIR, 'bar_chart_means_stddev.png'), dpi=300, bbox_inches='tight') # Added DPI and bbox_inches
print(f"Graphique 'bar_chart_means_stddev.png' sauvegardé dans {OUTPUT_DIR}")
plt.close()

# --- Plot 2: Boxplots ---
plt.figure(figsize=(8, 7)) # Made figure more compact (less wide)
sns.boxplot(data=scores_df, palette='Greys', width=0.6, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}) # Slightly wider boxes
plt.ylabel('Score (Échelle Likert 1-5)') # Use rcParams
plt.title('Distribution des niveaux de réponse par thème', pad=20) # Use rcParams
plt.xticks(rotation=30, ha='right') # Use rcParams
plt.yticks() # Use rcParams
plt.ylim(0, 6) # Adjusted ylim
# plt.grid(axis='y', linestyle='--', alpha=0.6) # Keep grid commented out for ticks style
sns.despine(trim=True) # Remove top and right spines for cleaner look
plt.tight_layout(pad=2.0) # Increased padding slightly for larger fonts
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplots_themes.png'), dpi=300, bbox_inches='tight') # Added DPI and bbox_inches
print(f"Graphique 'boxplots_themes.png' sauvegardé dans {OUTPUT_DIR}")
plt.close()

# --- Plot 3: Radar Chart ---
labels = means.index.to_list()
stats = means.values.flatten().tolist()

# Ensure the data is circular
stats += stats[:1]
labels_radar = labels + labels[:1] # Close the loop

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1] # Close the loop for plotting

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True)) # Increased size
ax.fill(angles, stats, 'darkgrey', alpha=0.35) # Changed color
ax.plot(angles, stats, 'black', linewidth=1.8) # Changed color and thickness

# Set axis limits and labels
ax.set_yticks(np.arange(1, 6, 1)) # Likert 1-5
ax.set_yticklabels([str(i) for i in np.arange(1, 6, 1)]) # Use rcParams
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels) # Use rcParams
plt.ylim(0, 6) # Adjusted ylim
plt.title('Profil Moyen des Réponses par Thème', y=1.12) # Use rcParams

# plt.tight_layout() # Often not needed/problematic with polar plots + bbox_inches
plt.savefig(os.path.join(OUTPUT_DIR, 'radar_chart_means.png'), dpi=300, bbox_inches='tight') # Added DPI and bbox_inches
print(f"Graphique 'radar_chart_means.png' sauvegardé dans {OUTPUT_DIR}")
plt.close()

print("Génération des graphiques terminée.")