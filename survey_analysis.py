import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load data
df = pd.read_csv("responses_recoded.csv")

# Basic data cleaning
# Replace empty strings with NaN
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Convert relevant columns to categorical
categorical_cols = [
    "age", "genre", "niveau_etude", "ville", "quartier", "logement", "climatisation"
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Convert Likert scale columns to numeric
likert_cols = [
    "vagues_intenses", "conseq_sante", "menace_env", "vulnerable", "logement_protection",
    "crainte_futur", "parcs_efficace", "eau_publique", "arbres_inutile", "trouver_endroits_frais",
    "difficile_comportements", "acces_parcs", "transport_chaud", "valoriser_espaces_verts",
    "soutenir_politiques", "prefere_interieur", "contre_reduction_voiture", "espaces_agreables",
    "proteger_acces_eau", "municipalite_responsable", "intention_frequent_parcs",
    "intention_soutenir_initiatives", "pas_intention_s_informer", "intention_comportements_eco",
    "pas_intention_transport_actif", "motivation_si_municipalite"
]
for col in likert_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Descriptive statistics ---

# 1. Demographics: Age, Genre, Education, City
demographics = ["age", "genre", "niveau_etude", "ville", "logement"]
for col in demographics:
    plt.figure(figsize=(8, 4))
    df[col].value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/{col}_distribution.png")
    plt.close()

# 2. Key perceptions: e.g., vagiues_intenses, menace_env, crainte_futur
perception_cols = ["vagues_intenses", "menace_env", "crainte_futur", "municipalite_responsable"]
for col in perception_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        df[col].dropna().astype(float).plot.hist(bins=range(1, 8), rwidth=0.8)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"plots/{col}_hist.png")
        plt.close()

# 3. Relationship: crainte_futur by age group (boxplot)
if "crainte_futur" in df.columns and "age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="age", y="crainte_futur", data=df)
    plt.title("Crainte futur by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Crainte futur")
    plt.tight_layout()
    plt.savefig("plots/crainte_futur_by_age.png")
    plt.close()

# 4. Relationship: intention_frequent_parcs by parcs_efficace (scatter)
if "intention_frequent_parcs" in df.columns and "parcs_efficace" in df.columns:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="parcs_efficace", y="intention_frequent_parcs", data=df)
    plt.title("Intention to Frequent Parks vs. Perceived Park Effectiveness")
    plt.xlabel("Parcs efficace")
    plt.ylabel("Intention frequent parcs")
    plt.tight_layout()
    plt.savefig("plots/intention_vs_parcs_efficace.png")
    plt.close()

# 5. Relationship: vulnerability by housing type (boxplot)
if "vulnerable" in df.columns and "logement" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="logement", y="vulnerable", data=df)
    plt.title("Vulnerability by Housing Type")
    plt.xlabel("Housing Type")
    plt.ylabel("Vulnerability")
    plt.tight_layout()
    plt.savefig("plots/vulnerable_by_logement.png")
    plt.close()

# 6. Summary statistics for Likert columns
summary_stats = df[likert_cols].describe().transpose()
summary_stats.to_csv("plots/likert_summary_stats.csv")

print("Analysis complete. Plots and summary statistics saved in the 'plots/' directory.")
