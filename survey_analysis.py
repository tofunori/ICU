import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pingouin as pg
# Ensure plots directory exists
os.makedirs("results", exist_ok=True) # Updated directory

# Load data
df = pd.read_csv("data/responses_recoded.csv") # Updated path

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
    plt.savefig(f"results/{col}_distribution.png") # Updated path
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
        plt.savefig(f"results/{col}_hist.png") # Updated path
        plt.close()

# 3. Relationship: crainte_futur by age group (boxplot)
if "crainte_futur" in df.columns and "age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="age", y="crainte_futur", data=df)
    plt.title("Crainte futur by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Crainte futur")
    plt.tight_layout()
    plt.savefig("results/crainte_futur_by_age.png") # Updated path
    plt.close()

# 4. Relationship: intention_frequent_parcs by parcs_efficace (scatter)
if "intention_frequent_parcs" in df.columns and "parcs_efficace" in df.columns:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="parcs_efficace", y="intention_frequent_parcs", data=df)
    plt.title("Intention to Frequent Parks vs. Perceived Park Effectiveness")
    plt.xlabel("Parcs efficace")
    plt.ylabel("Intention frequent parcs")
    plt.tight_layout()
    plt.savefig("results/intention_vs_parcs_efficace.png") # Updated path
    plt.close()

# 5. Relationship: vulnerability by housing type (boxplot)
if "vulnerable" in df.columns and "logement" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="logement", y="vulnerable", data=df)
    plt.title("Vulnerability by Housing Type")
    plt.xlabel("Housing Type")
    plt.ylabel("Vulnerability")
    plt.tight_layout()
    plt.savefig("results/vulnerable_by_logement.png") # Updated path
    plt.close()

# 6. Summary statistics for Likert columns
summary_stats = df[likert_cols].describe().transpose()
summary_stats.to_csv("data/likert_summary_stats.csv") # Updated path
# --- Cronbach's Alpha Calculation ---

# Function to calculate and print Cronbach's Alpha
def calculate_and_print_cronbach_alpha(df):
    """
    Calculates and prints Cronbach's Alpha for predefined dimensions.

    Args:
        df (pd.DataFrame): DataFrame containing the survey responses,
                           with columns corresponding to Likert scale items.
    """
    # Check if pingouin is available (import already added at top)
    # Define the items for each dimension based on the previous analysis/image
    dimensions = {
        "Risque lié à la chaleur": [
            'vagues_intenses', 'conseq_sante', 'menace_env', 'vulnerable',
            'logement_protection', 'crainte_futur'
        ],
        "Efficacité perçue des stratégies": [ # Updated based on CSV headers
            'parcs_efficace', 'eau_publique', 'arbres_inutile',
            'difficile_comportements', 'acces_parcs', 'transport_chaud'
            # Note: 'auto_efficace', 'acces_verts' from image not found in CSV
        ],
        "Attitude environnementale": [
            'valoriser_espaces_verts', 'soutenir_politiques', 'prefere_interieur',
            'contre_reduction_voiture', 'espaces_agreables', 'proteger_acces_eau',
            'municipalite_responsable'
        ],
        "Intention environnementale": [ # Updated based on CSV headers
            'intention_frequent_parcs', 'intention_soutenir_initiatives',
            'pas_intention_s_informer', 'intention_comportements_eco', # Was 'intention_comportement' in image
            'pas_intention_transport_actif', 'motivation_si_municipalite'
        ],
        "Comportements observés": [ # Updated based on CSV headers
            'freq_parcs', 'activite_verdissement', 'petition_contact', 'reduire_eau', # Was 'reduction_eau' in image
            'transport_alternatif', 'compost', 'refroidissement_passif' # Was 'transport_actif' in image
        ]
    }

    print("\n--- Cronbach's Alpha Internal Consistency Analysis ---")
    print("-" * 55)
    all_alphas_valid = True
    for dim_name, items in dimensions.items():
        # Check if all items for the dimension exist in the DataFrame
        missing_items = [item for item in items if item not in df.columns]
        if missing_items:
            print(f"Warning: Dimension '{dim_name}' - Missing columns: {', '.join(missing_items)}. Skipping alpha calculation.")
            all_alphas_valid = False
            print("-" * 55)
            continue

        # Select the relevant columns for the current dimension
        # Use .copy() to avoid SettingWithCopyWarning later if modifications were needed
        dimension_data = df[items].copy()

        # Ensure data is numeric, coercing errors (might be redundant if done globally, but safe)
        for item in items:
             dimension_data[item] = pd.to_numeric(dimension_data[item], errors='coerce')

        # Drop rows with any missing values *within that dimension's items*
        dimension_data_cleaned = dimension_data.dropna()
        n_complete = len(dimension_data_cleaned)
        n_total = len(dimension_data)

        print(f"Dimension: {dim_name}")
        # Limit the number of items printed for brevity if too many
        if len(items) > 5:
             print(f"  Items: {', '.join(items[:3])}, ..., {items[-1]}")
        else:
             print(f"  Items: {', '.join(items)}")


        if n_complete < 2:
             print(f"  Result: Not enough valid data ({n_complete} complete responses out of {n_total}) to calculate alpha.")
             all_alphas_valid = False
        elif dimension_data_cleaned.shape[1] < 2:
             print(f"  Result: Needs at least two items to calculate alpha (found {dimension_data_cleaned.shape[1]}).")
             all_alphas_valid = False
        else:
            # Calculate Cronbach's alpha
            try:
                # Note: pingouin returns alpha, confidence interval, etc.
                alpha_results = pg.cronbach_alpha(data=dimension_data_cleaned)
                alpha_value = alpha_results[0]
                confidence_interval = alpha_results[1] # 95% CI

                print(f"  Cronbach's Alpha = {alpha_value:.3f}")
                # Optional: Print confidence interval
                # print(f"  95% CI = [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
                print(f"  (Based on {n_complete} complete responses for this dimension out of {n_total} total)")

                # Item analysis code removed.

            # Correctly indented except blocks
            except ValueError as ve:
                 # Catch specific errors like 'Input must contain at least two columns'
                 print(f"  Error calculating alpha for '{dim_name}': {ve}")
                 all_alphas_valid = False
            except Exception as e:
                 # Catch any other unexpected errors during calculation
                 print(f"  Unexpected error calculating alpha for '{dim_name}': {e}")
                 all_alphas_valid = False

        # Correctly indented print statement (aligned with the 'if n_complete < 2:' block)
        print("-" * 55)

    if not all_alphas_valid:
        print("Note: Some alpha calculations could not be performed or encountered errors.")
    print("--- End Cronbach's Alpha Analysis ---\n")

# Call the function after data loading and cleaning
calculate_and_print_cronbach_alpha(df)

# Modify the final print statement to reflect the added analysis
print("Analysis complete. Plots saved in 'results/', summary statistics in 'data/', and Cronbach's Alpha printed.")
print("Analysis complete. Plots saved in 'results/' and summary statistics in 'data/'.") # Updated message
