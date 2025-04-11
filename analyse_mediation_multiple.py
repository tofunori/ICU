import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- 1. Configuration ---

# !!! IMPORTANT: Vérifiez que ces noms correspondent aux entêtes EXACTES de votre fichier CSV/Excel !!!
# Si vos entêtes sont légèrement différents (ex: majuscules/minuscules), ajustez les noms ci-dessous.

# Noms des variables Python basés sur votre image:
# CORRECTION: Retrait des espaces superflus à la fin des noms de variables pour éviter des erreurs potentielles
var_X = 'score_risque_chaleur'          # Updated to match CSV header
var_M1 = 'score_efficacite'             # Updated to match CSV header
var_M2 = 'score_attitude_env'           # Updated to match CSV header
var_M3 = 'score_intention_env'          # Updated to match CSV header
var_Y = 'score_comportements'           # Updated to match CSV header

# --- Chargement des Données ---
# !!! MODIFIEZ LA LIGNE CI-DESSOUS si nécessaire pour indiquer le bon chemin et nom de votre fichier !!!
# Assurez-vous que le fichier est bien dans le dossier 'results'
df = pd.read_csv("results/responses_with_scores.csv") # Changed to read_csv and used forward slash

# --- Vérification (Optionnelle mais recommandée) ---
# Décommentez les lignes suivantes pour vérifier que les données sont bien chargées
print("Aperçu des données chargées:")
print(df.head())
print("\nColonnes disponibles:")
print(df.columns)
print(f"\nVariables utilisées dans l'analyse:")
print(f"X: {var_X}, M1: {var_M1}, M2: {var_M2}, M3: {var_M3}, Y: {var_Y}")
print("-" * 30)
# # S'il y a une erreur ici (ex: "KeyError"), cela signifie que les noms de variables ci-dessus
# # ne correspondent pas EXACTEMENT aux noms des colonnes dans votre fichier. Vérifiez les majuscules/minuscules, espaces, accents.

# --- Correlation Matrix Heatmap --- 
try:
    # Select only the columns used in the analysis
    cols_for_corr = [var_X, var_M1, var_M2, var_M3, var_Y]
    df_corr = df[cols_for_corr]
    
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    print("\nCorrelation Matrix (R values):")
    print(corr_matrix)
    
    # Create heatmap (enhanced for report)
    
    # Define mapping for clearer labels in the plot
    label_mapping = {
        var_X: 'Risque lié à la chaleur',
        var_M1: 'Efficacité perçue des stratégies',
        var_M2: 'Attitude environnementale',
        var_M3: 'Intention environnementale',
        var_Y: 'Comportements pro-environnementaux'
    }
    # Get the labels in the correct order of the correlation matrix columns/index
    plot_labels = [label_mapping.get(col, col) for col in corr_matrix.columns]

    plt.figure(figsize=(12, 10)) # Increased figure size further for longer labels
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                xticklabels=plot_labels, yticklabels=plot_labels, # Use mapped labels
                annot_kws={"size": 10})
    plt.title('Corrélations entre les cinq thèmes du modèle de médiation', fontsize=16) # Updated title
    plt.xticks(rotation=45, ha='right', fontsize=10) # Adjusted xticks font size/rotation
    plt.yticks(rotation=0, fontsize=12) # Increased yticks font size
    
    # Save the plot
    corr_plot_filename = 'plots/correlation_matrix.png' # Changed path to plots/
    plt.tight_layout()
    plt.savefig(corr_plot_filename, dpi=300) # Added dpi=300 for higher resolution
    print(f"\nCorrelation matrix plot saved as: {corr_plot_filename}")
    plt.close() # Close the figure to prevent interference with the next plot

except Exception as e:
    print(f"\nError generating correlation matrix plot: {e}")


# --- Pairplot for Visualizing Relationships with R/R^2 ---
try:
    print("\nGenerating pairplot with R/R^2 annotations...")
    
    # Function to calculate and annotate R and R^2
    def corrfunc(x, y, **kws):
        # Check if x and y are identical (diagonal) or if either has zero variance
        if x is y or x.var() == 0 or y.var() == 0:
            return
        r, _ = pearsonr(x, y)
        ax = plt.gca()
        # Use a smaller font size for annotations
        ax.annotate(f"R = {r:.2f}\nR² = {r**2:.2f}",
                    xy=(.1, .9), xycoords=ax.transAxes, fontsize=8) # Adjusted position and font size

    # Use the same selected columns as the correlation matrix
    pairplot = sns.pairplot(df_corr)
    pairplot.map_upper(corrfunc) # Map the function to the upper triangle
    pairplot.fig.suptitle('Pairwise Relationships Between Variables (with R/R²)', y=1.02) # Add title slightly above plot
    
    # Save the plot
    pairplot_filename = 'plots/scatter_pairplot_with_R.png' # Changed path to plots/
    pairplot.savefig(pairplot_filename)
    print(f"Pairplot saved as: {pairplot_filename}")
    plt.close(pairplot.fig) # Close the figure

except Exception as e:
    print(f"\nError generating pairplot: {e}")


# --- Le reste du code pour l'analyse de médiation ---

# Number of bootstrap samples for estimating confidence intervals
n_bootstrap = 5000 # Use at least 1000, 5000 is common

# --- 2. Define Mediation Models ---
# Wrap variable names with spaces in Q() for statsmodels formulas
model_m1_formula = f"Q('{var_M1}') ~ Q('{var_X}')"
model_m2_formula = f"Q('{var_M2}') ~ Q('{var_X}')"
model_m3_formula = f"Q('{var_M3}') ~ Q('{var_X}')"
model_y_formula = f"Q('{var_Y}') ~ Q('{var_X}') + Q('{var_M1}') + Q('{var_M2}') + Q('{var_M3}')"
model_total_effect_formula = f"Q('{var_Y}') ~ Q('{var_X}')"

# --- 3. Bootstrapping for Indirect Effects ---
bootstrap_indirect_effects = {
    'indirect_M1': [],
    'indirect_M2': [],
    'indirect_M3': [],
    'total_indirect': []
}
bootstrap_direct_effects = []
# bootstrap_total_effects = [] # Retiré car non utilisé dans le code fourni pour le CI

print(f"Starting bootstrapping with {n_bootstrap} samples...")

# Assurez-vous que 'df' est bien chargé et non vide avant cette boucle !
if 'df' not in locals() or df is None or df.empty:
    print("ERREUR: Le DataFrame 'df' n'a pas été chargé ou est vide. Vérifiez la section 'Chargement des Données'.")
else:
    # Vérification initiale de variance non nulle sur les données originales
    if df[var_X].var() == 0 or df[var_M1].var() == 0 or \
       df[var_M2].var() == 0 or df[var_M3].var() == 0 or \
       df[var_Y].var() == 0:
        print("ERREUR: Variance nulle dans une ou plusieurs colonnes des données originales. Impossible de continuer.")
    else:
        for i in range(n_bootstrap):
            df_sampled = df.sample(n=len(df), replace=True, random_state=i)

            # Vérifier la variance dans l'échantillon
            if df_sampled[var_X].var() == 0 or df_sampled[var_M1].var() == 0 or \
               df_sampled[var_M2].var() == 0 or df_sampled[var_M3].var() == 0 or \
               df_sampled[var_Y].var() == 0:
               # print(f"Warning: Skipping bootstrap sample {i+1} due to zero variance in sample.")
               continue # Passer à l'échantillon suivant

            try:
                # Fit models on the sampled data
                fit_m1 = smf.ols(model_m1_formula, data=df_sampled).fit()
                fit_m2 = smf.ols(model_m2_formula, data=df_sampled).fit()
                fit_m3 = smf.ols(model_m3_formula, data=df_sampled).fit()
                fit_y = smf.ols(model_y_formula, data=df_sampled).fit()

                # Extract parameters using Q()-wrapped names
                a1 = fit_m1.params.get(f"Q('{var_X}')", 0)
                a2 = fit_m2.params.get(f"Q('{var_X}')", 0)
                a3 = fit_m3.params.get(f"Q('{var_X}')", 0)
                b1 = fit_y.params.get(f"Q('{var_M1}')", 0)
                b2 = fit_y.params.get(f"Q('{var_M2}')", 0)
                b3 = fit_y.params.get(f"Q('{var_M3}')", 0)
                c_prime = fit_y.params.get(f"Q('{var_X}')", 0)

                # Calculate indirect effects for this sample
                indirect_m1 = a1 * b1
                indirect_m2 = a2 * b2
                indirect_m3 = a3 * b3
                total_indirect = indirect_m1 + indirect_m2 + indirect_m3

                # Store results
                bootstrap_indirect_effects['indirect_M1'].append(indirect_m1)
                bootstrap_indirect_effects['indirect_M2'].append(indirect_m2)
                bootstrap_indirect_effects['indirect_M3'].append(indirect_m3)
                bootstrap_indirect_effects['total_indirect'].append(total_indirect)
                bootstrap_direct_effects.append(c_prime)

            except Exception as e:
                # Décommentez pour voir les erreurs spécifiques si le bootstrap échoue encore
                # print(f"Warning: Skipping bootstrap sample {i+1} due to error: {e}")
                pass

        print("Bootstrapping finished.")

        # --- 4. Calculate Confidence Intervals (95% CI using percentile method) ---
        print("\n--- Bootstrapped Confidence Intervals (95%) ---")
        results_ci = {}
        for key, values in bootstrap_indirect_effects.items():
            if values: # S'assurer que la liste n'est pas vide
                lower_bound = np.percentile(values, 2.5)
                upper_bound = np.percentile(values, 97.5)
                results_ci[key] = (lower_bound, upper_bound)
                print(f"{key.replace('_', ' ').title()}: [{lower_bound:.4f}, {upper_bound:.4f}]")
            else:
                 print(f"{key.replace('_', ' ').title()}: Could not calculate CI (no valid bootstrap samples)")

        if bootstrap_direct_effects: # S'assurer que la liste n'est pas vide
            lower_c_prime = np.percentile(bootstrap_direct_effects, 2.5)
            upper_c_prime = np.percentile(bootstrap_direct_effects, 97.5)
            results_ci['direct_effect_c_prime'] = (lower_c_prime, upper_c_prime)
            print(f"Direct Effect (c'): [{lower_c_prime:.4f}, {upper_c_prime:.4f}]")
        else:
            print(f"Direct Effect (c'): Could not calculate CI")

        # --- 5. Fit Models on Original Data for Point Estimates ---
        print("\n--- Point Estimates from Original Data ---")
        try:
            # La vérification de variance nulle a déjà été faite avant la boucle bootstrap
            fit_m1_orig = smf.ols(model_m1_formula, data=df).fit()
            fit_m2_orig = smf.ols(model_m2_formula, data=df).fit()
            fit_m3_orig = smf.ols(model_m3_formula, data=df).fit()
            fit_y_orig = smf.ols(model_y_formula, data=df).fit()
            fit_total_orig = smf.ols(model_total_effect_formula, data=df).fit()

            # Extract parameters using Q()-wrapped names
            a1_est = fit_m1_orig.params.get(f"Q('{var_X}')", 0)
            a2_est = fit_m2_orig.params.get(f"Q('{var_X}')", 0)
            a3_est = fit_m3_orig.params.get(f"Q('{var_X}')", 0)
            b1_est = fit_y_orig.params.get(f"Q('{var_M1}')", 0)
            b2_est = fit_y_orig.params.get(f"Q('{var_M2}')", 0)
            b3_est = fit_y_orig.params.get(f"Q('{var_M3}')", 0)
            c_prime_est = fit_y_orig.params.get(f"Q('{var_X}')", 0)
            c_total_est = fit_total_orig.params.get(f"Q('{var_X}')", 0)

            indirect_m1_est = a1_est * b1_est
            indirect_m2_est = a2_est * b2_est
            indirect_m3_est = a3_est * b3_est
            total_indirect_est = indirect_m1_est + indirect_m2_est + indirect_m3_est

            print(f"\nPath Estimates:")
            # Construct keys for accessing p-values correctly
            key_X = f"Q('{var_X}')"
            key_M1 = f"Q('{var_M1}')"
            key_M2 = f"Q('{var_M2}')"
            key_M3 = f"Q('{var_M3}')"

            # Displaying original header names for clarity in output and using constructed keys for p-values
            print(f"  a1 (Risque chaleur -> Efficacité stratégies): {a1_est:.4f} (p={fit_m1_orig.pvalues.get(key_X, np.nan):.3f})")
            print(f"  a2 (Risque chaleur -> Attitude env.): {a2_est:.4f} (p={fit_m2_orig.pvalues.get(key_X, np.nan):.3f})")
            print(f"  a3 (Risque chaleur -> Intention env.): {a3_est:.4f} (p={fit_m3_orig.pvalues.get(key_X, np.nan):.3f})")
            print(f"  b1 (Efficacité strat. -> Comportements | ...): {b1_est:.4f} (p={fit_y_orig.pvalues.get(key_M1, np.nan):.3f})")
            print(f"  b2 (Attitude env. -> Comportements | ...): {b2_est:.4f} (p={fit_y_orig.pvalues.get(key_M2, np.nan):.3f})")
            print(f"  b3 (Intention env. -> Comportements | ...): {b3_est:.4f} (p={fit_y_orig.pvalues.get(key_M3, np.nan):.3f})")
            print(f"  c' (Risque chaleur -> Comportements | Med.): {c_prime_est:.4f} (p={fit_y_orig.pvalues.get(key_X, np.nan):.3f})")
            print(f"  c  (Risque chaleur -> Comportements) Total Effect: {c_total_est:.4f} (p={fit_total_orig.pvalues.get(key_X, np.nan):.3f})")


            print(f"\nIndirect Effect Estimates (Point Estimates):")
            print(f"  Indirect via Efficacité stratégies (a1*b1): {indirect_m1_est:.4f}")
            print(f"  Indirect via Attitude env. (a2*b2): {indirect_m2_est:.4f}")
            print(f"  Indirect via Intention env. (a3*b3): {indirect_m3_est:.4f}")
            print(f"  Total Indirect Effect: {total_indirect_est:.4f}")

            print(f"\nVerification Check:")
            print(f"  Total Indirect + Direct = {total_indirect_est + c_prime_est:.4f}")
            print(f"  Total Effect (c)      = {c_total_est:.4f}")
            print(f"(Note: These should be close, differences due to model estimation variance)")

        except Exception as e:
            print(f"\nError fitting models on original data: {e}")
            print("Cannot provide point estimates.")

        # --- 6. Interpretation Guidance ---

# --- 7. Generate Plot of Indirect Effects --- 
# Check if results are available for plotting
if 'results_ci' in locals() and 'indirect_m1_est' in locals():
    try:
        # Prepare data for the plot
        labels = ['Indirect via M1\n(Efficacité strat.)', 
                  'Indirect via M2\n(Attitude env.)', 
                  'Indirect via M3\n(Intention env.)', 
                  'Effet Indirect Total'] # French label
        point_estimates = [indirect_m1_est, indirect_m2_est, indirect_m3_est, total_indirect_est]
        
        # Calculate error bar lengths (distance from estimate to CI bounds)
        ci_bounds = [results_ci.get('indirect_M1', (np.nan, np.nan)),
                     results_ci.get('indirect_M2', (np.nan, np.nan)),
                     results_ci.get('indirect_M3', (np.nan, np.nan)),
                     results_ci.get('total_indirect', (np.nan, np.nan))]
        
        lower_errors = [est - ci[0] for est, ci in zip(point_estimates, ci_bounds)]
        upper_errors = [ci[1] - est for est, ci in zip(point_estimates, ci_bounds)]
        error_bars = [lower_errors, upper_errors]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 7)) # Increased figure size
        x_pos = np.arange(len(labels))
        
        ax.bar(x_pos, point_estimates, yerr=error_bars, align='center', alpha=0.7, ecolor='black', capsize=10)
        
        ax.set_ylabel('Estimation de l\'Effet', fontsize=12) # French label
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12) # Increased font size
        ax.set_title('Estimations des Effets Indirects et Intervalles de Confiance à 95%', fontsize=14) # French title
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='grey', linewidth=0.8) # Add line at y=0
        
        # Save the plot
        plot_filename = 'plots/mediation_indirect_effects.png' # Changed path to plots/
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300) # Added dpi=300
        print(f"\nPlot saved as: {plot_filename}")

    except Exception as e:
        print(f"\nError generating plot: {e}")
else:
    print("\nSkipping plot generation because results were not calculated.")

print("\n--- Interpretation ---")
print("For each indirect effect:")
print("- Look at the Point Estimate for the direction and magnitude.")
print("- Look at the 95% Bootstrapped Confidence Interval (CI).")
print("- If the CI does *not* contain zero, the indirect effect is statistically significant at the p < .05 level.")
print("- Your hypothesis posits *positive* indirect effects. Check if the point estimate is positive and the CI is entirely above zero.")


# --- 8. Generate Summary Text File ---
summary_file_path = 'results/mediation_summary.txt'

# Check if results are available before attempting to write the file
if 'results_ci' in locals() and 'indirect_m1_est' in locals() and 'corr_matrix' in locals():
    try:
        # Prepare the summary string
        summary_lines = []
        summary_lines.append("--- Analyse de Médiation Multiple : Résumé des Résultats ---")
        summary_lines.append("===========================================================")
        summary_lines.append(f"Variable Indépendante (X): {var_X}")
        summary_lines.append(f"Médiateurs (M1, M2, M3): {var_M1}, {var_M2}, {var_M3}")
        summary_lines.append(f"Variable Dépendante (Y): {var_Y}")
        summary_lines.append("\n--- 1. Matrice de Corrélation (Valeurs R) ---")
        summary_lines.append(corr_matrix.to_string(float_format="{:.3f}".format))

        summary_lines.append("\n--- 2. Estimations des Chemins (Coefficients de Régression) ---")
        # Re-extract p-values for clarity in this section
        key_X = f"Q('{var_X}')"
        key_M1 = f"Q('{var_M1}')"
        key_M2 = f"Q('{var_M2}')"
        key_M3 = f"Q('{var_M3}')"
        p_a1 = fit_m1_orig.pvalues.get(key_X, np.nan)
        p_a2 = fit_m2_orig.pvalues.get(key_X, np.nan)
        p_a3 = fit_m3_orig.pvalues.get(key_X, np.nan)
        p_b1 = fit_y_orig.pvalues.get(key_M1, np.nan)
        p_b2 = fit_y_orig.pvalues.get(key_M2, np.nan)
        p_b3 = fit_y_orig.pvalues.get(key_M3, np.nan)
        p_c_prime = fit_y_orig.pvalues.get(key_X, np.nan)
        p_c_total = fit_total_orig.pvalues.get(key_X, np.nan)

        summary_lines.append(f"  Chemin a1 ({var_X} -> {var_M1}): {a1_est:.4f} (p = {p_a1:.3f})")
        summary_lines.append(f"  Chemin a2 ({var_X} -> {var_M2}): {a2_est:.4f} (p = {p_a2:.3f})")
        summary_lines.append(f"  Chemin a3 ({var_X} -> {var_M3}): {a3_est:.4f} (p = {p_a3:.3f})")
        summary_lines.append(f"  Chemin b1 ({var_M1} -> {var_Y} | X, M2, M3): {b1_est:.4f} (p = {p_b1:.3f})")
        summary_lines.append(f"  Chemin b2 ({var_M2} -> {var_Y} | X, M1, M3): {b2_est:.4f} (p = {p_b2:.3f})")
        summary_lines.append(f"  Chemin b3 ({var_M3} -> {var_Y} | X, M1, M2): {b3_est:.4f} (p = {p_b3:.3f})")
        summary_lines.append(f"  Effet Direct (c') ({var_X} -> {var_Y} | M1, M2, M3): {c_prime_est:.4f} (p = {p_c_prime:.3f})")
        summary_lines.append(f"  Effet Total (c) ({var_X} -> {var_Y}): {c_total_est:.4f} (p = {p_c_total:.3f})")

        summary_lines.append("\n--- 3. Estimations des Effets Indirects (Point Estimates) ---")
        summary_lines.append(f"  Indirect via {var_M1} (a1*b1): {indirect_m1_est:.4f}")
        summary_lines.append(f"  Indirect via {var_M2} (a2*b2): {indirect_m2_est:.4f}")
        summary_lines.append(f"  Indirect via {var_M3} (a3*b3): {indirect_m3_est:.4f}")
        summary_lines.append(f"  Effet Indirect Total (a1*b1 + a2*b2 + a3*b3): {total_indirect_est:.4f}")

        summary_lines.append("\n--- 4. Intervalles de Confiance Bootstrap (95% CI) ---")
        for key, ci in results_ci.items():
            label = key.replace('_', ' ').title()
            if 'Indirect' in label or 'Direct' in label:
                summary_lines.append(f"  {label}: [{ci[0]:.4f}, {ci[1]:.4f}]")

        summary_lines.append("\n--- 5. Interprétation ---")
        summary_lines.append("  Significativité : Un effet (direct ou indirect) est considéré statistiquement significatif")

        summary_lines.append("\n--- 6. Analyse Spécifique des Résultats ---")
        # Analyse de l'Effet Total
        if p_c_total < 0.05:
            summary_lines.append(f"  - L'effet total de '{var_X}' sur '{var_Y}' est statistiquement significatif (c = {c_total_est:.4f}, p = {p_c_total:.3f}).")
        else:
            summary_lines.append(f"  - L'effet total de '{var_X}' sur '{var_Y}' n'est pas statistiquement significatif (c = {c_total_est:.4f}, p = {p_c_total:.3f}).")
        
        # Analyse de l'Effet Direct
        direct_ci = results_ci.get('direct_effect_c_prime', (np.nan, np.nan))
        if 0 < direct_ci[0] or 0 > direct_ci[1]: # Check if CI excludes 0
            summary_lines.append(f"  - L'effet direct de '{var_X}' sur '{var_Y}', en contrôlant pour les médiateurs, est significatif (c' = {c_prime_est:.4f}, CI 95% [{direct_ci[0]:.4f}, {direct_ci[1]:.4f}]).")
        else:
            summary_lines.append(f"  - L'effet direct de '{var_X}' sur '{var_Y}', en contrôlant pour les médiateurs, n'est pas significatif (c' = {c_prime_est:.4f}, CI 95% [{direct_ci[0]:.4f}, {direct_ci[1]:.4f}]).")

        # Analyse des Effets Indirects
        significant_indirect = False
        for m_key, m_var, ind_est in [('indirect_M1', var_M1, indirect_m1_est), ('indirect_M2', var_M2, indirect_m2_est), ('indirect_M3', var_M3, indirect_m3_est)]:
            ind_ci = results_ci.get(m_key, (np.nan, np.nan))
            if 0 < ind_ci[0] or 0 > ind_ci[1]: # Check if CI excludes 0
                summary_lines.append(f"  - L'effet indirect via '{m_var}' est statistiquement significatif (Est. = {ind_est:.4f}, CI 95% [{ind_ci[0]:.4f}, {ind_ci[1]:.4f}]).")
                significant_indirect = True
            else:
                 summary_lines.append(f"  - L'effet indirect via '{m_var}' n'est pas statistiquement significatif (Est. = {ind_est:.4f}, CI 95% [{ind_ci[0]:.4f}, {ind_ci[1]:.4f}]).")
        
        # Analyse de l'Effet Indirect Total
        total_ind_ci = results_ci.get('total_indirect', (np.nan, np.nan))
        if 0 < total_ind_ci[0] or 0 > total_ind_ci[1]: # Check if CI excludes 0
            summary_lines.append(f"  - L'effet indirect total est statistiquement significatif (Est. = {total_indirect_est:.4f}, CI 95% [{total_ind_ci[0]:.4f}, {total_ind_ci[1]:.4f}]).")
            significant_indirect = True # Also mark if total is significant
        else:
            summary_lines.append(f"  - L'effet indirect total n'est pas statistiquement significatif (Est. = {total_indirect_est:.4f}, CI 95% [{total_ind_ci[0]:.4f}, {total_ind_ci[1]:.4f}]).")

        # Conclusion Générale sur la Médiation
        summary_lines.append("\n  Conclusion sur la Médiation:")
        if significant_indirect:
            summary_lines.append(f"  - L'analyse suggère qu'au moins un des médiateurs ({var_M1}, {var_M2}, {var_M3}) joue un rôle significatif dans la relation entre '{var_X}' et '{var_Y}'.")
        else:
            summary_lines.append(f"  - L'analyse n'apporte pas de preuve statistique pour supporter un effet de médiation par '{var_M1}', '{var_M2}', ou '{var_M3}' dans la relation entre '{var_X}' et '{var_Y}' sur la base des intervalles de confiance bootstrap à 95%.")
        summary_lines.append("  (Note: L'interprétation finale doit considérer la théorie sous-jacente et la taille de l'échantillon.)")

        summary_lines.append("  au seuil de p < .05 si son intervalle de confiance à 95% n'inclut PAS zéro.")
        summary_lines.append("  Effet Direct (c') : L'effet de X sur Y après avoir contrôlé pour les médiateurs.")
        summary_lines.append("  Effets Indirects (a*b) : L'effet de X sur Y qui passe *par* chaque médiateur.")
        summary_lines.append("  Effet Indirect Total : La somme des effets indirects spécifiques.")
        summary_lines.append("  Effet Total (c) : L'effet global de X sur Y, sans considérer les médiateurs.")
        summary_lines.append(f"  Vérification : Effet Total (c) ≈ Effet Indirect Total + Effet Direct (c'). ({c_total_est:.4f} ≈ {total_indirect_est:.4f} + {c_prime_est:.4f} = {total_indirect_est + c_prime_est:.4f})")
        summary_lines.append("  (Les petites différences sont dues à la variance d'estimation des modèles)")

        # Write to file
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        
        print(f"\nSummary text file saved as: {summary_file_path}")

    except Exception as e:
        print(f"\nError generating summary text file: {e}")
else:
    print("\nSkipping summary text file generation because results were not calculated.")

