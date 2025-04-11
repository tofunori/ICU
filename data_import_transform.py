
import requests
import pandas as pd

# --- Configuration ---
SUPABASE_URL = "https://wlsggpanveasjrtnpuhs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indsc2dncGFudmVhc2pydG5wdWhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQwNDI4MjQsImV4cCI6MjA1OTYxODgyNH0.qX-J-UnrA3u5IabOI36rJdoUN2cuRATFaF0dL1iN4og"
SUPABASE_TABLE = "responses"
OUTPUT_CSV_FILE = "results/responses_with_scores.csv" # Updated path

# Colonnes à recoder à l'envers (Likert inversé)
COLUMNS_TO_REVERSE = [
    'logement_protection',
    'arbres_inutile',
    'difficile_comportements',
    'prefere_interieur',
    'contre_reduction_voiture',
    'pas_intention_s_informer',
    'pas_intention_transport_actif'
]

# Thèmes et leurs questions associées
THEMES = {
    "score_risque_chaleur": [
        'vagues_intenses', 'conseq_sante', 'menace_env',
        'vulnerable', 'logement_protection', 'crainte_futur'
    ],
    "score_efficacite": [
        'parcs_efficace', 'eau_publique', 'arbres_inutile',
        'trouver_endroits_frais', 'difficile_comportements',
        'acces_parcs', 'transport_chaud'
    ],
    "score_attitude_env": [
        'valoriser_espaces_verts', 'soutenir_politiques',
        'prefere_interieur', 'contre_reduction_voiture',
        'espaces_agreables', 'proteger_acces_eau',
        'municipalite_responsable'
    ],
    "score_intention_env": [
        'intention_frequent_parcs', 'intention_soutenir_initiatives',
        'pas_intention_s_informer', 'intention_comportements_eco',
        'pas_intention_transport_actif', 'motivation_si_municipalite'
    ],
    "score_comportements": [
        'freq_parcs', 'activite_verdissement', 'petition_contact',
        'reduire_eau', 'transport_alternatif', 'compost',
        'refroidissement_passif'
    ]
}

# Fonction de reverse coding
def reverse_code(value):
    try:
        val = int(value)
        if 1 <= val <= 5:
            return 6 - val
    except:
        pass
    return value

# --- Main ---
def main():
    print("Fetching data from Supabase...")

    fetch_url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    try:
        response = requests.get(fetch_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            print("No data found in the Supabase table.")
            return

        print(f"Successfully fetched {len(data)} rows.")

        df = pd.DataFrame(data)

        print("Applying reverse coding...")
        for col in COLUMNS_TO_REVERSE:
            if col in df.columns:
                print(f" - Reversing column: {col}")
                df[col] = df[col].apply(reverse_code)


        print("Converting theme columns to numeric...")
        all_theme_cols = set()
        for cols in THEMES.values():
            all_theme_cols.update(cols)

        for col in all_theme_cols:
            if col in df.columns:
                # print(f"   - Converting column: {col}") # Optional: uncomment for detailed logging
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # else: # Optional: uncomment to see if any theme columns are missing from the DataFrame
                # print(f"   - Warning: Theme column '{col}' not found in DataFrame.")

        print("\nDataFrame Info after reverse coding:")
        df.info()
        print("\n")
        print("Calculating theme scores...")
        for theme, cols in THEMES.items():
            valid_cols = [c for c in cols if c in df.columns]
            print(f" --> Processing theme: {theme} with columns: {valid_cols}")
            df[theme] = df[valid_cols].mean(axis=1, skipna=True)
            print(f" - {theme} calculated ({len(valid_cols)} items)")

        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Processed data with theme scores saved to '{OUTPUT_CSV_FILE}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Supabase: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
