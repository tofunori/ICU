import requests
import pandas as pd
import os

# --- Configuration ---
# IMPORTANT: For better security, consider using environment variables
# or a configuration file for credentials.
SUPABASE_URL = "https://wlsggpanveasjrtnpuhs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indsc2dncGFudmVhc2pydG5wdWhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQwNDI4MjQsImV4cCI6MjA1OTYxODgyNH0.qX-J-UnrA3u5IabOI36rJdoUN2cuRATFaF0dL1iN4og"
SUPABASE_TABLE = "responses"
OUTPUT_CSV_FILE = "responses_recoded.csv"

# List of columns to reverse-code (based on user feedback)
# These correspond to questions 1.5, 2.3, 2.5, 3.3, 3.4, 4.3, 4.5
COLUMNS_TO_REVERSE = [
    'logement_protection',
    'arbres_inutile',
    'difficile_comportements',
    'prefere_interieur',
    'contre_reduction_voiture',
    'pas_intention_s_informer',
    'pas_intention_transport_actif'
]

# --- Helper Function for Reverse Coding ---
def reverse_code(value):
    """Applies reverse coding (6 - value) for a 1-5 scale."""
    try:
        # Attempt to convert to integer and apply formula
        original_score = int(value)
        if 1 <= original_score <= 5:
            return 6 - original_score
        else:
            # Return original if out of expected range (or handle as needed)
            return value
    except (ValueError, TypeError):
        # Return original value if it's not a valid number (e.g., empty string, text)
        return value

# --- Main Script Logic ---
def main():
    print("Fetching data from Supabase...")

    # Construct Supabase API endpoint for fetching all rows
    fetch_url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    try:
        response = requests.get(fetch_url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()

        if not data:
            print("No data found in the Supabase table.")
            return

        print(f"Successfully fetched {len(data)} rows.")

        # Load data into pandas DataFrame
        df = pd.DataFrame(data)

        print("Applying reverse coding...")
        for col in COLUMNS_TO_REVERSE:
            if col in df.columns:
                print(f" - Reversing column: {col}")
                # Apply the reverse_code function to the column
                # Ensure apply works correctly even if column contains non-numeric strings
                df[col] = df[col].apply(reverse_code)
            else:
                print(f" - Warning: Column '{col}' not found in fetched data.")

        # Save the processed DataFrame to a new CSV file
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"\nProcessed data saved to '{OUTPUT_CSV_FILE}'.")
        print("This file now contains the original data with the specified columns reverse-coded.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Supabase: {e}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()