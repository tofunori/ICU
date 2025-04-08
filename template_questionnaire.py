import os
import requests # Import requests library
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# --- Supabase Configuration ---
# IMPORTANT: For production, use environment variables!
SUPABASE_URL = "https://wlsggpanveasjrtnpuhs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indsc2dncGFudmVhc2pydG5wdWhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQwNDI4MjQsImV4cCI6MjA1OTYxODgyNH0.qX-J-UnrA3u5IabOI36rJdoUN2cuRATFaF0dL1iN4og"
SUPABASE_TABLE = "responses" # CHANGE THIS if your table name is different

# Construct the API endpoint
supabase_api_url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"

# Set headers for Supabase API
supabase_headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal", # Don't return the inserted data
}
# -- End Supabase Configuration --


@app.route("/", methods=["GET", "POST"])
def questionnaire():
    if request.method == "POST":
        # Get form data as a dictionary
        responses_dict = request.form.to_dict()

        # --- Send data to Supabase ---
        try:
            # Make the POST request to Supabase
            response = requests.post(
                supabase_api_url,
                headers=supabase_headers,
                json=responses_dict # Send data as JSON
            )

            # Check for errors
            response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

            # Redirect to a 'thank you' page on success
            return redirect(url_for('merci'))

        except requests.exceptions.RequestException as e:
            # Handle potential network or Supabase errors
            print(f"Error sending data to Supabase: {e}")
            # You might want to render an error page or message here
            return "Une erreur est survenue lors de l'enregistrement de vos réponses.", 500

    # For GET requests, render the questionnaire form
    return render_template("index.html")

@app.route("/merci")
def merci():
    return render_template("thank_you.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000) # Make accessible on network
