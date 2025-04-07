import csv
import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Define the path for the CSV file
CSV_FILE = 'responses.csv'

# Define the headers based on the form's 'name' attributes in index.html
CSV_HEADERS = [
    'genre', 'age', 'niveau_etude', 'risque_sante', 'preoccupation_temp',
    'efficacite_arbres', 'efficacite_materiaux', 'priorite_env',
    'impact_individuel', 'intention_energie', 'intention_transport',
    'habitude_tri', 'habitude_eau', 'commentaires'
]

@app.route("/", methods=["GET", "POST"])
def questionnaire():
    if request.method == "POST":
        # Get form data
        responses = request.form

        # Prepare data row for CSV
        data_row = [responses.get(header, '') for header in CSV_HEADERS] # Use .get for safety

        # Check if file exists to write headers
        file_exists = os.path.isfile(CSV_FILE)

        # Append data to CSV
        try:
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists or os.path.getsize(CSV_FILE) == 0:
                    writer.writerow(CSV_HEADERS) # Write header only if file is new or empty
                writer.writerow(data_row)
            # Redirect to a 'thank you' page or show a message
            return redirect(url_for('merci'))
        except IOError as e:
            # Handle potential file writing errors
            print(f"Error writing to CSV: {e}")
            # You might want to render an error page or message here
            return "Une erreur est survenue lors de l'enregistrement de vos réponses.", 500

    # For GET requests, render the questionnaire form
    return render_template("index.html")

@app.route("/merci")
def merci():
    return "<h1>Merci d'avoir répondu au questionnaire !</h1><p><a href='/'>Retour au questionnaire</a></p>"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000) # Make accessible on network
