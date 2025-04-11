# Analyse du Sondage sur la Perception des Îlots de Chaleur Urbains

Ce projet contient des scripts Python pour importer, traiter, analyser et visualiser les données d'un sondage sur la perception des îlots de chaleur urbains, hébergé sur Supabase.

## Structure du Projet

```
.
├── data/                 # Fichiers de données (.csv)
├── results/              # Fichiers de résultats (graphiques .png, tableaux .xlsx/.csv)
├── docs/                 # Fichiers de documentation (.md, .docx)
├── static/               # Fichiers statiques pour l'interface web (CSS, images)
├── templates/            # Modèles HTML pour l'interface web (Flask)
├── logs/                 # Fichiers journaux (logs)
├── .git/                 # Données du dépôt Git
├── analyse_mediation_multiple.py  # Script d'analyse de médiation et création de graphiques
├── data_import_transform.py       # Script d'importation depuis Supabase, transformation et calcul des scores
├── survey_analysis.py             # Script d'analyse descriptive et création de graphiques
├── analyze_data.py                # Script pour importer et recoder (inverser) certaines colonnes depuis Supabase
├── template_questionnaire.py      # Application Flask pour afficher le questionnaire et envoyer les données à Supabase
├── README.md             # Ce fichier
├── requirements.txt      # Dépendances Python
└── ...                   # Autres fichiers (ex: render.yaml, .gitignore)
```

## Scripts Principaux

1.  **`data_import_transform.py`**:
    *   Récupère les réponses brutes depuis la table Supabase (`responses`).
    *   Applique le codage inversé (reverse coding) aux colonnes spécifiées.
    *   Convertit les colonnes de type Likert en numérique.
    *   Calcule les scores moyens pour chaque thème défini (`score_risque_chaleur`, `score_efficacite`, etc.).
    *   Sauvegarde les données traitées avec les scores dans `results/responses_with_scores.csv`.

2.  **`analyse_mediation_multiple.py`**:
    *   Charge les données traitées depuis `results/scores_participants_par_theme.xlsx` (Note: Assurez-vous que ce fichier est généré ou renommez le fichier source si nécessaire).
    *   Effectue une analyse de médiation multiple en utilisant `statsmodels`.
    *   Calcule les effets directs et indirects avec des intervalles de confiance par bootstrap.
    *   Génère et sauvegarde une matrice de corrélation (`results/correlation_matrix.png`).
    *   Génère et sauvegarde un graphique des effets indirects (`results/mediation_indirect_effects.png`).

3.  **`survey_analysis.py`**:
    *   Charge les données recodées depuis `data/responses_recoded.csv`.
    *   Effectue des analyses descriptives (distributions démographiques, perceptions clés).
    *   Génère et sauvegarde divers graphiques exploratoires dans le dossier `results/`.
    *   Sauvegarde un résumé statistique des échelles de Likert dans `data/likert_summary_stats.csv`.

4.  **`analyze_data.py`**:
    *   Récupère les données brutes depuis Supabase.
    *   Applique le codage inversé aux colonnes spécifiées.
    *   Sauvegarde les données recodées dans `data/responses_recoded.csv`. (Utilisé par `survey_analysis.py`).

5.  **`template_questionnaire.py`**:
    *   Application web Flask qui affiche le questionnaire (`templates/index.html`).
    *   Lors de la soumission, envoie les réponses à la table Supabase configurée.
    *   Redirige vers une page de remerciement (`templates/thank_you.html`).

## Installation

1.  Clonez le dépôt (si applicable).
2.  Assurez-vous d'avoir Python 3 installé.
3.  Créez un environnement virtuel (recommandé) :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    venv\Scripts\activate    # Sur Windows
    ```
4.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1.  **Questionnaire (si nécessaire)**: Lancez l'application Flask pour collecter des données :
    ```bash
    python template_questionnaire.py
    ```
    Accédez à `http://localhost:5000` dans votre navigateur.

2.  **Importation et Traitement**: Exécutez le script pour récupérer les données de Supabase, les transformer et calculer les scores :
    ```bash
    python data_import_transform.py
    ```
    Cela créera/mettra à jour `results/responses_with_scores.csv`.

3.  **Analyse de Médiation**: (Assurez-vous que `results/scores_participants_par_theme.xlsx` existe et contient les scores nécessaires). Exécutez le script d'analyse de médiation :
    ```bash
    python analyse_mediation_multiple.py
    ```
    Cela affichera les résultats de l'analyse et générera les graphiques dans `results/`.

4.  **Analyse Descriptive (Optionnel)**: Exécutez le script pour générer des statistiques descriptives et des graphiques exploratoires :
    *   D'abord, générez les données recodées (si non déjà fait) : `python analyze_data.py`
    *   Ensuite, lancez l'analyse : `python survey_analysis.py`
    Les graphiques seront sauvegardés dans `results/` et le résumé statistique dans `data/`.

## Configuration Supabase

Les informations de connexion à Supabase (URL, Clé API, Nom de table) sont définies au début des scripts `data_import_transform.py`, `analyze_data.py`, et `template_questionnaire.py`. Pour une meilleure sécurité en production, il est fortement recommandé d'utiliser des variables d'environnement ou un fichier de configuration dédié plutôt que de les coder en dur.