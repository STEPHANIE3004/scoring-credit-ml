# Scoring Credit - Pipeline ML Bancaire

Pipeline complet de machine learning pour le scoring de credit (Probability of Default), inspire des modeles Bale II / Bale III utilises dans les banques.

## Fonctionnalites

- Generation de donnees synthetiques realistes (5 000 clients)
- Preprocessing : encodage, imputation, normalisation
- 3 modeles : Logistic Regression, Random Forest, Gradient Boosting
- Metriques bancaires : AUC-ROC, Gini, KS-Statistic
- Validation croisee stratifiee (5-fold)
- Rapport graphique complet (6 visualisations)
- Scoring en temps reel d'un nouveau client
- Sauvegarde des modeles (joblib)

## Resultats obtenus

| Modele | AUC | Gini | KS |
|--------|-----|------|----|
| Logistic Regression | 0.735 | 0.470 | 0.418 |
| Random Forest | 0.687 | 0.375 | 0.351 |
| Gradient Boosting | 0.697 | 0.393 | 0.314 |

## Structure

```
scoring-credit-ml/
├── scoring_credit.py    # Pipeline principal
├── data/
│   └── credit_data.csv  # Dataset synthetique (genere auto)
├── models/
│   └── *.pkl            # Modeles sauvegardes
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python scoring_credit.py
```

Le script :
1. Genere 5 000 clients synthetiques
2. Entraine les 3 modeles
3. Affiche les metriques et le rapport de classification
4. Sauvegarde le meilleur modele
5. Evalue un client exemple
6. Genere `scoring_credit_rapport.png`

## Variables utilisees

| Variable | Description |
|----------|-------------|
| age | Age du client (18-75 ans) |
| revenu_annuel | Revenu brut annuel (EUR) |
| anciennete_emp | Anciennete dans l'emploi (annees) |
| nb_credits_act | Nombre de credits actifs |
| ratio_endett | Taux d'endettement (0-1) |
| historique_cb | Score historique credit (300-850) |
| montant_credit | Montant demande (EUR) |
| duree_credit | Duree en mois |
| type_emploi | CDI / CDD / Independant / Retraite |
| possession_bien | Proprietaire immobilier (0/1) |

## Technologies

- Python 3.x
- scikit-learn (ML pipelines)
- pandas / numpy (data)
- matplotlib (visualisation)
- joblib (serialisation)

## Auteure

Vanelle Stephanie MANGOUA DJOUSSEU
Etudiante en IA & Systemes Embarques - Recherche d'alternance
