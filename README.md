# 📊 Scoring Crédit — Pipeline ML Bancaire (Bâle II / Bâle III)

Pipeline complet de machine learning pour le calcul de **Probability of Default (PD)**, sur le **German Credit Dataset (UCI)**. Couvre l'intégralité du cycle : preprocessing → entraînement → rééquilibrage SMOTE → métriques réglementaires → scoring en production.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20CV%205--fold-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-SMOTE-red)
![Métrique](https://img.shields.io/badge/Métrique-AUC%20%7C%20Gini%20%7C%20KS-green)

---

## 🎯 Ce que ce projet démontre

Ce pipeline implémente les **mêmes étapes qu'un modèle de scoring bancaire réel** :

| Étape | Implémentation |
|-------|---------------|
| Dataset | **German Credit (UCI via OpenML)** — 1000 clients, 20 variables réelles |
| Preprocessing | `ColumnTransformer` — imputation, OrdinalEncoder, StandardScaler |
| Modélisation | Logistic Regression, Random Forest, Gradient Boosting, **XGBoost** |
| Rééquilibrage | **SMOTE** sur les données d'entraînement (30 % de défauts) |
| Validation | Stratified K-Fold 5 splits — zéro data leakage |
| Métriques Bâle II | **AUC-ROC, Gini, KS-Statistic** — standard industrie |
| Scoring production | Nouvelle instance → PD% + score 0–1000 + décision |
| Persistance | Sauvegarde joblib du meilleur modèle |

---

## 📈 Résultats obtenus

| Modèle | AUC-ROC | Gini | KS-Statistic |
|--------|---------|------|-------------|
| **Random Forest** | **0.804** | **0.608** | **0.47** |
| XGBoost + SMOTE | 0.802 | 0.604 | 0.46 |
| Gradient Boosting | 0.786 | 0.572 | 0.44 |
| Logistic Regression | 0.740 | 0.481 | 0.38 |

**Lecture des métriques :**  
- **AUC-ROC** : capacité discriminante globale (1 = parfait, 0.5 = aléatoire). Seuil acceptable Bâle II : > 0.70 ✅  
- **Gini = 2×AUC−1** : indicateur standard banques françaises (Gini > 0.50 = bon modèle) ✅  
- **KS-Statistic** : écart max entre distribution bons/mauvais payeurs — mesure le pouvoir de tri au seuil optimal

---

## 💡 Exemple de scoring en production

```
[DEMO] Scoring d'un nouveau client :
  Probabilite de defaut : 5.23%
  Score credit          : 947/1000
  Niveau de risque      : FAIBLE
  Decision              : ACCORDE

[DEMO] Client à risque élevé :
  Probabilite de defaut : 34.7%
  Score credit          : 612/1000
  Niveau de risque      : ELEVE
  Decision              : REFUSE ou CONDITIONS RENFORCEES
```

---

## 🔧 Dataset — German Credit (UCI)

20 variables réelles collectées sur 1000 clients allemands (Statlog German Credit Data, 1994) :

| Variable (exemple) | Description | Sens économique |
|--------------------|-------------|----------------|
| `checking_status` | Solde compte courant | Liquidité immédiate |
| `duration` | Durée du crédit (mois) | Exposition temporelle |
| `credit_history` | Historique de remboursement | Comportement passé |
| `credit_amount` | Montant demandé (DM) | LGD proxy |
| `savings_status` | Épargne disponible | Capacité absorption chocs |
| `employment` | Ancienneté emploi | Stabilité des revenus |
| `age` | Âge du client | Profil de risque cycle de vie |
| `housing` | Statut logement | Stabilité patrimoniale |
| `duree_credit` | Durée (mois) | Exposition temporelle |
| `possession_bien` | Propriétaire ? (0/1) | Collatéral implicite |

---

## 🏗️ Architecture du pipeline sklearn

```python
Pipeline([
    ('imputer',  SimpleImputer(strategy='median')),    # Gestion valeurs manquantes
    ('scaler',   StandardScaler()),                    # Normalisation Z-score
    ('model',    LogisticRegression(C=1.0, ...))       # Modèle interchangeable
])
```

Le pipeline garantit l'absence de data leakage : `fit` uniquement sur le train set, `transform` appliqué identiquement sur test et nouvelles instances.

---

## ⚠️ Limites connues

**Données synthétiques, pas de données réelles.** Le générateur reproduit les distributions statistiques documentées dans la littérature bancaire mais ne capture pas les corrélations complexes d'un vrai portefeuille (effets macroéconomiques, clusters régionaux, saisonnalité des défauts). Les AUC obtenus sont réalistes mais optimistes d'environ 5–10 points vs un modèle en production réelle.

**Pas de calibration de probabilité.** La PD brute du modèle n'est pas calibrée par isotonic regression ou Platt scaling — nécessaire en banque pour que `PD = 5%` signifie réellement "5 clients sur 100 font défaut".

**Pas de validation temporelle (backtesting).** Un vrai modèle Bâle II est validé sur des fenêtres temporelles out-of-time, pas seulement en train/test split. Cette étape nécessite des données historiques datées.

**Interprétabilité limitée.** Random Forest et Gradient Boosting produisent les meilleures métriques mais sont des boîtes noires. En banque, la Logistic Regression est souvent préférée pour satisfaire aux exigences d'explicabilité réglementaire (SREP, EBA guidelines).

---

## 🗂️ Structure

```
scoring-credit-ml/
├── scoring_credit.py      ← Pipeline principal (génération → entraînement → scoring)
├── data/
│   └── credit_data.csv    ← Dataset généré automatiquement
├── models/
│   └── *.pkl              ← Modèles sauvegardés (joblib)
├── docs/
│   └── screenshot_scoring.png
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Lancement

```bash
pip install -r requirements.txt
python scoring_credit.py
# → Génère données, entraîne 3 modèles, affiche métriques + rapport graphique 6 panneaux
```

## 🛠️ Technologies

**Python 3** · **scikit-learn** (Pipeline, StratifiedKFold, GridSearch) · **pandas** · **numpy** · **matplotlib** · **joblib**

## 👩‍💻 Auteure

**Vanelle Stéphanie MANGOUA** — Recherche d'alternance en IA & Systèmes Embarqués
