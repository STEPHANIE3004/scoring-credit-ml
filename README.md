# 📊 Scoring Crédit — Pipeline ML Bancaire (Bâle II / Bâle III)

Pipeline complet de machine learning pour le calcul de **Probability of Default (PD)**, inspiré des modèles de risque de crédit utilisés en banque de détail. Couvre l'intégralité du cycle : feature engineering → entraînement → validation croisée → métriques réglementaires → scoring en production.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20CV%205--fold-orange)
![Métrique](https://img.shields.io/badge/Métrique-AUC%20%7C%20Gini%20%7C%20KS-green)

---

## 🎯 Ce que ce projet démontre

Ce pipeline implémente les **mêmes étapes qu'un modèle de scoring bancaire réel** :

| Étape | Implémentation |
|-------|---------------|
| Feature engineering | 10 variables (ratio endettement, historique crédit, type emploi...) |
| Preprocessing | `sklearn.Pipeline` — imputation, encodage, normalisation en chaîne |
| Modélisation | Logistic Regression, Random Forest, Gradient Boosting |
| Validation | Stratified K-Fold 5 splits — évite le data leakage |
| Métriques Bâle II | **AUC-ROC, Gini, KS-Statistic** — standard industrie |
| Scoring production | Nouvelle instance → PD% + score 0–1000 + décision |
| Persistance | Sauvegarde joblib du meilleur modèle |

> **Pourquoi des données synthétiques ?**  
> Les données de crédit réelles sont couvertes par le secret bancaire et le RGPD. Utiliser un générateur calibré sur des distributions réelles (lognormale pour les revenus, beta pour le taux d'endettement, fonction logistique pour la PD) est la pratique standard dans les équipes de modélisation quand les données de production ne sont pas accessibles hors du SI bancaire. Ce projet démontre le **pipeline**, pas le dataset.

---

## 📈 Résultats obtenus

| Modèle | AUC-ROC | Gini | KS-Statistic |
|--------|---------|------|-------------|
| **Logistic Regression** | **0.797** | **0.594** | **0.498** |
| Random Forest | 0.790 | 0.580 | 0.478 |
| Gradient Boosting | ~0.785 | ~0.570 | ~0.465 |

**Lecture des métriques :**  
- **AUC-ROC** : capacité discriminante globale (1 = parfait, 0.5 = aléatoire). Seuil acceptable Bâle II : > 0.70  
- **Gini = 2×AUC−1** : indicateur standard banques françaises (Gini > 0.50 = bon modèle)  
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

## 🔧 Variables du modèle

| Variable | Description | Sens économique |
|----------|-------------|----------------|
| `age` | Âge du client | Profil de risque selon cycle de vie |
| `revenu_annuel` | Revenu brut annuel (EUR) | Capacité de remboursement |
| `anciennete_emp` | Ancienneté emploi (années) | Stabilité professionnelle |
| `ratio_endett` | Taux d'endettement (0–1) | Variable centrale Bâle II |
| `historique_cb` | Score historique crédit (300–850) | Comportement passé |
| `type_emploi` | CDI / CDD / Indépendant / Retraite | Stabilité des revenus |
| `nb_credits_act` | Nombre de crédits en cours | Exposition totale |
| `montant_credit` | Montant demandé (EUR) | LGD proxy |
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
