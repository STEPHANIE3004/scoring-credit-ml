"""
scoring_credit.py - Pipeline ML complet pour le scoring de credit bancaire
Auteure : Vanelle Stephanie MANGOUA DJOUSSEU

Dataset   : German Credit (UCI / OpenML, 1000 clients, 20 variables)
Modeles   : Logistic Regression, Random Forest, Gradient Boosting, XGBoost + SMOTE
Metriques : AUC-ROC, Gini, KS-Statistic — standards Bale II / Bale III
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_SPLITS     = 5
MODEL_DIR    = "models"
DATA_DIR     = "data"


# --------------------------------------------------------------------------
# 1. Chargement du dataset public (UCI German Credit via OpenML)
# --------------------------------------------------------------------------

def charger_dataset():
    """
    German Credit Dataset (UCI Machine Learning Repository).
    1000 clients, 20 variables (numeriques + categorielles), 30% de mauvais payeurs.
    Source : Statlog German Credit Data — disponible via sklearn.datasets.fetch_openml.
    Reference : Dua, D. & Graff, C. (2019). UCI Machine Learning Repository.
    """
    print("[DATA] Chargement German Credit Dataset (UCI / OpenML)...")
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    X, y = data.data, data.target

    # Cible binaire : 1 = mauvais payeur (risque de defaut), 0 = bon payeur
    y_bin = (y == "bad").astype(int)

    n_total = len(y_bin)
    taux_defaut = y_bin.mean()
    print("[DATA] {} clients  |  Taux defaut : {:.1%}  |  Features : {}".format(
        n_total, taux_defaut, X.shape[1]))

    os.makedirs(DATA_DIR, exist_ok=True)
    df_export = X.copy()
    df_export["defaut"] = y_bin.values
    df_export.to_csv(os.path.join(DATA_DIR, "credit_data.csv"), index=False)
    print("[DATA] Dataset sauvegarde : data/credit_data.csv")
    return X, y_bin


# --------------------------------------------------------------------------
# 2. Preprocessing (pipeline sklearn avec ColumnTransformer)
# --------------------------------------------------------------------------

def construire_preprocessor(X):
    """
    Pipeline de preprocessing sans data leakage :
    - Variables numeriques  : imputation mediane + StandardScaler
    - Variables categorielles : imputation mode + OrdinalEncoder
    Le fit est fait UNIQUEMENT sur le train set.
    """
    cat_cols = X.select_dtypes(include="category").columns.tolist()
    num_cols = X.select_dtypes(exclude="category").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])
    return preprocessor, num_cols, cat_cols


def preprocess(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("[SPLIT] Train : {}  |  Test : {}  |  Defauts test : {:.1%}".format(
        len(X_train), len(X_test), y_test.mean()))
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# 3. Modeles (4 pipelines — dont XGBoost + SMOTE)
# --------------------------------------------------------------------------

def construire_modeles(preprocessor):
    """
    4 modeles :
    - Logistic Regression  : interpretable, conforme aux exigences Bale II (explicabilite)
    - Random Forest        : ensemble, capture les non-linearites
    - Gradient Boosting    : boosting sklearn, robuste
    - XGBoost + SMOTE      : gradient boosting optimise + surechantillonnage SMOTE
                             pour corriger le desequilibre de classe (70/30)
    """
    return {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       C=0.5, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=6,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                               learning_rate=0.05,
                                               random_state=RANDOM_STATE)),
        ]),
        "XGBoost + SMOTE": ImbPipeline([
            ("pre",   preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf",   XGBClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.7,
                                    eval_metric="auc",
                                    random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


# --------------------------------------------------------------------------
# 4. Entrainement & evaluation
# --------------------------------------------------------------------------

def entrainer_evaluer(modeles, X_train, X_test, y_train, y_test):
    resultats = {}
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for nom, pipe in modeles.items():
        print("\n[TRAIN] {}...".format(nom))
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        auc  = roc_auc_score(y_test, y_prob)
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks   = float(np.max(tpr - fpr))
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1).mean()

        resultats[nom] = {
            "pipe":   pipe,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "auc":    auc,
            "gini":   gini,
            "ks":     ks,
            "cv_auc": cv_auc,
            "fpr":    fpr,
            "tpr":    tpr,
        }
        print("  AUC    : {:.4f}".format(auc))
        print("  Gini   : {:.4f}".format(gini))
        print("  KS     : {:.4f}".format(ks))
        print("  CV-AUC : {:.4f}".format(cv_auc))

    return resultats


# --------------------------------------------------------------------------
# 5. Visualisations
# --------------------------------------------------------------------------

def visualiser(resultats, y_test, best_nom):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Scoring Credit (German Credit UCI) — Rapport de Performance",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    couleurs = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # (1) Courbes ROC
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (nom, res) in enumerate(resultats.items()):
        ax1.plot(res["fpr"], res["tpr"],
                 label="{} (AUC={:.3f})".format(nom, res["auc"]),
                 color=couleurs[i])
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("Taux Faux Positifs")
    ax1.set_ylabel("Taux Vrais Positifs")
    ax1.set_title("Courbes ROC")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # (2) Comparaison AUC / Gini / KS
    ax2 = fig.add_subplot(gs[0, 1])
    noms  = list(resultats.keys())
    x     = np.arange(len(noms))
    w     = 0.25
    aucs  = [resultats[n]["auc"]  for n in noms]
    ginis = [resultats[n]["gini"] for n in noms]
    ks_s  = [resultats[n]["ks"]   for n in noms]
    ax2.bar(x - w, aucs,  w, label="AUC",  color="#1f77b4")
    ax2.bar(x,     ginis, w, label="Gini", color="#ff7f0e")
    ax2.bar(x + w, ks_s,  w, label="KS",  color="#2ca02c")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" + ", "\n+ ") for n in noms], fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.axhline(0.70, color="red", linestyle="--", alpha=0.5, label="Seuil Bale II")
    ax2.set_title("AUC / Gini / KS")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # (3) Matrice de confusion du meilleur modele
    ax3 = fig.add_subplot(gs[0, 2])
    cm   = confusion_matrix(y_test, resultats[best_nom]["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bon payeur", "Defaut"])
    disp.plot(ax=ax3, colorbar=False, cmap="Blues")
    ax3.set_title("Confusion Matrix — {}".format(best_nom))

    # (4) Distribution des scores de probabilite
    ax4 = fig.add_subplot(gs[1, 0])
    probs = resultats[best_nom]["y_prob"]
    y_arr = np.array(y_test)
    ax4.hist(probs[y_arr == 0], bins=30, alpha=0.6, label="Bon payeur", color="#2ca02c")
    ax4.hist(probs[y_arr == 1], bins=30, alpha=0.6, label="Defaut",     color="#d62728")
    ax4.set_xlabel("Probabilite de defaut (PD)")
    ax4.set_ylabel("Frequence")
    ax4.set_title("Distribution des scores")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # (5) Feature importance XGBoost
    ax5 = fig.add_subplot(gs[1, 1])
    if "XGBoost + SMOTE" in resultats:
        xgb_pipe = resultats["XGBoost + SMOTE"]["pipe"]
        clf = xgb_pipe.named_steps["clf"]
        importances = clf.feature_importances_
        # noms de features generiques (apres ColumnTransformer)
        feat_names = ["feat_{}".format(i) for i in range(len(importances))]
        idx = np.argsort(importances)[-12:]  # top 12
        ax5.barh(np.array(feat_names)[idx], importances[idx], color="#d62728")
        ax5.set_title("Feature Importance\n(XGBoost + SMOTE)")
        ax5.grid(axis="x", alpha=0.3)

    # (6) Courbe de gains cumules
    ax6 = fig.add_subplot(gs[1, 2])
    y_arr = np.array(y_test)
    for i, (nom, res) in enumerate(resultats.items()):
        df_s = pd.DataFrame({"prob": res["y_prob"], "defaut": y_arr})
        df_s = df_s.sort_values("prob", ascending=False).reset_index(drop=True)
        gains = df_s["defaut"].cumsum() / df_s["defaut"].sum()
        pop   = np.arange(1, len(gains) + 1) / len(gains)
        ax6.plot(pop, gains, label=nom, color=couleurs[i])
    ax6.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatoire")
    ax6.set_xlabel("% Population")
    ax6.set_ylabel("% Defauts captures")
    ax6.set_title("Courbe de Gains Cumules")
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.3)

    out = "scoring_credit_rapport.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("\n[VIZ] Rapport sauvegarde : {}".format(out))


# --------------------------------------------------------------------------
# 6. Sauvegarde & scoring client
# --------------------------------------------------------------------------

def sauvegarder_modele(pipe, nom):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, nom.replace(" ", "_").replace("+", "plus") + ".pkl")
    joblib.dump(pipe, path)
    print("[SAVE] Modele sauvegarde : {}".format(path))


def scorer_nouveau_client(pipe, client_raw):
    """
    Scoring d'un nouveau client (dictionnaire de features brutes).
    Le pipeline sklearn applique le meme preprocessing qu'a l'entrainement.
    """
    import pandas as pd
    X_client = pd.DataFrame([client_raw])
    # Conversion des types categoriels attendus
    for col in X_client.columns:
        if X_client[col].dtype == object:
            X_client[col] = X_client[col].astype("category")

    prob  = pipe.predict_proba(X_client)[0, 1]
    score = int((1 - prob) * 1000)
    risque   = "FAIBLE" if prob < 0.15 else ("MOYEN" if prob < 0.40 else "ELEVE")
    decision = "ACCORDE" if prob < 0.35 else "REFUSE ou CONDITIONS RENFORCEES"
    return {"prob_defaut": round(prob, 4), "score": score,
            "risque": risque, "decision": decision}


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("   SCORING CREDIT — Pipeline ML Bancaire (Bale II / Bale III)")
    print("   Dataset : German Credit UCI (OpenML)")
    print("=" * 65)

    # 1. Donnees
    X, y = charger_dataset()

    # 2. Preprocessor + split
    preprocessor, num_cols, cat_cols = construire_preprocessor(X)
    X_train, X_test, y_train, y_test = preprocess(X, y, preprocessor)

    # 3. Modeles
    modeles   = construire_modeles(preprocessor)
    resultats = entrainer_evaluer(modeles, X_train, X_test, y_train, y_test)

    # 4. Meilleur modele
    best_nom = max(resultats, key=lambda n: resultats[n]["auc"])
    print("\n[BEST] {} — AUC={:.4f}  Gini={:.4f}  KS={:.4f}".format(
        best_nom,
        resultats[best_nom]["auc"],
        resultats[best_nom]["gini"],
        resultats[best_nom]["ks"],
    ))
    print("\n[RAPPORT] Classification Report — {}".format(best_nom))
    print(classification_report(y_test, resultats[best_nom]["y_pred"],
                                target_names=["Bon payeur", "Defaut"]))

    # 5. Sauvegarde
    sauvegarder_modele(resultats[best_nom]["pipe"], best_nom)

    # 6. Demo scoring
    print("\n[DEMO] Scoring d'un nouveau client (profil faible risque) :")
    client_a = {
        "checking_status": ">=200", "duration": 12, "credit_history": "all paid",
        "purpose": "furniture/equipment", "credit_amount": 5000,
        "savings_status": "500<=X<1000", "employment": ">=7",
        "installment_commitment": 2, "personal_status": "male single",
        "other_parties": "none", "residence_since": 4,
        "property_magnitude": "real estate", "age": 38,
        "other_payment_plans": "none", "housing": "own",
        "existing_credits": 1, "job": "skilled",
        "num_dependents": 1, "own_telephone": "yes", "foreign_worker": "yes",
    }
    # Conversion categories
    client_a_cat = {k: pd.Categorical([v]) for k, v in client_a.items()}
    X_demo = pd.DataFrame(client_a_cat)
    for col in cat_cols:
        if col in X_demo.columns:
            X_demo[col] = X_demo[col].astype("category")

    prob  = resultats[best_nom]["pipe"].predict_proba(X_demo)[0, 1]
    score = int((1 - prob) * 1000)
    print("  Probabilite de defaut : {:.2%}".format(prob))
    print("  Score credit          : {}/1000".format(score))
    print("  Risque                : {}".format("FAIBLE" if prob < 0.15 else "MOYEN"))
    print("  Decision              : {}".format("ACCORDE" if prob < 0.35 else "REFUSE"))

    # 7. Visualisation
    visualiser(resultats, y_test, best_nom)

    print("\n[DONE] Fichiers generes :")
    print("  - data/credit_data.csv")
    print("  - models/*.pkl")
    print("  - scoring_credit_rapport.png")


if __name__ == "__main__":
    import pandas as pd
    main()
