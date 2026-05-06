"""
scoring_credit.py - Pipeline ML complet pour le scoring de credit bancaire
Auteure : Vanelle Stephanie MANGOUA DJOUSSEU

Modeles   : Logistic Regression, Random Forest, Gradient Boosting
Metriques : AUC-ROC, Gini, KS-Statistic, matrice de confusion
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_SPLITS     = 5
MODEL_DIR    = "models"
DATA_DIR     = "data"


# --------------------------------------------------------------------------
# 1. Generation de donnees synthetiques
# --------------------------------------------------------------------------

def generer_dataset(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    age            = rng.randint(18, 75, n)
    revenu_annuel  = rng.lognormal(10.5, 0.6, n).astype(int)
    anciennete_emp = rng.randint(0, 40, n)
    nb_credits_act = rng.randint(0, 8, n)
    ratio_endett   = rng.beta(2, 5, n)
    historique_cb  = rng.randint(300, 850, n)
    montant_credit = rng.lognormal(9, 1.2, n).astype(int)
    duree_credit   = rng.choice([12, 24, 36, 48, 60, 84, 120], n)
    type_emploi    = rng.choice(["CDI", "CDD", "Independant", "Retraite"], n,
                                p=[0.55, 0.20, 0.15, 0.10])
    possession_bien = rng.choice([0, 1], n, p=[0.4, 0.6])

    emploi_score = np.where(type_emploi == "CDI", -0.5,
                   np.where(type_emploi == "CDD",  0.3,
                   np.where(type_emploi == "Independant", 0.6, -0.2)))

    logit = (
        -1.5
        + 0.015 * np.clip(35 - age, -20, 20)
        - 0.000008 * revenu_annuel
        - 0.04  * anciennete_emp
        + 0.20  * nb_credits_act
        + 2.5   * ratio_endett
        - 0.003 * historique_cb
        + 0.0000003 * montant_credit
        + emploi_score
        - 0.3   * possession_bien
        + rng.normal(0, 0.8, n)
    )
    prob_defaut = 1.0 / (1.0 + np.exp(-logit))
    defaut = (rng.uniform(0, 1, n) < prob_defaut).astype(int)

    df = pd.DataFrame({
        "age":             age,
        "revenu_annuel":   revenu_annuel,
        "anciennete_emp":  anciennete_emp,
        "nb_credits_act":  nb_credits_act,
        "ratio_endett":    ratio_endett.round(4),
        "historique_cb":   historique_cb,
        "montant_credit":  montant_credit,
        "duree_credit":    duree_credit,
        "type_emploi":     type_emploi,
        "possession_bien": possession_bien,
        "defaut":          defaut,
    })
    rate = defaut.mean()
    print("[DATA] Dataset genere : {} clients  |  Taux defaut : {:.2%}".format(n, rate))
    return df


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------

def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    df["type_emploi_enc"] = le.fit_transform(df["type_emploi"])

    features = ["age", "revenu_annuel", "anciennete_emp", "nb_credits_act",
                "ratio_endett", "historique_cb", "montant_credit",
                "duree_credit", "type_emploi_enc", "possession_bien"]

    X = df[features]
    y = df["defaut"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("[SPLIT] Train : {}  |  Test : {}".format(len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test, features


# --------------------------------------------------------------------------
# 3. Modeles
# --------------------------------------------------------------------------

def construire_modeles():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000,
                                           random_state=RANDOM_STATE,
                                           class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(n_estimators=200, max_depth=8,
                                               class_weight="balanced",
                                               random_state=RANDOM_STATE,
                                               n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     GradientBoostingClassifier(n_estimators=200,
                                                   learning_rate=0.05,
                                                   max_depth=4,
                                                   random_state=RANDOM_STATE)),
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

def visualiser(resultats, y_test, features, best_nom):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Scoring Credit - Rapport de Performance", fontsize=16, fontweight="bold")
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
    ax2.set_xticklabels([n.replace(" ", "\n") for n in noms], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_title("AUC / Gini / KS")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # (3) Matrice de confusion du meilleur modele
    ax3 = fig.add_subplot(gs[0, 2])
    cm   = confusion_matrix(y_test, resultats[best_nom]["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-defaut", "Defaut"])
    disp.plot(ax=ax3, colorbar=False, cmap="Blues")
    ax3.set_title("Confusion Matrix - {}".format(best_nom))

    # (4) Distribution des scores
    ax4 = fig.add_subplot(gs[1, 0])
    probs  = resultats[best_nom]["y_prob"]
    y_arr  = np.array(y_test)
    ax4.hist(probs[y_arr == 0], bins=40, alpha=0.6, label="Non-defaut", color="#2ca02c")
    ax4.hist(probs[y_arr == 1], bins=40, alpha=0.6, label="Defaut",     color="#d62728")
    ax4.set_xlabel("Probabilite de defaut")
    ax4.set_ylabel("Frequence")
    ax4.set_title("Distribution des scores")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # (5) Feature importance
    ax5 = fig.add_subplot(gs[1, 1])
    best_pipe = resultats[best_nom]["pipe"]
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        idx = np.argsort(importances)
        ax5.barh(np.array(features)[idx], importances[idx], color="#1f77b4")
        ax5.set_title("Importance des variables\n({})".format(best_nom))
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
    path = os.path.join(MODEL_DIR, nom.replace(" ", "_") + ".pkl")
    joblib.dump(pipe, path)
    print("[SAVE] Modele sauvegarde : {}".format(path))


def scorer_nouveau_client(pipe, client_data, features):
    emploi_map = {"CDI": 0, "CDD": 1, "Independant": 2, "Retraite": 3}
    data = dict(client_data)
    data["type_emploi_enc"] = emploi_map.get(data.pop("type_emploi", "CDI"), 0)
    X_client = pd.DataFrame([data])[features]
    prob     = pipe.predict_proba(X_client)[0, 1]
    score    = int((1 - prob) * 1000)
    risque   = "FAIBLE" if prob < 0.15 else ("MOYEN" if prob < 0.40 else "ELEVE")
    decision = "ACCORDE" if prob < 0.30 else "REFUSE"
    return {"prob_defaut": round(prob, 4), "score": score,
            "risque": risque, "decision": decision}


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("   SCORING CREDIT - Pipeline ML Bancaire")
    print("=" * 60)

    df = generer_dataset(n=5000)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "credit_data.csv"), index=False)

    X_train, X_test, y_train, y_test, features = preprocess(df)
    modeles   = construire_modeles()
    resultats = entrainer_evaluer(modeles, X_train, X_test, y_train, y_test)

    best_nom = max(resultats, key=lambda n: resultats[n]["auc"])
    print("\n[BEST] {} - AUC={:.4f}".format(best_nom, resultats[best_nom]["auc"]))

    print("\n[RAPPORT] {}\n".format(best_nom))
    print(classification_report(y_test, resultats[best_nom]["y_pred"],
                                 target_names=["Non-defaut", "Defaut"]))

    sauvegarder_modele(resultats[best_nom]["pipe"], best_nom)

    print("\n[DEMO] Scoring d'un nouveau client :")
    client = {
        "age": 32, "revenu_annuel": 35000, "anciennete_emp": 5,
        "nb_credits_act": 2, "ratio_endett": 0.28, "historique_cb": 680,
        "montant_credit": 15000, "duree_credit": 48,
        "type_emploi": "CDI", "possession_bien": 1,
    }
    r = scorer_nouveau_client(resultats[best_nom]["pipe"], client, features)
    print("  Probabilite de defaut : {:.2%}".format(r["prob_defaut"]))
    print("  Score credit          : {}/1000".format(r["score"]))
    print("  Niveau de risque      : {}".format(r["risque"]))
    print("  Decision              : {}".format(r["decision"]))

    visualiser(resultats, y_test, features, best_nom)


if __name__ == "__main__":
    main()
