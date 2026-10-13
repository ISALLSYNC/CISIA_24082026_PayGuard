"""PayGuard — Évaluation métier d'une semaine (labels arrivés). 🧑‍🎓 À COMPLÉTER (étape 6).

⚠️ N'ouvrez ce fichier qu'à l'étape 6 du TP : il utilise les labels de data/labels/,
   qui « n'existent pas encore » pendant les étapes 3 à 5 (en production, les fraudes
   ne sont confirmées que plusieurs semaines après la transaction).

Le chargement (modèle gelé, seuil, jointure features/labels) est fourni 💻.
À compléter 🧑‍🎓 : la décision au seuil, la matrice de confusion et les métriques —
exactement ce que vous avez pratiqué aux modules 11 et 21 du Sprint 2.

Vérifiez votre travail avec :  uv run python -m pytest
Puis lancez l'évaluation     :  uv run python scripts/evaluate_semaine.py --semaine 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "data"
MODELES = RACINE / "models"
RAPPORTS = RACINE / "reports"


def evaluer_semaine(semaine: int) -> dict:
    """Évalue le modèle GELÉ (seuil compris) sur une semaine dont les labels sont arrivés.

    Retourne un dictionnaire : semaine, n, taux_fraude, taux_alerte, tn/fp/fn/tp,
    accuracy, precision, rappel, pr_auc, roc_auc.
    """
    modele = joblib.load(MODELES / "model.joblib")
    carte = json.loads((MODELES / "threshold.json").read_text(encoding="utf-8"))
    seuil, features = carte["seuil"], carte["features"]

    df = pd.read_csv(DATA / f"courant_semaine{semaine}.csv").merge(
        pd.read_csv(DATA / "labels" / f"labels_semaine{semaine}.csv"),
        on="transaction_id",
        validate="one_to_one",
    )

    proba = modele.predict_proba(df[features])[:, 1]
    y_true = df["fraude"].to_numpy()

    # --- cœur de l'étape 6 — TODO 🧑‍🎓 --------------------------------------
    # 1. y_pred : décision binaire au seuil GELÉ (proba >= seuil → 1, sinon 0).
    # 2. tn, fp, fn, tp : matrice de confusion (indice : confusion_matrix(...).ravel()).
    # 3. accuracy, precision, rappel : recalculez-les À LA MAIN depuis tn/fp/fn/tp
    #    (révision module 11 — attention aux divisions par zéro : renvoyer 0.0).
    # 4. pr_auc, roc_auc : depuis les PROBABILITÉS, pas depuis y_pred
    #    (révision module 21 — l'AUC est indépendante du seuil).
    y_pred = (proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    rappel = tp / (tp + fn) if (tp + fn) else 0.0
    pr_auc = float(average_precision_score(y_true, proba))
    roc_auc = float(roc_auc_score(y_true, proba))
    # -----------------------------------------------------------------------

    return {
        "semaine": semaine,
        "n": int(len(y_true)),
        "taux_fraude": float(y_true.mean()),
        "taux_alerte": float(y_pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "rappel": float(rappel),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "seuil": seuil,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation métier PayGuard (labels arrivés)")
    parser.add_argument("--semaine", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    m = evaluer_semaine(args.semaine)

    print(f"\n=== Semaine {m['semaine']} — {m['n']} transactions · seuil gelé = {m['seuil']} ===")
    print(f"  Taux de fraude réel   : {m['taux_fraude']:.2%}")
    print(f"  Taux d'alerte modèle  : {m['taux_alerte']:.2%}  "
          f"(environ {int(m['taux_alerte'] * m['n'])} dossiers pour l'équipe risque)")
    print(f"  Matrice de confusion  : [TN={m['tn']}  FP={m['fp']} / FN={m['fn']}  TP={m['tp']}]")
    print(f"  Accuracy              : {m['accuracy']:.3f}")
    print(f"  Précision             : {m['precision']:.3f}")
    print(f"  Rappel (fraude)       : {m['rappel']:.3f}")
    print(f"  PR-AUC                : {m['pr_auc']:.3f}   ·   ROC-AUC : {m['roc_auc']:.3f}")

    # Journal de suivi : une ligne par semaine (réécrite si relancée)
    RAPPORTS.mkdir(parents=True, exist_ok=True)
    chemin = RAPPORTS / "suivi_semaines.csv"
    ligne = pd.DataFrame([m])
    if chemin.exists():
        suivi = pd.read_csv(chemin)
        suivi = suivi[suivi["semaine"] != m["semaine"]]
        suivi = pd.concat([suivi, ligne], ignore_index=True).sort_values("semaine")
    else:
        suivi = ligne
    suivi.to_csv(chemin, index=False)
    print(f"\nJournal mis à jour : reports/suivi_semaines.csv")


if __name__ == "__main__":
    main()
