"""PayGuard — Laboratoire de dérive : PSI + KS par feature. 🧑‍🎓 À COMPLÉTER (étape 2).

Trois fonctions à écrire : psi(), ks_pvalue(), drift_table().
Le reste (verdict, graphiques, ligne de commande) est fourni 💻.

Vérifiez votre travail avec :  uv run python -m pytest -k "psi or ks or verdict or table"
Puis lancez l'analyse avec  :  uv run python scripts/drift_lab.py --semaine 1

Rappels du cours (module 31) :
  PSI = Σ (p_cur − p_ref) · ln(p_cur / p_ref)          par tranche (bin)
  PSI < 0,10 : RAS · 0,10–0,25 : à surveiller · > 0,25 : dérive forte
  KS : p-value ≈ 0 → les distributions diffèrent significativement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sans écran : les figures sont écrites sur disque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "data"
RAPPORTS = RACINE / "reports"
FIGURES = RAPPORTS / "figures"

FEATURES = ["montant_eur", "heure", "anciennete_client_j", "nb_articles", "distance_domicile_km"]

# Seuils de lecture usuels du PSI (cf. module 31 §2.2)
SEUIL_PSI_SURVEILLER = 0.10
SEUIL_PSI_FORT = 0.25


# ---------------------------------------------------------------------------
# Étape 2 du TP — les trois fonctions à compléter 🧑‍🎓
# ---------------------------------------------------------------------------


def psi(ref, cur, bins: int = 10) -> float:
    """Population Stability Index entre une distribution de référence et une courante.

    PSI = Σ (p_cur − p_ref) · ln(p_cur / p_ref), calculé par tranche (bin).

    Conventions attendues (identiques au module 31) :
      - les bords des bins se calculent sur la RÉFÉRENCE
        (indice : np.histogram_bin_edges(ref, bins=bins)) ;
      - p_ref et p_cur = effectifs par bin (np.histogram(x, edges)[0]) divisés par
        la taille de l'échantillon, PLUS 1e-6 pour éviter ln(0) ;
      - ⚠️ question à se poser : que fait np.histogram d'une valeur courante qui
        sort de la plage de la référence ? Comment garantir qu'elle soit comptée
        quand même ? (indice : ouvrir les bords extrêmes avec ±np.inf)
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    # Bords calculés sur la référence ; extrêmes ouverts pour compter les valeurs hors plage
    edges = np.histogram_bin_edges(ref, bins=bins)
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    p_cur = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks_pvalue(ref, cur) -> float:
    """p-value du test de Kolmogorov-Smirnov à 2 échantillons.

    p ≈ 0  → les deux distributions diffèrent significativement.
    p élevée → rien ne permet de dire qu'elles diffèrent.
    (indice : scipy.stats.ks_2samp — regardez l'attribut .pvalue du résultat)
    """
    return float(stats.ks_2samp(ref, cur).pvalue)


def drift_table(df_ref: pd.DataFrame, df_cur: pd.DataFrame, features=FEATURES, bins: int = 10) -> pd.DataFrame:
    """Table de dérive : une ligne par feature avec les colonnes
    `feature`, `psi`, `ks_pvalue`, `verdict` (verdict_psi ci-dessous est fourni).

    La table doit être triée par PSI décroissant (les features qui dérivent
    le plus en premier) et réindexée (reset_index(drop=True)).
    """
    lignes = []
    for f in features:
        valeur = psi(df_ref[f], df_cur[f], bins=bins)
        lignes.append({
            "feature": f,
            "psi": valeur,
            "ks_pvalue": ks_pvalue(df_ref[f], df_cur[f]),
            "verdict": verdict_psi(valeur),
        })
    table = pd.DataFrame(lignes, columns=["feature", "psi", "ks_pvalue", "verdict"])
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fournis 💻 — verdict et graphiques
# ---------------------------------------------------------------------------


def verdict_psi(valeur: float) -> str:
    """Lecture usuelle du PSI : < 0,10 RAS · 0,10–0,25 à surveiller · > 0,25 dérive forte."""
    if valeur < SEUIL_PSI_SURVEILLER:
        return "[OK] RAS"
    if valeur < SEUIL_PSI_FORT:
        return "[SURVEILLER] à surveiller"
    return "[ALERTE] dérive forte"


def tracer_feature(ref, cur, feature: str, semaine: int) -> Path:
    """Superpose les histogrammes référence / semaine courante (densités)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ensemble = np.concatenate([ref, cur])
    borne_max = np.quantile(ensemble, 0.99)  # coupe la longue traîne pour la lisibilité
    edges = np.histogram_bin_edges(ensemble, bins=40, range=(ensemble.min(), borne_max))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ref, bins=edges, density=True, alpha=0.55, label="référence", color="#4878cf")
    ax.hist(cur, bins=edges, density=True, alpha=0.55, label=f"semaine {semaine}", color="#d65f5f")
    ax.set_title(f"{feature} — référence vs semaine {semaine} (PSI = {psi(ref, cur):.3f})")
    ax.set_xlabel(feature)
    ax.set_ylabel("densité")
    ax.legend()
    fig.tight_layout()
    chemin = FIGURES / f"drift_s{semaine}_{feature}.png"
    fig.savefig(chemin, dpi=110)
    plt.close(fig)
    return chemin


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse de dérive PayGuard (PSI + KS)")
    parser.add_argument("--semaine", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    df_ref = pd.read_csv(DATA / "reference.csv")
    df_cur = pd.read_csv(DATA / f"courant_semaine{args.semaine}.csv")

    table = drift_table(df_ref, df_cur, bins=args.bins)

    RAPPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RAPPORTS / f"drift_semaine{args.semaine}.csv", index=False)
    for feature in FEATURES:
        tracer_feature(df_ref[feature], df_cur[feature], feature, args.semaine)

    affichage = table.copy()
    affichage["psi"] = affichage["psi"].map(lambda v: f"{v:.3f}")
    affichage["ks_pvalue"] = affichage["ks_pvalue"].map(lambda v: f"{v:.2e}")
    print(f"\n=== Dérive référence vers semaine {args.semaine} "
          f"({len(df_ref)} vs {len(df_cur)} transactions, {args.bins} bins) ===")
    print(affichage.to_string(index=False))
    print(f"\nTable : reports/drift_semaine{args.semaine}.csv · figures : reports/figures/drift_s{args.semaine}_*.png")


if __name__ == "__main__":
    main()
