# Journal de bord — TP PayGuard (Sprint 3, module 31)

Apprenant : Ismael Sall · Branche : `ismael-sall`
Référence des consignes : `pas_a_pas_apprenant_payguard_AELION_20260928.pdf`

---

## Étape 0 — Installation (13/10/2026)

```powershell
$env:PYTHONUTF8 = "1"
uv sync --frozen --extra dev
uv run python -m pytest -q
```

- Python 3.13, environnement `.venv/` créé depuis `uv.lock` (sans recalcul du verrou).
- Sortie de départ obtenue : **10 failed · 1 passed · 1 skipped**, conforme à l'attendu.

## Étape 1 — Baseline : entraîner et geler le modèle

```powershell
uv run python scripts/train_model.py
```

Valeurs relevées (validation = 25 % de la référence, 20 000 transactions) :

| Métrique | Valeur | Ligne de base |
|---|---|---|
| Prévalence de la fraude | 4,48 % | — |
| PR-AUC | 0,260 | 0,045 (= prévalence) |
| ROC-AUC | 0,852 | 0,500 |
| Seuil gelé (coût FN = 50 €, FP = 2 €) | 0,03 | — |
| Rappel au seuil | 0,817 | — |
| Précision au seuil | 0,131 | — |
| Taux d'alerte | 27,98 % | — |

Matrice de confusion [TN FP / FN TP] : `[3560 1216 / 41 183]`

Artefacts produits (non versionnés) : `models/model.joblib`, `models/threshold.json`.

Après entraînement, la suite passe à **11 failed · 1 passed** : le test d'intégration semaine 3,
auparavant ignoré faute de modèle, s'exécute désormais et échoue en attendant le code.

### Questions Q1 – Q3

- **Q1** (PR-AUC vs accuracy) : _à rédiger_
- **Q2** (seuil bas et ratio de coûts) : _à rédiger_
- **Q3** (28 % d'alertes pour 4,5 % de fraude) : _à rédiger_

## Étape 2 — Implémentation PSI et KS (`scripts/drift_lab.py`)

### `psi(ref, cur, bins=10)`

```python
edges = np.histogram_bin_edges(ref, bins=bins)
edges[0], edges[-1] = -np.inf, np.inf
p_ref = np.histogram(ref, edges)[0] / len(ref) + 1e-6
p_cur = np.histogram(cur, edges)[0] / len(cur) + 1e-6
return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
```

- **Bins calculés sur la référence** : 10 tranches de même largeur sur la plage de la référence ;
  les données courantes sont rangées dans la même grille pour être comparables.
- **Piège des valeurs hors plage** : `np.histogram` ignore silencieusement toute valeur hors des
  bords. Une dérive vers des montants extrêmes deviendrait invisible. En ouvrant les bords extrêmes
  à −∞ / +∞, ces valeurs tombent dans la première ou la dernière tranche et sont comptées
  (test `test_psi_compte_les_valeurs_hors_plage_de_reference`).
- **`+ 1e-6`** : évite `ln(0)` et la division par zéro quand une tranche est vide.
- **Formule** : chaque terme `(p_cur − p_ref) · ln(p_cur / p_ref)` est ≥ 0 (les deux facteurs ont
  le même signe). Le PSI mesure donc l'**ampleur** de l'écart, sans direction ; il vaut 0 si les
  distributions sont identiques.

### `ks_pvalue(ref, cur)`

```python
return float(stats.ks_2samp(ref, cur).pvalue)
```

- Test de Kolmogorov-Smirnov à deux échantillons : compare l'écart maximal entre les deux fonctions
  de répartition, sous H₀ « même distribution ».
- p ≈ 0 → différence **statistiquement significative**, pas forcément importante : avec de gros
  échantillons (20 000 vs 3 000), KS détecte des écarts minimes.

### `drift_table(df_ref, df_cur, ...)`

- Pour chaque feature : calcul du PSI, de la p-value KS et du verdict (`verdict_psi`, fourni).
- Construction d'un DataFrame aux colonnes `feature / psi / ks_pvalue / verdict`.
- Tri par PSI décroissant puis `reset_index(drop=True)` : les features qui dérivent le plus
  apparaissent en premier.

### Validation

```powershell
uv run python -m pytest -k "psi or ks or verdict or table" -v
```

- Tests unitaires : **8 passed**.
- Suite complète : **11 passed · 1 failed** — l'échec restant
  (`test_integration_semaine3_le_rappel_s_effondre`) dépend de `evaluate_semaine.py` (étape 6).

### Question Q4

- **Q4** (PSI vs KS, effet de la taille d'échantillon) : _à rédiger_

## Étapes suivantes

- [ ] Étape 3 — semaine 1 (Q5, Q6)
- [ ] Étape 4 — semaine 2, Vente Flash (Q7 – Q10)
- [ ] Étape 5 — semaine 3 (Q11, Q12)
- [ ] Étape 6 — labels, `evaluate_semaine.py`, 12 tests verts (Q13 – Q18)
- [ ] Étape 7 — `reports/drift_spec_payguard.md`
