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

## Étapes 3 à 5 — Audit des entrées, sans labels

```powershell
uv run python scripts/drift_lab.py --semaine 1   # puis 2, puis 3
```

Fichiers produits (non versionnés) : `reports/drift_semaine{1,2,3}.csv`,
`reports/figures/drift_s{1,2,3}_<feature>.png`. Comparaison : 20 000 transactions de référence vs
3 000 par semaine, 10 bins.

### Semaine 1 (étape 3)

| Feature | PSI | p-value KS | Verdict |
|---|---|---|---|
| anciennete_client_j | 0,008 | 8,51e-02 | RAS |
| montant_eur | 0,005 | 9,74e-01 | RAS |
| nb_articles | 0,004 | 3,93e-01 | RAS |
| heure | 0,002 | 7,37e-01 | RAS |
| distance_domicile_km | 0,002 | 7,06e-01 | RAS |

PSI max = 0,008. Les p-values KS sont dispersées entre 0,09 et 0,97, et aucune n'est inférieure à 0,05.

- **Q5** (PSI max, verdicts, bulletin) : _à rédiger_
- **Q6** (p-values ≠ 1, tests multiples) : _à rédiger_

### Semaine 2 — Vente Flash de rentrée (étape 4)

| Feature | PSI | p-value KS | Verdict |
|---|---|---|---|
| nb_articles | 0,471 | 1,00e-163 | dérive forte |
| montant_eur | 0,370 | 1,24e-187 | dérive forte |
| anciennete_client_j | 0,311 | 3,92e-96 | dérive forte |
| heure | 0,087 | 1,43e-33 | RAS |
| distance_domicile_km | 0,002 | 1,92e-01 | RAS |

Ce que montrent les figures :
- `nb_articles` : moins de paniers à 1 ou 2 articles, nettement plus à 4, 5, 6 ou 7 articles.
- `montant_eur` : distribution décalée vers la droite, avec une queue plus épaisse au-delà de 60 €.
- `anciennete_client_j` : plus de comptes récents (moins de 250 jours), moins de clients anciens.
- `heure` : creux en milieu de journée, pic de transactions entre 20 h et 23 h.
- `distance_domicile_km` : inchangée.

Pour `heure`, le PSI (0,087 < 0,10) donne RAS, alors que KS donne p ≈ 1e-33 : c'est le cas de Q8.

- **Q7** (classement, seuils 0,25 / 0,10, explication par la Vente Flash) : _à rédiger_
- **Q8** (heure : PSI ou KS ?) : _à rédiger_
- **Q9** (feature en dérive mais peu contributive) : _à rédiger_
- **Q10** (décision sans labels) : _à rédiger_

### Semaine 3 (étape 5)

| Feature | PSI | p-value KS | Verdict |
|---|---|---|---|
| anciennete_client_j | 0,006 | 7,08e-01 | RAS |
| montant_eur | 0,003 | 1,48e-01 | RAS |
| heure | 0,003 | 8,19e-01 | RAS |
| nb_articles | 0,001 | 1,00e+00 | RAS |
| distance_domicile_km | 0,001 | 1,76e-01 | RAS |

PSI max = 0,006 : les entrées sont aussi calmes qu'en semaine 1.

- **Q11** (bulletin sur la seule foi du PSI) : _à rédiger_
- **Q12** (le silence prouve-t-il que le modèle va bien ?) : _à rédiger_

## Étapes suivantes

- [x] Étape 3 — semaine 1
- [x] Étape 4 — semaine 2, Vente Flash
- [x] Étape 5 — semaine 3
- [ ] Étape 6 — labels, `evaluate_semaine.py`, 12 tests verts (Q13 – Q18)
- [ ] Étape 7 — `reports/drift_spec_payguard.md`
