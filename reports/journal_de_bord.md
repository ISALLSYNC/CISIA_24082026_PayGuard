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

**Q1 (m11) — Pourquoi la PR-AUC plutôt que l'accuracy ?**

Avec 4,5 % de fraude, un modèle qui répond toujours « pas fraude » obtient 95,5 % d'accuracy sans
attraper une seule fraude : l'accuracy est dominée par la classe majoritaire et ne dit rien de ce
qui compte. La PR-AUC ne regarde que la classe rare : elle combine la précision (part des alertes
qui sont de vraies fraudes) et le rappel (part des fraudes attrapées). Sa ligne de base est la
**prévalence**, soit 0,045 ici, car un modèle aléatoire a une précision égale au taux de fraude à
tous les niveaux de rappel. En ROC, un modèle aléatoire a autant de vrais positifs que de faux
positifs en proportion (TPR = FPR), d'où 0,5 quelle que soit la prévalence. Notre PR-AUC de 0,260
vaut donc environ 6 fois le hasard, alors que la ROC-AUC de 0,852 paraît plus flatteuse.

**Q2 (m21) — Pourquoi un seuil aussi bas (0,03) ?**

Il est rentable de vérifier une transaction dès que le coût attendu d'une fraude ratée dépasse
celui d'une vérification : p × 50 € > (1 − p) × 2 €, soit p > 2 / 52 ≈ 0,038. Comme une fraude
ratée coûte 25 fois plus cher qu'une vérification inutile, le seuil optimal est très bas. Le seuil a
été choisi sur le **jeu de validation** (25 % de la référence), jamais vu à l'entraînement. Le
rechoisir sur les semaines de production serait malhonnête : on ajusterait le modèle sur les
données qui servent à l'évaluer, ce qui donnerait des performances trop optimistes. En plus, on
masquerait la dérive au lieu de la mesurer, et ces labels n'existent de toute façon pas au moment
de la décision.

**Q3 (m11) — 28 % d'alertes pour 4,5 % de fraude, est-ce défendable ?**

Sur la validation (5 000 transactions), le modèle coûte 41 FN × 50 € + 1 216 FP × 2 € = 4 482 €,
contre 224 fraudes × 50 € = 11 200 € sans modèle : le compromis divise le coût par 2,5 environ.
Il attrape 82 % des fraudes au prix d'une précision faible (13 % des alertes sont de vraies
fraudes). Sur une semaine de 3 000 transactions, cela fait environ 840 dossiers, soit environ 120
par jour. Ce choix est défendable économiquement, à condition que l'équipe risque ait la capacité
de traiter ce volume. Pour trancher, il faudrait connaître :
- la capacité réelle de l'équipe (dossiers traités par analyste et par jour) ;
- le coût réel d'une vérification et d'une fraude ;
- la gêne causée aux clients légitimes dont le paiement est retardé.

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

**Q4 (m31/m21) — Que dit le PSI que KS ne dit pas, et inversement ?**

Le PSI mesure l'**ampleur** du changement de distribution. C'est une taille d'effet, avec des seuils
de lecture métier (0,10 / 0,25), qui ne dépend presque pas de la taille de l'échantillon ; il
indique aussi, bin par bin, où la distribution a bougé. KS répond à une autre question : « l'écart
observé peut-il être dû au hasard ? ». Sa p-value mesure la **significativité** statistique, pas
l'importance de l'écart, et le PSI ne fournit pas cette information. Or la significativité dépend de
la taille des échantillons : avec 20 000 transactions de référence contre 3 000, KS déclare
significatif un écart minime, sans intérêt métier. C'est le cas de `heure` en semaine 2 : p ≈ 1e-33,
alors que le PSI vaut 0,087, sous le seuil de surveillance.

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

**Q5 — PSI max, verdicts et bulletin de la semaine 1**

Le PSI max vaut 0,008 (`anciennete_client_j`), très loin du seuil de 0,10 : les 5 features sont
« RAS ». Les distributions d'entrée de la semaine 1 sont identiques à celles de la référence, aux
fluctuations d'échantillonnage près. Les figures le confirment : les histogrammes se superposent.

> Bulletin semaine 1 — Surveillance des entrées PayGuard : aucune dérive détectée (PSI max = 0,008
> sur 5 features, seuil d'alerte 0,10). Le modèle travaille sur des données conformes à celles de
> son entraînement. Aucune action requise. Rappel : les performances réelles (rappel, précision)
> ne pourront être confirmées qu'à l'arrivée des labels.

**Q6 — Des p-values KS différentes de 1 : est-ce anormal ? Pourquoi ne pas alerter dès p < 0,05 ?**

Non, ce n'est pas anormal. Quand rien n'a changé (sous H₀), la p-value suit une loi uniforme
entre 0 et 1 : elle vaut **0,5 en moyenne**, pas 1. Les valeurs observées, de 0,085 à 0,974, sont
donc parfaitement normales. Par construction, un test au seuil de 5 % se trompe une fois sur
vingt quand il n'y a aucune dérive. Avec 5 features testées chaque semaine, la probabilité d'au
moins une fausse alerte est de 1 − 0,95⁵ ≈ 23 % par semaine, soit environ 13 fausses alertes par
an (5 × 52 × 0,05). Alerter sur p < 0,05 provoquerait donc une fatigue d'alerte : l'équipe finirait
par ignorer les vraies alertes. On décide plutôt sur la taille d'effet (PSI), en corrigeant
éventuellement le seuil de KS pour les tests multiples (Bonferroni : 0,05 / 5 = 0,01).

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

## Étape 6 — Les labels arrivent (`scripts/evaluate_semaine.py`)

### Code ajouté dans `evaluer_semaine()`

```python
y_pred = (proba >= seuil).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
accuracy = (tp + tn) / len(y_true)
precision = tp / (tp + fp) if (tp + fp) else 0.0
rappel = tp / (tp + fn) if (tp + fn) else 0.0
pr_auc = float(average_precision_score(y_true, proba))
roc_auc = float(roc_auc_score(y_true, proba))
```

- **Décision au seuil gelé (0,03)** : une transaction est en alerte si `proba ≥ seuil`. On ne
  rechoisit pas le seuil sur la production : il a été fixé sur la validation (étape 1).
- **Matrice de confusion** : `labels=[0, 1]` garantit une matrice 2×2 même si une classe était
  absente ; `.ravel()` donne l'ordre TN, FP, FN, TP.
- **Métriques recalculées à la main** :
  - accuracy = (TP + TN) / n ;
  - précision = TP / (TP + FP), part des alertes qui sont de vraies fraudes ;
  - rappel = TP / (TP + FN), part des fraudes attrapées.
  - Les divisions par zéro renvoient 0,0.
- **PR-AUC et ROC-AUC** : calculées sur les **probabilités**, pas sur `y_pred`. Elles mesurent la
  qualité du classement des transactions, indépendamment du seuil.

### Résultats (`reports/suivi_semaines.csv`)

```powershell
uv run python scripts/evaluate_semaine.py --semaine 1   # puis 2, puis 3
```

| Semaine | Taux de fraude | Taux d'alerte | TN / FP / FN / TP | Accuracy | Précision | Rappel | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Validation (étape 1) | 4,48 % | 27,98 % | 3560 / 1216 / 41 / 183 | — | 0,131 | 0,817 | 0,260 | 0,852 |
| 1 | 4,37 % | 27,60 % | 2144 / 725 / 28 / 103 | 0,749 | 0,124 | 0,786 | 0,319 | 0,851 |
| 2 | 11,23 % | 56,60 % | 1276 / 1387 / 26 / 311 | 0,529 | 0,183 | 0,923 | 0,394 | 0,824 |
| 3 | 5,57 % | 27,40 % | 2023 / 810 / 155 / 12 | 0,678 | 0,015 | 0,072 | 0,035 | 0,278 |

Faits marquants :
- **Semaine 1** : métriques proches de la validation.
- **Semaine 2** :
  - Le taux de fraude est multiplié par environ 2,5 (11,2 %).
  - Le taux d'alerte double : environ 1 700 dossiers au lieu de 830.
  - Le rappel reste élevé (0,92) et la ROC-AUC quasi stable (0,82).
- **Semaine 3** :
  - Le taux d'alerte reste normal (27,4 %).
  - Le rappel s'effondre, de 0,79 à 0,07 : 12 fraudes attrapées sur 167.
  - La ROC-AUC tombe à 0,278, sous le hasard.
  - L'accuracy (0,678) baisse beaucoup moins que le rappel.

### Validation

```powershell
uv run python -m pytest
```

Suite complète : **12 passed** (definition of done atteinte côté code).

### Questions Q13 – Q18

- **Q13** (métriques par semaine vs verdicts PSI) : _à rédiger_
- **Q14** (accuracy vs rappel en semaine 3) : _à rédiger_
- **Q15** (ROC-AUC < 0,5 en semaine 3) : _à rédiger_
- **Q16** (fiches semaine et type de dérive) : _à rédiger_
- **Q17** (panne la plus dangereuse, signal intermédiaire) : _à rédiger_
- **Q18** (actions à déclencher semaines 2 et 3) : _à rédiger_

## Étapes suivantes

- [x] Étape 3 — semaine 1
- [x] Étape 4 — semaine 2, Vente Flash
- [x] Étape 5 — semaine 3
- [x] Étape 6 — labels, `evaluate_semaine.py`, 12 tests verts
- [x] Étape 7 — [`reports/drift_spec_payguard.md`](drift_spec_payguard.md) : les six éléments
  exigés (référence, fenêtre et fréquence, seuils, KPI métier, réactions, limites connues), avec
  les chiffres des étapes 1 à 6.
- [ ] Réponses Q1 – Q18
