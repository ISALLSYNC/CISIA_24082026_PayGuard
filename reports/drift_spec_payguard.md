# Drift spec - PayGuard v1

Modèle surveillé : `HistGradientBoostingClassifier(random_state=42)` · seuil gelé 0,03
(coût FN = 50 €, FP = 2 €) · carte d'identité : `models/threshold.json`.

## 1. Référence

- Jeu de référence : `data/reference.csv`, soit 20 000 transactions labellisées, celles qui ont
  servi à entraîner le modèle et à choisir le seuil (prévalence de la fraude : 4,48 %).
- Figée en même temps que le modèle (étape 1). Elle ne change pas tant que le modèle ne change pas.
- Condition de remplacement : seulement lors d'un **réentraînement validé** sur une période récente
  **labellisée**. La nouvelle référence est alors le jeu d'entraînement du nouveau modèle.
- On ne prend jamais comme référence une semaine atypique non labellisée. Exemple : la semaine 2
  (Vente Flash) ne doit pas devenir la référence.

## 2. Fenêtre courante & fréquence

- Fenêtre : semaine calendaire de production, sans chevauchement (3 000 transactions environ).
- Fréquence de calcul : une fois par semaine, à la clôture de la semaine.
- Taille minimale d'échantillon : 1 000 transactions. En dessous, le bulletin est reporté et la
  fenêtre est fusionnée avec la suivante, car le PSI sur 10 bins devient trop instable.

## 3. Features surveillées & tests

Tests appliqués à toutes les features :
- **PSI** sur 10 bins : bords calculés sur la référence, extrêmes ouverts à ±∞. C'est **le PSI qui
  décide** de l'alerte.
- **KS à deux échantillons** : à titre d'information seulement. Avec 20 000 contre 3 000
  observations, KS juge significatifs des écarts minimes ; exemple : `heure` en semaine 2, p ≈ 1e-33
  pour un PSI de 0,087.

Seuils d'alerte (lecture du module 31) : PSI ≥ 0,10 → à surveiller · PSI ≥ 0,25 → dérive forte.

| Feature | Test(s) | Seuil d'alerte | Justification |
|---|---|---|---|
| montant_eur | PSI (+ KS informatif) | 0,10 / 0,25 | Variable centrale du risque (coût d'une fraude). Dérive forte observée en S2 (PSI 0,370), distribution décalée vers les gros montants. |
| heure | PSI (+ KS informatif) | 0,10 / 0,25 | Sensible aux campagnes commerciales (pic de 20 h à 23 h en S2). PSI 0,087 en S2 : à suivre sans alerte, KS seul ne déclenche rien. |
| anciennete_client_j | PSI (+ KS informatif) | 0,10 / 0,25 | Les comptes récents sont un signal de risque classique. Dérive forte en S2 (0,311), afflux de nouveaux comptes. |
| nb_articles | PSI (+ KS informatif) | 0,10 / 0,25 | Variable discrète (1 à 7) : le PSI reste lisible, KS est peu adapté aux nombreuses égalités. Plus forte dérive en S2 (0,471). |
| distance_domicile_km | PSI (+ KS informatif) | 0,10 / 0,25 | Signal de fraude (livraison loin du domicile). Stable sur les 3 semaines (PSI ≤ 0,002) : sert de témoin. |

## 4. KPI métier (second rideau, dès que les labels tombent)

- KPI principal : **rappel** au seuil gelé (part des fraudes attrapées).
  - Plancher déclencheur : **0,65**, soit environ 80 % du rappel de validation (0,817).
  - Délai de disponibilité : plusieurs semaines, le temps que les fraudes soient confirmées
    (3 semaines dans ce TP).
- KPI complémentaires :
  - précision au seuil, pour mesurer la charge inutile de l'équipe risque ;
  - taux d'alerte, dans une plage attendue de **20 % à 35 %** (validation 28 %) ;
  - PR-AUC et ROC-AUC, indépendantes du seuil.
- **Signal intermédiaire, sans attendre les labels officiels** : chaque jour, le taux de fraudes
  confirmées dans les dossiers vérifiés par l'équipe risque (précision « terrain »).
  - Normale de référence : environ 12 à 13 % (validation 0,131, S1 0,124).
  - Plancher : 6 %.
  - En S3, cette précision tombe à 1,5 % : l'équipe risque l'aurait vu en quelques jours.

Valeurs observées pendant le TP :

| Semaine | PSI max | Rappel | Précision | Taux d'alerte | ROC-AUC | Diagnostic |
|---|---|---|---|---|---|---|
| 1 | 0,008 | 0,786 | 0,124 | 27,6 % | 0,851 | aucune dérive |
| 2 | 0,471 | 0,923 | 0,183 | 56,6 % | 0,824 | covariate drift |
| 3 | 0,006 | 0,072 | 0,015 | 27,4 % | 0,278 | concept drift |

## 5. Réactions au franchissement

- **PSI ≥ 0,10 sur au moins 1 feature** :
  - mention « à surveiller » dans le bulletin hebdomadaire ;
  - aucune action sur le modèle.
- **PSI ≥ 0,25 sur au moins 2 features** :
  - alerte de dérive des entrées ;
  - chercher la cause métier auprès du marketing (campagne, nouveau canal) ;
  - suivre chaque jour le taux d'alerte et la précision terrain ;
  - prévenir l'équipe risque de la charge à venir (S2 : environ 1 700 dossiers au lieu de 830).
  - **Pas de réentraînement immédiat** : il n'y a pas encore de labels, et un événement ponctuel
    ne doit pas devenir la nouvelle norme.
- **Taux d'alerte hors de la plage 20 % – 35 %** :
  - alerte de capacité à l'équipe risque, même si le PSI est muet.
- **Précision terrain < 6 % sur 3 jours consécutifs** :
  - suspicion de concept drift ;
  - lancer une analyse sur les labels déjà confirmés, sans attendre la fin de la période ;
  - prévenir le responsable PayGuard.
- **Rappel < 0,65 (labels confirmés)** : incident modèle. Actions dans cet ordre :
  1. analyser les fraudes ratées (FN) : quel nouveau mode opératoire ?
  2. réentraîner sur une période récente labellisée ;
  3. rechoisir le seuil par le coût, sur une **nouvelle validation**, pas sur la production ;
  4. remplacer la référence ;
  5. mettre à jour la model card et les tests de non-régression.

  Baisser le seuil ne suffit pas : avec une ROC-AUC sous 0,5, le modèle classe mal, quel que soit
  le seuil.
- **Qui est prévenu, par quel canal, avec quel délai** :
  - l'équipe data reçoit le bulletin hebdomadaire ;
  - l'équipe risque est prévenue dans les 24 h en cas d'alerte forte ou d'alerte de capacité, par le
    canal d'incident de l'équipe ;
  - le responsable PayGuard est prévenu le jour même en cas d'incident modèle.

## 6. Limites connues du dispositif

- **Le concept drift est invisible dans P(X)** : en semaine 3, tous les PSI sont inférieurs à 0,01
  alors que le rappel passe de 0,79 à 0,07. Seuls les labels, ou le signal intermédiaire de
  l'équipe risque, révèlent ce type de panne.
- **Les labels arrivent plusieurs semaines après les transactions** : le KPI métier est tardif, et
  une panne silencieuse peut durer tout ce temps.
- **Le PSI est univarié** : un changement dans la combinaison des features (par exemple gros
  montant et compte récent) peut passer inaperçu si chaque distribution prise seule ne bouge pas.
- **Une dérive n'est pas forcément une panne** : la semaine 2 dérive fortement, mais le rappel y
  reste élevé (0,92). Une alerte PSI demande une interprétation, pas une réaction automatique.
- **Sensibilité au découpage** : le PSI dépend du nombre et des bords des bins ; KS devient
  significatif sur des écarts négligeables quand l'échantillon est grand.
