# Support d'Oral & Guide de Présentation PowerPoint (8 à 10 minutes)
## Mini-Projet 01 : Détection du Cancer du Sein (*Breast Cancer Wisconsin Diagnostic*)
### INSEA — Année 2026/2027

**Encadré par :** Pr. SAIDI Mohamed Nabil  
**Membres de l'équipe :**
* **Yassir BOUITA** *(Master M2SI)* — Modélisation Supervisée & Benchmarking Global (Slides 6 & 7)
* **Zakarya KAIROUCH** *(Master M2SI)* — Exploration des Données & Protocole Anti-Leakage (Slides 4 & 5)
* **Abdellah LAASSRIA** *(Master M2SI)* — Cadrage Clinique, Sommaire & Question Centrale (Slides 1, 2 & 3)
* **Yasser HAIMED** *(3e Année DS-E)* — Diagnostic des Erreurs, Optimisation du Seuil & Conclusion (Slides 8, 9, 10 & 11)

---

## Guide Général de Soutenance
* **Durée totale ciblée :** 8 à 10 minutes (soit ~45 à 55 secondes par diapositive).
* **Diaporama PowerPoint officiel :** [`Presentation_ProjetML.pptx`](file:///C:/Users/bouit/OneDrive/Documents/Projet_MachineLearning/Presentation_ProjetML.pptx) (11 diapositives 16:9, moderne, cartes sans pastilles d'initiales, zéro émoji, numérotation de `Slide 2 / 11` à `Slide 10 / 11`).
* **Rapport académique associé :** [`Rapport_ProjetML.pdf`](file:///C:/Users/bouit/OneDrive/Documents/Projet_MachineLearning/Rapport_ProjetML.pdf) (Corps strictement fixé à 4 pages, conforme aux 3 à 5 pages requises).
* **Script complet mot à mot :** [`Script_Presentation_ProjetML.pdf`](file:///C:/Users/bouit/OneDrive/Documents/Projet_MachineLearning/Script_Presentation_ProjetML.pdf) (9 pages incluant les réponses aux questions du jury).
* **Notebook exécutable :** [`Projet_MachineLearning_WDBC.ipynb`](file:///C:/Users/bouit/OneDrive/Documents/Projet_MachineLearning/Projet_MachineLearning_WDBC.ipynb) (Priorité sur le dossier local `DATA/wdbc.data`).

---

## Découpage des 11 Diapositives & Répartition du Temps

### Diapositive 1 : Page de Garde & Accueil du Jury (00:00 – 00:35)
* **Visuel :** Titre institutionnel INSEA, logos HCP / INSEA, encadrant Pr. SAIDI Mohamed Nabil, cartes auteurs épurées (sans cercles d'initiales) avec filières (Master M2SI et 3A DS-E). Sans numéro de diapositive.
* **Orateur :** Abdellah LAASRIA.

---

### Diapositive 2 : Plan Détaillé de la Présentation / Sommaire (00:35 – 01:05)
* **Visuel :** Grille des 4 modules de la soutenance avec mention des intervenants et minutage (`Slide 2 / 11`).
* **Orateur :** Abdellah LAASRIA.
* **Contenu :**
  1. Module 1 : Cadrage Clinique & Question Centrale (A. Laasria · 00:00 - 02:15)
  2. Module 2 : Exploration & Protocole Expérimental (Z. Kairouch · 02:15 - 04:30)
  3. Module 3 : Modélisation Supervisée & Benchmark (Y. Bouita · 04:30 - 06:45)
  4. Module 4 : Diagnostic, Seuil & Conclusion (Y. Haimed · 06:45 - 09:00)

---

### Diapositive 3 : Contexte Clinique & Question Centrale (01:05 – 02:15)
* **Visuel :** Cytoponction FNA, question centrale posée par l'INSEA, comparaison asymétrique FN vs FP (`Slide 3 / 11`).
* **Orateur :** Abdellah LAASRIA.
* **Message clé :** Le Faux Négatif est vitalement asymétrique ; maximiser le Rappel de la classe maligne pour tendre vers zéro cas manqué.

---

### Diapositive 4 : Base WDBC & Enseignements Clés de l'EDA (02:15 – 03:20)
* **Visuel :** 4 badges statistiques (569 cas, 30 variables continues, 0 valeur manquante, 62.7% B / 37.3% M), camembert et projection ACP 2D (`Slide 4 / 11`).
* **Orateur :** Zakarya KAIROUCH.
* **Message clé :** Forte hypertrophie volumique des tumeurs malignes, colinéarités géométriques massives ($r > 0.98$ entre rayon, périmètre et aire).

---

### Diapositive 5 : Protocole Expérimental Anti-Leakage (03:20 – 04:30)
* **Visuel :** Split stratifié 80/20, étanchéité absolue du StandardScaler (ajusté sur le Train seul), 5-Fold Stratified CV (`Slide 5 / 11`).
* **Orateur :** Zakarya KAIROUCH.
* **Message clé :** Zéro fuite de données (*Data Leakage*), le jeu de test est resté vierge de toute décision.

---

### Diapositive 6 : Méthodologie : Modèles Évalués & Hyperparamètres (04:30 – 05:35)
* **Visuel :** Tableau des espaces de recherche et configurations optimales retenues par GridSearchCV (`Slide 6 / 11`).
* **Orateur :** Yassir BOUITA.
* **Message clé :** La pénalité $L_1$ (Lasso) de la Régression Logistique annule les coefficients redondants liés aux colinéarités, assurant une régularisation optimale.

---

### Diapositive 7 : Résultats Expérimentaux sur le Jeu de Test (05:35 – 06:45)
* **Visuel :** 4 cartes KPI (97.37% Accuracy, 95.24% Recall M, 0.9970 ROC-AUC, 2 FN), grand tableau comparatif et graphique en barres (`Slide 7 / 11`).
* **Orateur :** Yassir BOUITA.
* **Message clé :** La Régression Logistique et le SVM RBF sont co-champions en exactitude (97.37%), mais la Régression Logistique domine le Rappel malin (95.24%) et l'aire ROC-AUC (0.9970).

---

### Diapositive 8 : Diagnostic Visuel : Matrices de Confusion & ROC (06:45 – 07:30)
* **Visuel :** Grille des matrices de confusion et courbes ROC comparées (`Slide 8 / 11`).
* **Orateur :** Yasser HAIMED.
* **Message clé :** Décompte des Faux Négatifs : 2 FN pour la Régression Logistique, 3 pour SVM, 4 pour KNN/GB, 5 pour Random Forest et 8 pour CART (taux d'échec inacceptable de 19%).

---

### Diapositive 9 : Optimisation Clinique : Ajustement du Seuil Décisionnel (07:30 – 08:15)
* **Visuel :** Graphique de variation du seuil $\tau$ et badges de performances à $\tau^* = 0.30$ (`Slide 9 / 11`).
* **Orateur :** Yasser HAIMED.
* **Message clé :** En abaissant le seuil à $\tau^* = 0.30$, le Rappel atteint **100.0% (ZÉRO Faux Négatif, 42/42 cancers détectés)** avec 95.8% de spécificité et 93.3% de précision.

---

### Diapositive 10 : Réponse Formelle à la Question Centrale (08:15 – 08:45)
* **Visuel :** 2 blocs de synthèse (Modèle le plus performant / Modèle le plus protecteur) et explicabilité par coefficients L1 (`Slide 10 / 11`).
* **Orateur :** Yasser HAIMED.
* **Message clé :** La Régression Logistique répond aux deux volets de la question centrale, avec l'avantage décisif de la transparence clinique face aux boîtes noires.

---

### Diapositive 11 : Conclusion Générale & Session Q&A (08:45 – 09:00+)
* **Visuel :** 3 piliers de conclusion, remerciements du jury et ouverture officielle de la session de Questions / Réponses. Sans numéro de diapositive.
* **Orateurs :** Yasser HAIMED puis toute l'équipe pour les échanges techniques.
