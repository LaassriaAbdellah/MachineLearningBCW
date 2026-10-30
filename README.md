# Breast Cancer Detection & Diagnosis (WDBC) — Machine Learning Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ID%2017%20(WDBC)-green.svg)](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
[![Institution](https://img.shields.io/badge/Institution-INSEA%20Rabat-005C38.svg)](http://www.insea.ac.ma)
[![Academic Year](https://img.shields.io/badge/Année%20Universitaire-2026--2027-lightgrey.svg)]()

> **Projet Académique · Cours de Machine Learning**  
> **Institut National de Statistique et d'Économie Appliquée (INSEA), Rabat**  
> **Encadré par :** Pr. SAIDI Mohamed Nabil

---

## Sommaire
1. [Contexte Clinique & Problématique](#-contexte-clinique--problématique)
2. [Équipe du Projet](#-équipe-du-projet)
3. [Jeu de Données (WDBC)](#-jeu-de-données-wdbc)
4. [Protocole Méthodologique & Anti-Fuite](#-protocole-méthodologique--anti-fuite)
5. [Résultats Expérimentaux & Benchmark](#-résultats-expérimentaux--benchmark)
6. [Optimisation Clinique du Seuil (Zéro Faux Négatif)](#-optimisation-clinique-du-seuil-zéro-faux-négatif)
7. [Structure du Dépôt](#-structure-du-dépôt)
8. [Installation & Reproduction](#-installation--reproduction)

---

## 🔬 Contexte Clinique & Problématique

Ce projet porte sur la classification supervisée assistée par ordinateur pour le diagnostic du cancer du sein à partir de mesures cytologiques obtenues par cytoponction à l'aiguille fine (**Fine Needle Aspirate - FNA**).

### Question Centrale du Projet
> *« Quel modèle classe le mieux les observations, et lequel limite le risque de ne pas identifier une tumeur maligne ? »*

En oncologie clinique, la matrice des coûts est profondément asymétrique :
* **Faux Négatif (FN) — Coût Critique :** Une tumeur maligne classée bénigne entraîne un retard thérapeutique dramatique pouvant compromettre le pronostic vital de la patiente.
* **Faux Positif (FP) — Coût Modéré :** Une tumeur bénigne classée suspecte conduit à des examens complémentaires (biopsie histologique, IRM) anxiogènes mais protecteurs sur le plan vital.

L'objectif méthodologique est donc de maximiser l'**Accuracy** et le **ROC-AUC** tout en garantissant un **Rappel (Recall) maximal sur la classe Maligne**.

---

## 👥 Équipe du Projet

| Membre | Formation | Rôle & Contributions Spécifiques |
| :--- | :--- | :--- |
| **Yassir BOUITA** | Master M2SI | Modélisation Supervisée, Optimisation des Hyperparamètres & Benchmarking Global |
| **Zakarya KAIROUCH** | Master M2SI | Exploration des Données (EDA), Étanchéité Méthodologique & Pipelines Anti-Leakage |
| **Abdellah LAASSRIA** | Master M2SI | Cadrage Clinique Oncologique, Sommaire & Formalisation de la Question Centrale |
| **Yasser HAIMED** | 3e Année DS-E | Diagnostic Fin des Erreurs, Optimisation du Seuil Clinique ($\tau = 0.35$) & Recommandations |

* **Supervision Académique :** Pr. SAIDI Mohamed Nabil (INSEA)

---

## 📊 Jeu de Données (WDBC)

* **Source :** UCI Machine Learning Repository — [Dataset ID 17](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
* **Créateurs :** Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin)
* **Dimensions :** 569 observations, 30 variables descriptives continues, 1 variable cible binaire.
* **Répartition des classes :**
  * Bénigne ($B$) : 357 cas (62.74 %)
  * Maligne ($M$) : 212 cas (37.26 %)
* **Variables :** 10 caractéristiques morphologiques calculées selon la moyenne (`mean`), l'erreur-type (`se`) et la plus grande valeur (`worst`) :
  * *Rayon (radius), Texture, Périmètre, Aire, Régularité (smoothness), Compacité (compactness), Concavité, Points concaves, Symétrie, Dimension fractale*.
* **Qualité des données :** 0 valeur manquante, séparabilité morphologique élevée sur les variables de volume (`worst perimeter`, `worst area`, `worst concave points`).

---

## 🛡️ Protocole Méthodologique & Anti-Fuite

Afin de garantir une reproductibilité stricte et des résultats statistiquement robustes :
1. **Partitionnement Stratifié Fixé :** Découpage 80 % entraînement ($N_{train} = 455$) et 20 % test ($N_{test} = 114$), avec stratification sur la variable cible et graine aléatoire fixée (`random_state=42`).
2. **Étanchéité Parfaite (Zero Data Leakage) :** Toute étape de transformation (notamment le `StandardScaler`) est strictement encapsulée dans des `Pipeline` Scikit-Learn. Les paramètres de normalisation $(\mu, \sigma)$ sont appris exclusivement sur les folds d'entraînement.
3. **Validation Croisée Interne :** Recherche d'hyperparamètres par `GridSearchCV` / `StratifiedKFold` (5 folds) sur le jeu d'entraînement uniquement.
4. **Évaluation Finale Unique :** Les modèles optimisés ne sont évalués sur le jeu de test vierge qu'une seule et unique fois à la fin du protocole.

---

## 📈 Résultats Expérimentaux & Benchmark

Synthèse des performances obtenues sur le jeu de test indépendant ($N_{test} = 114$ dont 42 cas malins et 72 bénins) :

| Modèle | Accuracy | Précision (M) | Rappel (M) | F1-Score (M) | ROC-AUC | Faux Négatifs (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Régression Logistique ($C=0.1$)** | **97.37 %** | **97.56 %** | **95.24 %** | **0.9639** | **0.9947** | **2** |
| **SVM RBF ($C=1, \gamma=\text{scale}$)** | **97.37 %** | 97.56 % | **95.24 %** | **0.9639** | **0.9947** | **2** |
| **Random Forest ($n=100, d=6$)** | 96.49 % | 97.50 % | 92.86 % | 0.9512 | 0.9924 | 3 |
| **Gradient Boosting ($n=100, lr=0.1$)** | 95.61 % | 95.00 % | 90.48 % | 0.9268 | 0.9911 | 4 |
| **KNN ($k=5$, pondération distance)** | 94.74 % | 94.74 % | 85.71 % | 0.9000 | 0.9858 | 6 |
| **Arbre de Décision ($d=4$)** | 92.98 % | 88.64 % | 92.86 % | 0.9070 | 0.9304 | 3 |

### Principales Conclusions
* **Meilleure performance globale :** La **Régression Logistique** et le **SVM à noyau RBF** dominent le benchmark avec une exactitude de **97.37 %**, un ROC-AUC de **0.9947** et seulement 2 Faux Négatifs sur le seuil standard de $0.50$.
* **Robustesse face à la dimension :** En espace régularisé 30-D, les frontières de décision linéaires/noyautées évitent le surapprentissage constaté sur les arbres de décision purs.

---

## 🎯 Optimisation Clinique du Seuil (Zéro Faux Négatif)

Sur le seuil de coupure standard par défaut ($\tau = 0.50$), la Régression Logistique laisse 2 Faux Négatifs. En médecine oncologique, ces 2 erreurs sont inacceptables.

Une analyse fine de sensibilité de la courbe ROC a permis d'optimiser le seuil de décision :
* En ajustant le seuil à **$\tau^* = 0.35$** :
  * **Rappel Maligne = 100.0 %** (42 / 42 tumeurs malignes détectées).
  * **Faux Négatifs = 0**.
  * **Précision Maligne = 91.30 %** avec seulement 4 Faux Positifs (tumeurs bénignes orientées vers une vérification histologique sécuritaire).

Ce réglage offre le meilleur compromis clinique opérationnel en garantissant la sécurité maximale des patientes.

---

## 📁 Structure du Dépôt

```
MachineLearningBCW/
├── DATA/                                    # Jeu de données officiel UCI WDBC
│   ├── wdbc.data                            # Données brutes (569 lignes x 32 colonnes)
│   └── wdbc.names                           # Documentation officielle UCI
├── figures/                                 # Visualisations et artefacts générés
│   ├── best_hyperparameters.json            # Hyperparamètres optimisés par validation croisée
│   ├── fig1_class_distribution.png          # Répartition des diagnostics (B vs M)
│   ├── fig2_feature_distributions.png       # Distributions comparées par classe
│   ├── fig3_correlation_matrix.png          # Matrice de corrélation de Spearman (30 variables)
│   ├── fig4_pca_projection.png              # Projection PCA 2D (variance expliquée 63.2%)
│   ├── fig5_confusion_matrices.png          # Matrices de confusion des 6 modèles
│   ├── fig6_roc_curves.png                  # Courbes ROC comparées et valeurs d'AUC
│   ├── fig7_model_comparison_bar.png        # Comparaison barplot multi-métrique
│   ├── fig8_threshold_tuning.png            # Compromis Rappel/Précision vs Seuil Clinique
│   └── model_comparison_results.csv         # Tableau complet des métriques de test
├── Report_Latex/                            # Code source LaTeX du rapport académique
│   ├── figures/                             # Figures intégrées dans la compilation
│   ├── logos/                               # Logos officiels HCP et INSEA
│   ├── Rapport_ProjetML.tex                 # Source LaTeX principal (Corps strict de 4 pages)
│   ├── Rapport_Template.tex                 # Gabarit LaTeX institutionnel
│   └── Script_Presentation_ProjetML.tex     # Source LaTeX du script oral
├── src/                                     # Scripts Python modulaires
│   └── train_and_evaluate.py                # Pipeline d'entraînement et d'évaluation complet
├── Mini_Projet_01_Classification_Breast-Cancer.pdf # Énoncé officiel du sujet
├── Presentation_ProjetML.md                 # Guide markdown de soutenance orale
├── Presentation_ProjetML.pptx               # Diaporama officiel PowerPoint (11 slides 16:9)
├── Projet_MachineLearning_WDBC.ipynb        # Notebook Jupyter exécutable et reproductible
├── Rapport_ProjetML.pdf                     # Rapport technique final PDF (8 pages au total)
├── Script_Presentation_ProjetML.pdf         # Script oral intégral PDF (8 à 10 min, 9 pages)
├── requirements.txt                         # Dépendances Python nécessaires
├── .gitignore                               # Règles d'exclusion Git
└── README.md                                # Présentation complète du dépôt
```

---

## 🚀 Installation & Reproduction

### 1. Cloner le dépôt
```bash
git clone https://github.com/YassirB808/MachineLearningBCW.git
cd MachineLearningBCW
```

### 2. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv venv
# Windows :
venv\Scripts\activate
# Linux / macOS :
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Exécuter le Notebook Jupyter
```bash
jupyter notebook Projet_MachineLearning_WDBC.ipynb
```
*Toutes les cellules du notebook sont reproductibles de manière déterministe grâce aux graines aléatoires fixées.*

### 4. Ré-exécuter le script d'entraînement indépendant
```bash
python src/train_and_evaluate.py
```

---

## 📚 Références Bibliographiques
1. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993).** *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology.
2. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction.* Springer Series in Statistics.
3. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
4. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995).** *Breast Cancer Wisconsin (Diagnostic).* UCI Machine Learning Repository. [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

---
*Projet réalisé à l'INSEA — Rabat · Promotion 2026/2027*
