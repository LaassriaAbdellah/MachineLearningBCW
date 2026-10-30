"""
Projet Machine Learning - INSEA 2026/2027
Mini-Projet 01 : Détection du Cancer du Sein (Breast Cancer Wisconsin Diagnostic)
Auteurs : Yassir BOUITA, Zakarya KAIROUCH, Abdellah LAASRIA, Yassir
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx.physical_cores_cache = os.cpu_count() or 4
except Exception:
    pass
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

PRIMARY_COLOR = '#00613C'   # INSEA Green
SECONDARY_COLOR = '#D9534F' # Warning Red (Malignant)
BENIGN_COLOR = '#2B7A78'    # Teal / Soft Green (Benign)

def load_data():
    """Charge les données via ucimlrepo ou fichier local wdbc.data."""
    try:
        from ucimlrepo import fetch_ucirepo
        print("[INFO] Chargement via ucimlrepo...")
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features.copy()
        y_df = dataset.data.targets.copy()
        target_col = y_df.columns[0]
        y = y_df[target_col].map({'B': 0, 'M': 1, 0: 0, 1: 1}).astype(int)
    except Exception as e:
        print(f"[WARN] ucimlrepo a échoué ({e}), chargement local...")
        data_path = os.path.join("DATA", "wdbc.data")
        feature_names = [
            'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
            'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
            'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
            'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
            'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
            'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3'
        ]
        cols = ['id', 'diagnosis'] + feature_names
        df = pd.read_csv(data_path, header=None, names=cols)
        X = df[feature_names].copy()
        y = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

    print(f"[INFO] Données chargées avec succès : {X.shape[0]} observations, {X.shape[1]} variables.")
    print(f"[INFO] Répartition cible : Bénin (0) = {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%), "
          f"Malin (1) = {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
    return X, y

def generate_eda_figures(X, y, output_dir="figures"):
    """Génère et sauvegarde les visualisations exploratoires requises."""
    os.makedirs(output_dir, exist_ok=True)
    
    # --- FIGURE 1 : Répartition des Classes ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    counts = y.value_counts().sort_index()
    labels = ['Bénigne (B)', 'Maligne (M)']
    colors = [BENIGN_COLOR, SECONDARY_COLOR]
    
    # Barplot
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    axes[0].set_title("Nombre d'observations par classe", fontweight='bold', pad=12)
    axes[0].set_ylabel("Nombre de patientes")
    axes[0].set_ylim(0, 420)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 8, f"{int(h)} ({h/len(y)*100:.1f}%)",
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Camembert
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, explode=(0.04, 0.04), textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.2}
    )
    for autotext in autotexts:
        autotext.set_color('white')
    axes[1].set_title("Proportion relative (%)", fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_class_distribution.png"), dpi=300)
    plt.close()
    
    # --- FIGURE 2 : Distributions de Caractéristiques Clés ---
    key_features = ['radius1', 'concave_points1', 'area1', 'texture1']
    titles = [
        "Rayon moyen (radius_mean)",
        "Points concaves moyens (concave_points_mean)",
        "Aire moyenne (area_mean)",
        "Texture moyenne (texture_mean)"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for idx, col in enumerate(key_features):
        ax = axes[idx]
        sns.kdeplot(data=X[y == 0][col], ax=ax, label="Bénigne (0)", color=BENIGN_COLOR, fill=True, alpha=0.4, linewidth=2)
        sns.kdeplot(data=X[y == 1][col], ax=ax, label="Maligne (1)", color=SECONDARY_COLOR, fill=True, alpha=0.4, linewidth=2)
        ax.set_title(titles[idx], fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Densité")
        ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_feature_distributions.png"), dpi=300)
    plt.close()

    # --- FIGURE 3 : Matrice de Corrélation ---
    # Top 10 mean features
    mean_cols = [c for c in X.columns if c.endswith('1') or 'mean' in c][:10]
    corr = X[mean_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Matrice de Corrélation (Variables Moyennes - mean)", fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_correlation_matrix.png"), dpi=300)
    plt.close()

    # --- FIGURE 4 : Projection PCA 2D ---
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_all)
    var_exp = pca.explained_variance_ratio_
    
    plt.figure(figsize=(9, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c=BENIGN_COLOR, label='Bénigne (B)', alpha=0.7, edgecolors='none', s=45)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c=SECONDARY_COLOR, label='Maligne (M)', alpha=0.8, edgecolors='k', linewidth=0.5, s=55)
    plt.title(f"Projection ACP (PCA) 2D - Séparabilité des Classes\nVariance expliquée : CP1 = {var_exp[0]*100:.1f}%, CP2 = {var_exp[1]*100:.1f}% (Total = {var_exp.sum()*100:.1f}%)",
              fontweight='bold')
    plt.xlabel(f"Composante Principale 1 ({var_exp[0]*100:.1f}%)")
    plt.ylabel(f"Composante Principale 2 ({var_exp[1]*100:.1f}%)")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_pca_projection.png"), dpi=300)
    plt.close()
    
    print("[INFO] Visualisations EDA générées dans :", output_dir)

def train_and_optimize_models(X_train, y_train):
    """
    Entraîne et optimise les hyperparamètres via Stratified 5-Fold Cross Validation.
    Recherche systématique d'hyperparamètres pour au moins 4 modèles (ici les 5 obligatoires + 1 bonus).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Régression Logistique (C, penalty, solver)
    param_lr = {
        'C': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }
    grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_lr, cv=cv, scoring='f1', n_jobs=None)
    
    # 2. KNN (n_neighbors, weights, metric)
    param_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_knn, cv=cv, scoring='f1', n_jobs=None)
    
    # 3. Arbre de Décision (max_depth, min_samples_split, criterion)
    param_dt = {
        'max_depth': [2, 3, 4, 5, 6, 8, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_dt, cv=cv, scoring='f1', n_jobs=None)
    
    # 4. Random Forest (n_estimators, max_depth, min_samples_split)
    param_rf = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 8, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=cv, scoring='f1', n_jobs=None)
    
    # 5. SVM (kernel, C, gamma)
    param_svm = {
        'C': [0.1, 1.0, 5.0, 10.0, 20.0],
        'gamma': ['scale', 'auto', 0.01, 0.05, 0.1],
        'kernel': ['rbf', 'linear']
    }
    grid_svm = GridSearchCV(SVC(probability=True, random_state=42), param_svm, cv=cv, scoring='f1', n_jobs=None)

    # 6. Gradient Boosting (Bonus)
    param_gb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [2, 3]
    }
    grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_gb, cv=cv, scoring='f1', n_jobs=None)

    models_grid = {
        'Régression Logistique': grid_lr,
        'K-Nearest Neighbors': grid_knn,
        'Arbre de Décision': grid_dt,
        'Forêt Aléatoire': grid_rf,
        'Support Vector Machine (SVM)': grid_svm,
        'Gradient Boosting': grid_gb
    }
    
    best_models = {}
    print("\n--- Début de l'optimisation des hyperparamètres (Stratified 5-Fold CV) ---")
    for name, grid in models_grid.items():
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"[CV OPTIMISÉ] {name:<28} | F1-Score CV: {grid.best_score_:.4f} | Meilleurs paramètres: {grid.best_params_}")
        
    return best_models, models_grid

def evaluate_models(best_models, X_test, y_test, output_dir="figures"):
    """Évalue tous les modèles sur le jeu de test final."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    roc_data = {}
    cm_dict = {}

    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        
        # Probabilités pour ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)
        
        cm_dict[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        
        results.append({
            'Modèle': name,
            'Accuracy': acc,
            'Precision (M)': prec,
            'Recall (M)': rec,
            'F1-Score (M)': f1,
            'Spécificité (B)': spec,
            'ROC-AUC': auc,
            'TP (Vrais M)': tp,
            'FN (M manqués)': fn,
            'FP (Fausses alertes)': fp,
            'TN (Vrais B)': tn
        })

    df_results = pd.DataFrame(results).sort_values(by='Recall (M)', ascending=False)
    
    # --- FIGURE 5 : Matrices de confusion ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Bénin (0)', 'Malin (1)'],
                    yticklabels=['Bénin (0)', 'Malin (1)'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f"{name}\nFN={cm[1,0]} (Malins manqués)", fontweight='bold')
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_confusion_matrices.png"), dpi=300)
    plt.close()
    
    # --- FIGURE 6 : Courbes ROC comparées ---
    plt.figure(figsize=(9, 7))
    palette = [PRIMARY_COLOR, '#3498DB', '#E67E22', '#9B59B6', '#E74C3C', '#1ABC9C']
    for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, color=palette[idx % len(palette)], lw=2.2,
                 label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Aléatoire (AUC = 0.50)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel("Taux de Faux Positifs (1 - Spécificité)")
    plt.ylabel("Taux de Vrais Positifs (Rappel / Sensibilité)")
    plt.title("Courbes ROC Comparatives sur le Jeu de Test (N = 114)", fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_roc_curves.png"), dpi=300)
    plt.close()

    # --- FIGURE 7 : Graphique à barres comparatif ---
    metrics_to_plot = ['Accuracy', 'Recall (M)', 'F1-Score (M)', 'ROC-AUC']
    df_plot = df_results.set_index('Modèle')[metrics_to_plot]
    
    ax = df_plot.plot(kind='bar', figsize=(13, 6), colormap='viridis', width=0.8, edgecolor='black')
    plt.title("Comparaison des Performances Multimétriques sur le Jeu de Test", fontweight='bold', pad=15)
    plt.ylabel("Score")
    plt.ylim(0.85, 1.01)
    plt.xticks(rotation=20, ha='right')
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_model_comparison_bar.png"), dpi=300)
    plt.close()
    
    # --- FIGURE 8 : Analyse d'ajustement de seuil (Threshold Tuning) pour Logistic Regression ---
    best_lr = best_models['Régression Logistique']
    lr_probs = best_lr.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    prec_list, rec_list, f1_list = [], [], []
    for t in thresholds:
        preds = (lr_probs >= t).astype(int)
        prec_list.append(precision_score(y_test, preds, zero_division=0))
        rec_list.append(recall_score(y_test, preds))
        f1_list.append(f1_score(y_test, preds, zero_division=0))
        
    plt.figure(figsize=(9, 5.5))
    plt.plot(thresholds, rec_list, label="Rappel (Recall M)", color=SECONDARY_COLOR, lw=2.5)
    plt.plot(thresholds, prec_list, label="Précision (Precision M)", color=BENIGN_COLOR, lw=2.5)
    plt.plot(thresholds, f1_list, label="F1-Score", color=PRIMARY_COLOR, linestyle='--', lw=2.5)
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Seuil standard (0.50)')
    # Trouver le seuil optimal pour Recall = 1.0 avec précision max
    idx_100_recall = np.where(np.array(rec_list) == 1.0)[0]
    if len(idx_100_recall) > 0:
        opt_t = thresholds[idx_100_recall[-1]]
        plt.axvline(x=opt_t, color='purple', linestyle='-.', label=f'Seuil zéro FN ({opt_t:.2f})')
        plt.scatter([opt_t], [1.0], color='purple', s=80, zorder=5)
    plt.title("Arbitrage Précision / Rappel selon le Seuil de Décision (Régression Logistique)", fontweight='bold')
    plt.xlabel("Seuil de probabilité ($\tau$)")
    plt.ylabel("Score")
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_threshold_tuning.png"), dpi=300)
    plt.close()

    print("[INFO] Évaluations terminées et visualisations sauvegardées.")
    return df_results

def main():
    print("=================================================================")
    print("  MINI-PROJET 01 - MACHINE LEARNING : DÉTECTION DU BREAST CANCER")
    print("  INSEA 2026/2027 - M2SI")
    print("=================================================================")
    
    # 1. Chargement des données
    X, y = load_data()
    
    # Visualisations EDA
    generate_eda_figures(X, y, output_dir="figures")
    
    # 2. Séparation stratifiée 80% Train / 20% Test (random_state=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[INFO] Séparation stratifiée 80/20 :")
    print(f"       Ensemble d'entraînement : {X_train_raw.shape[0]} patientes (B: {(y_train==0).sum()}, M: {(y_train==1).sum()})")
    print(f"       Ensemble de test         : {X_test_raw.shape[0]} patientes (B: {(y_test==0).sum()}, M: {(y_test==1).sum()})")
    
    # 3. Prétraitement : StandardScaler adapté (évitant tout Data Leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    print("[INFO] Mise à l'échelle (StandardScaler) effectuée sans fuite d'information.")

    # 4. Modélisation et Optimisation des Hyperparamètres (Stratified 5-Fold CV)
    best_models, models_grid = train_and_optimize_models(X_train, y_train)

    # 5. Évaluation finale stricte sur le jeu de test
    print("\n--- ÉVALUATION FINALE SUR LE JEU DE TEST (114 observations) ---")
    df_results = evaluate_models(best_models, X_test, y_test, output_dir="figures")
    
    # Formatage console et markdown
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nTABLEAU COMPARATIF DES SCORES TEST :")
    print(df_results.to_string(index=False))

    # Sauvegarder les résultats en CSV et JSON pour intégration facile
    df_results.to_csv("figures/model_comparison_results.csv", index=False)
    
    # Sauvegarde des meilleurs hyperparamètres
    best_params_dict = {name: grid.best_params_ for name, grid in models_grid.items()}
    with open("figures/best_hyperparameters.json", "w") as f:
        json.dump(best_params_dict, f, indent=4)
        
    # Copie des figures vers Report_Latex/figures
    latex_fig_dir = os.path.join("Report_Latex", "figures")
    os.makedirs(latex_fig_dir, exist_ok=True)
    for fname in os.listdir("figures"):
        if fname.endswith(('.png', '.csv', '.json')):
            shutil.copy(os.path.join("figures", fname), os.path.join(latex_fig_dir, fname))
    print(f"\n[INFO] Figures synchronisées avec succès vers '{latex_fig_dir}'.")

if __name__ == "__main__":
    main()
