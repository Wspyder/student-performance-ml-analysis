import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss, roc_auc_score
)

sns.set_style("whitegrid")
OUT_DIR = "ml_outputs/classification"
os.makedirs(OUT_DIR, exist_ok=True)

# Load cleaned synthetic data
CSV = "C:/Users/pc/Downloads/INT234 project/synth_outputs/synthetic_final_cleaned.csv"
df = pd.read_csv(CSV)
print("Loaded:", df.shape)

#1) Create classification target
# 0 = Low (<=60), 1 = Medium (61-80), 2 = High (>80)
df['Perf_Class'] = pd.cut(df['Final_Score_Percentage'],
                          bins=[-1,60,80,100], labels=[0,1,2]).astype(int)

# Optional: view distribution
print("Class distribution:\n", df['Perf_Class'].value_counts(normalize=True))

#2) Preprocess features 
# Map Gender and Preferred_Study_Time if present
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1}).fillna(0).astype(int)
if 'Preferred_Study_Time' in df.columns:
    df['Preferred_Study_Time'] = df['Preferred_Study_Time'].map({
        "Morning (6:00 AM - 12:00 PM)": 0,
        "Afternoon (12:00 PM - 6:00 PM)": 1,
        "Night (6:00 PM - 6:00 AM)": 2
    }).fillna(0).astype(int)

# One-hot Program_Stream if exists
if 'Program_Stream' in df.columns:
    df = pd.get_dummies(df, columns=['Program_Stream'], drop_first=True)

# Prepare X and y
X = df.drop(columns=['Final_Score_Percentage','Perf_Class'])
y = df['Perf_Class']

# Keep only numeric features for classifiers
X = X.select_dtypes(include=[np.number]).copy()

#3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scale features for algorithms that benefit (Logistic, KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

#4) Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB()
}

#confusion matrix
def plot_confusion(cm, labels, title, fname):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

#5) Train, predict, evaluate
rows = []
labels = ["Low","Med","High"]  # mapping for display

for name, model in models.items():
    print(f"\nTraining & evaluating: {name}")
    # Use scaled features for models that need it; DecisionTree/GaussianNB work with unscaled too
    if name in ["LogisticRegression", "KNN"]:
        model.fit(X_train_sc, y_train)
        probs = model.predict_proba(X_test_sc) if hasattr(model, "predict_proba") else None
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        y_pred = model.predict(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # log_loss requires probability estimates; if missing, set to np.nan
    if probs is not None:
        ll = log_loss(y_test, probs, labels=[0,1,2])
        # For ROC AUC (multiclass): compute macro AUC if possible
        try:
            # need one-hot of y_test
            y_test_bin = pd.get_dummies(y_test, columns=[0,1,2])
            auc = roc_auc_score(pd.get_dummies(y_test), probs, average='macro', multi_class='ovo')
        except Exception:
            auc = np.nan
    else:
        ll = np.nan
        auc = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, labels, f"{name} Confusion Matrix", os.path.join(OUT_DIR, f"{name}_confusion.png"))
    def plot_confusion(cm, labels, title, fname):
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()      


    
    rows.append({
        "model": name, "accuracy": acc, "precision_macro": prec, "recall_macro": rec,
        "f1_macro": f1, "log_loss": ll, "roc_auc_macro": auc
    })

    print(f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} LogLoss={ll if not np.isnan(ll) else 'N/A'} ROC_AUC={auc if not np.isnan(auc) else 'N/A'}")

#6) Save metrics summary
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUT_DIR, "classification_metrics.csv"), index=False)
print("\nSaved classification metrics to:", os.path.join(OUT_DIR, "classification_metrics.csv"))
print("Confusion matrix images saved in:", OUT_DIR)

#7) Optional: save models (if you want to load later)
import joblib
for name, model in models.items():
    joblib.dump(model, os.path.join(OUT_DIR, f"{name}.joblib"))

print("\nAll done. Models and outputs are in:", OUT_DIR)
