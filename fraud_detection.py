"""
Projekt Detekcja anomali
Marcin Ossoliński
Maksymilian Rak
"""

import os
import sys
import time                                         # do mierzenia, jak długo trwa trening/predykcja
import numpy as np                                  # do operacji matematycznych na dużych tablicach
import pandas as pd                                 # do wczytywania i przetwarzania danych
from tabulate import tabulate                       # do ładnego drukowania tabel w konsoli

from sklearn.model_selection import train_test_split  # do podziału danych na część treningową i testową
from sklearn.preprocessing import StandardScaler      # do przeskalowania danych tak, by miały średnią 0 i odchylenie 1
from sklearn.metrics import (                          # do obliczania metryk oceny modelu
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE             # do sztucznego wyrównania liczby przykładów obu klas

from sklearn.ensemble import RandomForestClassifier   # model Random Forest
from xgboost import XGBClassifier                     # model XGBoost
import lightgbm as lgbm                               # model LightGBM

import matplotlib.pyplot as plt                      # do rysowania wykresów

# ===== KONFIGURACJA =====
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "creditcard.csv")  # ścieżka do pliku CSV z danymi
RESULTS_DIR = "results"
# =======================

def load_and_preprocess(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    # podziel na treningowy (80%) i testowy (20%) zestaw, zachowując proporcje klas
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_tr)

    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te

def oversample_smote(X, y):
    sm = SMOTE(random_state=42)
    Xr, yr = sm.fit_resample(X, y)
    # pokaż, ile teraz jest przykładów każdej klasy
    print(f"  • Po SMOTE: normal={yr.tolist().count(0)}, fraud={yr.tolist().count(1)}")
    return Xr, yr

def ascii_bar(value, width=20):
    filled = int(value * width)
    return "[" + "#"*filled + "-"*(width-filled) + "]"

def benchmark_model(name, model, X_tr, y_tr, X_te, y_te, results):
    print(f"\n=== {name} ===")
    Xr, yr = oversample_smote(X_tr, y_tr)

    # 1) Mierz czas treningu
    t0 = time.time()
    model.fit(Xr, yr)
    train_time = time.time() - t0

    # 2) Mierz czas predykcji
    t1 = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - t1

    # 3) Oblicz macierz pomyłek [TN, FP; FN, TP]
    cm = confusion_matrix(y_te, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print("  Macierz pomyłek:")
    print(f"    TN (negatywy poprawne):        {TN}")
    print(f"    FP (fałszywe alarmy):          {FP}")
    print(f"    FN (przegapione fraudy):       {FN}")
    print(f"    TP (fraudy wykryte poprawnie): {TP}")

    # 4) Oblicz metryki dla klasy "fraud" (1)
    prec = precision_score(y_te, y_pred, pos_label=1)
    rec  = recall_score(y_te, y_pred, pos_label=1)
    f1   = f1_score(y_te, y_pred, pos_label=1)

    # 5) Pokaż czasy i metryki
    print(f"  Train time   : {train_time:.2f}s")
    print(f"  Predict time : {pred_time:.3f}s")
    print(f"  Precision    : {prec:.2f}")
    print(f"  Recall       : {rec:.2f}")
    print(f"  F1-score     : {f1:.2f}")


    results.append({
        "Model": name,
        "Train": train_time,
        "Predict": pred_time,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    })

def print_console_report(results):
    print("\n--- Podsumowanie tabelaryczne ---")

    table = [[
        r["Model"],
        f"{r['Train']:.2f}",
        f"{r['Predict']:.3f}",
        f"{r['Precision']:.2f}",
        f"{r['Recall']:.2f}",
        f"{r['F1']:.2f}"
    ] for r in results]

    print(tabulate(
        table,
        headers=["Model","Train(s)","Pred(s)","Prec","Rec","F1"],
        tablefmt="pretty"
    ))

    print("\n--- ASCII‐bar porównania ---")

    for metric in ("Precision","Recall","F1"):
        print(f"\n{metric}:")
        for r in results:
            val = r[metric]
            print(f"  {r['Model']:<12} {ascii_bar(val)} {val:.2f}")

def plot_results(df):
    plt.style.use("ggplot")
    x = np.arange(len(df))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()

    def bar(ax, y, title):
        bars = ax.bar(x, y, color='skyblue')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        # dodaj wartości nad słupkami
        for rect, val in zip(bars, y):
            offset = 0.02 if title in ("Precision","Recall") else 0.0
            ax.text(
                rect.get_x() + rect.get_width()/2,
                val + offset,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9
            )


    bar(axs[0], df["Train"],   "Train time (s)")
    bar(axs[1], df["Predict"], "Predict time (s)")
    bar(axs[2], df["Precision"], "Precision")
    bar(axs[3], df["Recall"],    "Recall")
    bar(axs[4], df["F1"],       "F1-score")


    ax = axs[5]
    ax.axis('off')
    legend_text = (
        "Precision = TP / (TP + FP)\n"
        "Recall    = TP / (TP + FN)\n"
        "F1-score  = 2 * Precision * Recall / (Precision + Recall)\n\n"
        "TP = poprawnie wykryte fraudy\n"
        "FP = normalne transakcje błędnie oznaczone jako fraud\n"
        "FN = prawdziwe fraudy, które nie zostały wykryte\n"
        "TN = normalne transakcje poprawnie oznaczone jako normalne"
    )
    ax.text(
        0.05, 0.95, legend_text,
        va='top', ha='left', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray")
    )

    fig.tight_layout()
    fig.suptitle("Porównanie modeli", y=1.02, fontsize=16)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()

def main():
    print("\n>>> START PORÓWNANIA MODELI <<<")
    X_tr, X_te, y_tr, y_te = load_and_preprocess(DATA_PATH)


    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":      XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "LightGBM":     lgbm.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1),
    }

    results = []

    for name, model in models.items():
        benchmark_model(name, model, X_tr, y_tr, X_te, y_te, results)

    print_console_report(results)

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    plot_results(df)

    print("\n>>> KONIEC PORÓWNANIA <<<\n")

if __name__ == "__main__":
    main()
