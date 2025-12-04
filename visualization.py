
import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

CUSTOMER_SEGMENTS_PATH = "reports/customer_segments_summary.csv"
CENTROIDS_PATH = "reports/cluster_centroids.csv"
FIGURES_DIR = "reports/figures"

NUMERIC_COLS: List[str] = [
    "frequency",
    "monetary",
    "avg_ticket",
    "max_ticket",
    "std_ticket",
    "n_products",
    "recency_days",
    "tenure_days",
]


def plot_pca_clusters(customer_features: pd.DataFrame, numeric_cols: List[str] = NUMERIC_COLS):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X = customer_features[numeric_cols].values
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        comps[:, 0],
        comps[:, 1],
        c=customer_features["cluster"],
        alpha=0.6,
        cmap="viridis",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer Segmentation (PCA + KMeans)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")

    out_path = f"{FIGURES_DIR}/pca_clusters.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PCA clusters guardado en: {out_path}")


def plot_boxplots_by_cluster(customer_features: pd.DataFrame, numeric_cols: List[str] = NUMERIC_COLS):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=customer_features, x="cluster", y=col)
        plt.title(f"{col} por cluster")
        out_path = f"{FIGURES_DIR}/boxplot_{col}_by_cluster.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Boxplot de {col} guardado en: {out_path}")


def plot_centroids_heatmap(centroids_path: str = CENTROIDS_PATH):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    centroids = pd.read_csv(centroids_path, index_col=0)

    plt.figure(figsize=(10, 4))
    sns.heatmap(centroids, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Centroides de clusters (escala estandarizada)")

    out_path = f"{FIGURES_DIR}/cluster_centroids_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap de centroides guardado en: {out_path}")


def main():
    if not os.path.exists(CUSTOMER_SEGMENTS_PATH):
        raise FileNotFoundError(
            f"No se encontró {CUSTOMER_SEGMENTS_PATH}. Ejecuta primero src/clustering.py."
        )

    customer_features = pd.read_csv(CUSTOMER_SEGMENTS_PATH, index_col=0)
    if "cluster" not in customer_features.columns:
        raise ValueError(
            "El archivo de segmentos no tiene columna 'cluster'. Verifica src/clustering.py."
        )

    plot_pca_clusters(customer_features)
    plot_boxplots_by_cluster(customer_features)
    plot_centroids_heatmap()


if __name__ == "__main__":
    main()
