
import os
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CUSTOMER_FEATURES_PATH = "data/processed/customer_features.csv"
SEGMENTS_SUMMARY_PATH = "reports/customer_segments_summary.csv"
CENTROIDS_PATH = "reports/cluster_centroids.csv"

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


def scale_features(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])
    return X_scaled, scaler


def choose_k_by_silhouette(X_scaled, k_range=range(2, 11)) -> int:
    best_k = None
    best_score = -1.0

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Mejor k según Silhouette: {best_k} (score={best_score:.4f})")
    return best_k


def run_clustering(
    customer_features_path: str = CUSTOMER_FEATURES_PATH,
    k: Optional[int] = None,
    k_range=range(2, 11),
):
    os.makedirs("reports", exist_ok=True)

    customer_features = pd.read_csv(customer_features_path, index_col=0)
    X_scaled, _ = scale_features(customer_features, NUMERIC_COLS)

    if k is None:
        k = choose_k_by_silhouette(X_scaled, k_range=k_range)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)

    customer_features["cluster"] = labels

    cluster_profile = customer_features.groupby("cluster")[NUMERIC_COLS].mean()
    centroids = pd.DataFrame(model.cluster_centers_, columns=NUMERIC_COLS)

    customer_features.to_csv(SEGMENTS_SUMMARY_PATH)
    centroids.to_csv(CENTROIDS_PATH, index_label="cluster")

    print(f"Segmentos guardados en: {SEGMENTS_SUMMARY_PATH}")
    print(f"Centroides guardados en: {CENTROIDS_PATH}")
    print("\nPerfil de clusters (media por variable):")
    print(cluster_profile)


def main():
    run_clustering()


if __name__ == "__main__":
    main()
