
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Customer Segmentation - Online Retail II",
    layout="wide"
)

# ----------------------------------------------------
# CARGA DE DATOS
# ----------------------------------------------------
@st.cache_data
def load_customer_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


# ----------------------------------------------------
# CLUSTERING + PCA
# ----------------------------------------------------
def run_clustering(df: pd.DataFrame, numeric_cols, k: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    df_clust = df.copy()
    df_clust["cluster"] = labels

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)

    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(X_scaled)
    df_clust["pc1"] = pc[:, 0]
    df_clust["pc2"] = pc[:, 1]

    return df_clust, centroids


# ----------------------------------------------------
# APLICACIÓN PRINCIPAL
# ----------------------------------------------------
def main():

    st.title("Customer Segmentation Dashboard - Online Retail II")

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    st.sidebar.header("Configuración")

    data_path = st.sidebar.text_input(
        "Ruta de customer_features.csv",
        value="data/processed/customer_features.csv"
    )

    # k por defecto = 2, porque el perfil que analizaste es con 2 clusters
    k = st.sidebar.slider(
        "Número de clusters (k)",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Modelo: K-Means + PCA + Plotly")

    # --------------------------------------------------------
    # CARGA DE ARCHIVO
    # --------------------------------------------------------
    try:
        df = load_customer_features(data_path)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")
        return

    numeric_cols = [
        "frequency",
        "monetary",
        "avg_ticket",
        "max_ticket",
        "std_ticket",
        "n_products",
        "recency_days",
        "tenure_days",
    ]

    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas numéricas en el dataset: {missing}")
        return

    # --------------------------------------------------------
    # EJECUTAR CLUSTERING
    # --------------------------------------------------------
    df_clust, centroids = run_clustering(df, numeric_cols, k)

    # --------------------------------------------------------
    # DEFINIR NOMBRES DE SEGMENTO
    # --------------------------------------------------------
    if k == 2:
        # Nombres basados en el perfil real que analizaste
        specific_names = {
            0: "Clientes regulares diversificados",
            1: "Clientes ultra VIP de alto ticket",
        }
        CLUSTER_NAMES = {
            i: specific_names.get(i, f"Segmento {i}")
            for i in sorted(df_clust["cluster"].unique())
        }
    else:
        # Si cambias k en la app, los nombres de negocio ya no son válidos
        CLUSTER_NAMES = {
            i: f"Segmento {i}" for i in sorted(df_clust["cluster"].unique())
        }

    df_clust["cluster_name"] = df_clust["cluster"].map(CLUSTER_NAMES).astype(str)

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------
    st.subheader("KPIs generales")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N° Clientes", f"{df_clust.shape[0]:,}")
    col2.metric("N° Clusters (k)", k)
    col3.metric("Monetary promedio", f"{df_clust['monetary'].mean():,.2f}")
    col4.metric("Frecuencia promedio", f"{df_clust['frequency'].mean():,.2f}")

    # --------------------------------------------------------
    # DISTRIBUCIÓN DE CLIENTES
    # --------------------------------------------------------
    st.subheader("Distribución de clientes por cluster")

    cluster_counts = (
        df_clust["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_clients")
    )

    total_clients = df_clust.shape[0]
    cluster_counts["pct"] = cluster_counts["n_clients"] / total_clients * 100
    cluster_counts["cluster_name"] = cluster_counts["cluster"].map(CLUSTER_NAMES)

    fig_bar = px.bar(
        cluster_counts,
        x="cluster_name",
        y="n_clients",
        text=cluster_counts["pct"].map(lambda x: f"{x:.1f}%"),
        title="N° de clientes por cluster",
        labels={"cluster_name": "Segmento", "n_clients": "N° de clientes"},
        template="plotly_white",
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------------
    # PCA SCATTER
    # --------------------------------------------------------
    st.subheader("Mapa de clientes (PCA)")

    fig_pca = px.scatter(
        df_clust,
        x="pc1",
        y="pc2",
        color="cluster_name",
        title="Clientes en espacio PCA por segmento",
        hover_data=["frequency", "monetary", "avg_ticket"],
        template="plotly_white",
    )

    fig_pca.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Segmento",
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    # --------------------------------------------------------
    # HEATMAP CENTROIDES
    # --------------------------------------------------------
    st.subheader("Centroides de clusters (escala estandarizada)")

    centroids_plot = centroids.copy()
    centroids_plot.index = [CLUSTER_NAMES.get(i, f"Segmento {i}") for i in centroids_plot.index]

    fig_heat = px.imshow(
        centroids_plot,
        x=numeric_cols,
        y=centroids_plot.index,
        color_continuous_scale="RdBu",
        aspect="auto",
        title="Heatmap de centroides (valores estandarizados)",
        template="plotly_white",
    )
    fig_heat.update_coloraxes(colorbar_title="Valor")
    st.plotly_chart(fig_heat, use_container_width=True)

    # --------------------------------------------------------
    # PERFIL DE CLUSTERS
    # --------------------------------------------------------
    st.subheader("Perfil de clusters (medias por variable)")

    cluster_profile = df_clust.groupby(["cluster", "cluster_name"])[numeric_cols].mean().reset_index()

    melted = cluster_profile.melt(
        id_vars=["cluster", "cluster_name"],
        value_vars=numeric_cols,
        var_name="feature",
        value_name="value",
    )

    fig_profile = px.bar(
        melted,
        x="feature",
        y="value",
        color="cluster_name",
        barmode="group",
        title="Media de cada variable por segmento",
        labels={"feature": "Variable", "value": "Valor medio", "cluster_name": "Segmento"},
        template="plotly_white",
    )

    st.plotly_chart(fig_profile, use_container_width=True)

    st.write("Tabla de perfil (valores medios por segmento):")
    st.dataframe(
        cluster_profile.set_index("cluster_name")[numeric_cols].round(2)
    )

    # --------------------------------------------------------
    # DETALLE DE CLIENTES POR SEGMENTO
    # --------------------------------------------------------
    st.subheader("Detalle de clientes por segmento")

    selected_cluster_name = st.selectbox(
        "Selecciona un segmento para ver detalle",
        options=sorted(df_clust["cluster_name"].unique())
    )

    df_filtered = df_clust[df_clust["cluster_name"] == selected_cluster_name].copy()

    st.write(f"N° de clientes en {selected_cluster_name}: {df_filtered.shape[0]}")

    cols_to_show = [
        "frequency",
        "monetary",
        "avg_ticket",
        "n_products",
        "recency_days",
        "tenure_days",
        "country",
        "cluster",
        "cluster_name",
    ]

    cols_to_show = [c for c in cols_to_show if c in df_filtered.columns]

    st.dataframe(df_filtered[cols_to_show].head(100))


# ----------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------
if __name__ == "__main__":
    main()
