# Customer Segmentation with K-Means on Online Retail II

## 1. Descripción general

Este proyecto implementa una **segmentación de clientes** utilizando el dataset real **Online Retail II**, que contiene transacciones de un retailer online del Reino Unido entre 2009 y 2011.

El objetivo es agrupar clientes según su comportamiento de compra (frecuencia, gasto, diversidad de productos, recencia, etc.) mediante **Machine Learning no supervisado (K-Means)**, y visualizar los resultados en un **dashboard interactivo con Streamlit**.

## 2. Objetivo de negocio

La segmentación de clientes permite:

- Identificar perfiles de alto valor.
- Detectar clientes dormidos o inactivos.
- Diseñar campañas de marketing y retención más efectivas.
- Priorizar esfuerzos comerciales según el perfil de cada segmento.

Aunque el dataset es de retail, la metodología es aplicable a contextos como **fintech, corredoras, banca y e-commerce**, donde también se trabaja con comportamiento transaccional.

## 3. Dataset

- Nombre: **Online Retail II**
- Fuente: UCI Machine Learning Repository / Kaggle (versión CSV utilizada en `data/raw/online_retail_II.csv`).
- Contenido:
  - `Invoice`: identificador de la factura
  - `StockCode`: código de producto
  - `Description`: descripción del producto
  - `Quantity`: cantidad vendida
  - `InvoiceDate`: fecha y hora de la transacción
  - `Price`: precio unitario
  - `CustomerID`: identificador del cliente
  - `Country`: país del cliente

Se utiliza principalmente para construir un dataset a nivel **cliente**, que resume su comportamiento de compra.

## 4. Metodología

La metodología sigue estas etapas:

1. **Limpieza de datos (nivel transacción)**
   - Conversión de `InvoiceDate` a formato datetime.
   - Eliminación de filas sin `CustomerID`.
   - Filtrado de transacciones con `Quantity > 0` y `Price > 0`.
   - Creación de `LineTotal = Quantity * Price`.

2. **Feature engineering (nivel cliente)**
   A partir de las transacciones limpias se construyen variables agregadas por `CustomerID`:

   - `frequency`: número de facturas únicas.
   - `monetary`: gasto total del cliente.
   - `avg_ticket`: gasto medio por factura.
   - `max_ticket`: importe máximo en una factura.
   - `std_ticket`: desviación estándar del gasto.
   - `n_products`: número de productos distintos comprados.
   - `recency_days`: días desde la última compra hasta la fecha máxima del dataset.
   - `tenure_days`: tiempo entre primera y última compra.
   - `country`: país predominante del cliente.

3. **Escalamiento de variables**
   - Estandarización de las columnas numéricas utilizando `StandardScaler`
     para que todas tengan media 0 y desviación estándar 1.

4. **Clustering (K-Means)**
   - Selección del número de clusters `k` mediante **Silhouette Score** en un rango de 2 a 10.
   - Entrenamiento del modelo **K-Means** con el valor óptimo de `k`.
   - Asignación de un label de cluster a cada cliente.

5. **Perfilado y visualización**
   - Cálculo de medias de cada variable por cluster.
   - Visualización en 2D mediante **PCA**.
   - Heatmap de los centroides de los clusters (en escala estandarizada).
   - Distribución de tamaño de cada segmento.

6. **Dashboard interactivo (Streamlit)**
   - Selector de `k` en la barra lateral.
   - KPIs generales (número de clientes, gasto medio, frecuencia media).
   - Gráfico de distribución de clientes por cluster.
   - Mapa PCA coloreado por cluster.
   - Heatmap de centroides.
   - Tabla de perfil de clusters.
   - Tabla filtrable de clientes por cluster.

## 5. Estructura del proyecto

```text
customer-segmentation-online-retail-ii/
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.csv          # dataset original
│   └── processed/
│       ├── transactions_clean.csv        # transacciones limpias
│       └── customer_features.csv         # variables a nivel cliente
│
├── notebooks/
│   ├── 01_eda.ipynb                      # análisis exploratorio
│   ├── 02_feature_engineering.ipynb      # limpieza + features por cliente
│   └── 03_clustering_and_profiles.ipynb  # clustering + perfiles
│
├── src/
│   ├── data_prep.py                      # script de limpieza de transacciones
│   ├── features.py                       # script de construcción de features
│   ├── clustering.py                     # script de clustering K-Means
│   └── visualization.py                  # generación de gráficos para reportes
│
├── reports/
│   ├── figures/                          # imágenes generadas (PCA, boxplots, heatmaps)
│   └── customer_segments_summary.csv     # clientes con su cluster asignado
│
├── streamlit_app.py                      # dashboard interactivo
├── requirements.txt
├── environment.yml                       # entorno conda (opcional)
└── README.md
