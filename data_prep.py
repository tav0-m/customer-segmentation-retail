
import os
import pandas as pd

RAW_PATH = "data/raw/online_retail_II.csv"
PROCESSED_PATH = "data/processed/transactions_clean.csv"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Carga el dataset crudo desde CSV y normaliza nombres de columnas.
    Espera columnas similares a las de Online Retail II.
    """
    df = pd.read_csv(path)

    df.columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "CustomerID",
        "Country",
    ]
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza estándar para Online Retail II:
    - Convierte InvoiceDate a datetime.
    - Elimina filas sin fecha válida.
    - Elimina filas sin CustomerID.
    - Convierte CustomerID a entero.
    - Filtra Quantity > 0 y Price > 0.
    - Crea LineTotal = Quantity * Price.
    """
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    df["LineTotal"] = df["Quantity"] * df["Price"]

    return df


def main():
    os.makedirs("data/processed", exist_ok=True)

    df_raw = load_raw_data()
    df_clean = clean_transactions(df_raw)

    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Datos limpios guardados en: {PROCESSED_PATH}")
    print(f"Filas finales: {len(df_clean)}")


if __name__ == "__main__":
    main()
