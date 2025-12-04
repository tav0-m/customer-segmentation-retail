
import os
import pandas as pd

TRANSACTIONS_PATH = "data/processed/transactions_clean.csv"
CUSTOMER_FEATURES_PATH = "data/processed/customer_features.csv"


def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Construye variables de comportamiento a nivel cliente:
    - frequency, monetary, avg_ticket, max_ticket, std_ticket
    - n_products, recency_days, tenure_days
    - country (modo)
    """
    grp = transactions.groupby("CustomerID")
    reference_date = transactions["InvoiceDate"].max()

    customer = pd.DataFrame()

    customer["frequency"] = grp["Invoice"].nunique()
    customer["monetary"] = grp["LineTotal"].sum()
    customer["avg_ticket"] = customer["monetary"] / customer["frequency"]
    customer["max_ticket"] = grp["LineTotal"].max()
    customer["std_ticket"] = grp["LineTotal"].std().fillna(0.0)
    customer["n_products"] = grp["StockCode"].nunique()

    customer["last_purchase"] = grp["InvoiceDate"].max()
    customer["first_purchase"] = grp["InvoiceDate"].min()

    customer["recency_days"] = (reference_date - customer["last_purchase"]).dt.days
    customer["tenure_days"] = (customer["last_purchase"] - customer["first_purchase"]).dt.days

    customer["country"] = grp["Country"].agg(lambda x: x.mode().iloc[0])

    customer_features = customer.drop(columns=["last_purchase", "first_purchase"])
    return customer_features


def main():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["InvoiceDate"])
    customer_features = build_customer_features(df)

    customer_features.to_csv(CUSTOMER_FEATURES_PATH)
    print(f"Features de clientes guardadas en: {CUSTOMER_FEATURES_PATH}")
    print(f"Número de clientes: {len(customer_features)}")


if __name__ == "__main__":
    main()
