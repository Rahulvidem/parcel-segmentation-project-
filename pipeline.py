import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ── Path setup (works locally AND on Streamlit Cloud) ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_raw_data():
    clients_path = os.path.join(DATA_DIR, "clients.csv")
    properties_path = os.path.join(DATA_DIR, "properties.csv")
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)
    return clients, properties


def clean_clients(clients: pd.DataFrame) -> pd.DataFrame:
    df = clients.copy()

    # Parse date_of_birth → age
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], format="mixed", errors="coerce")
    today = pd.Timestamp.today()
    df["age"] = ((today - df["date_of_birth"]).dt.days / 365.25).round(1)
    df["age"] = df["age"].fillna(df["age"].median())

    # Binary flags
    df["loan_applied_flag"] = (df["loan_applied"].str.strip().str.lower() == "yes").astype(int)
    df["is_individual"] = (df["client_type"].str.strip().str.lower() == "individual").astype(int)

    # Satisfaction score – fill missing with median
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    return df


def clean_properties(properties: pd.DataFrame) -> pd.DataFrame:
    df = properties.copy()

    # Parse sale_price  "$300,385.62" → float
    df["sale_price_num"] = (
        df["sale_price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["sale_price_num"] = df["sale_price_num"].fillna(df["sale_price_num"].median())

    # Boolean sold flag
    df["is_sold"] = (df["listing_status"].str.strip().str.lower() == "sold").astype(int)

    # Parse transaction date
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="mixed", errors="coerce")

    return df


def build_features(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """Merge client + property aggregates into one feature table."""
    props_clean = clean_properties(properties)
    clients_clean = clean_clients(clients)

    # Aggregate property stats per client
    sold = props_clean[props_clean["is_sold"] == 1]
    client_prop = (
        sold.groupby("client_ref")
        .agg(
            total_spent=("sale_price_num", "sum"),
            avg_price=("sale_price_num", "mean"),
            num_purchases=("listing_id", "count"),
            avg_area=("floor_area_sqft", "mean"),
            has_office=("unit_category", lambda x: int((x == "Office").any())),
        )
        .reset_index()
        .rename(columns={"client_ref": "client_id"})
    )

    features = clients_clean.merge(client_prop, on="client_id", how="left")
    features["total_spent"] = features["total_spent"].fillna(0)
    features["avg_price"] = features["avg_price"].fillna(0)
    features["num_purchases"] = features["num_purchases"].fillna(0)
    features["avg_area"] = features["avg_area"].fillna(0)
    features["has_office"] = features["has_office"].fillna(0)

    return features


FEATURE_COLS = [
    "age",
    "satisfaction_score",
    "loan_applied_flag",
    "is_individual",
    "total_spent",
    "avg_price",
    "num_purchases",
    "avg_area",
    "has_office",
]


def run_clustering(k: int = 4):
    clients, properties = load_raw_data()
    features = build_features(clients, properties)

    X = features[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features["cluster"] = kmeans.fit_predict(X_scaled)

    # PCA for 2-D scatter
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    features["pca_x"] = coords[:, 0]
    features["pca_y"] = coords[:, 1]

    # Silhouette & elbow scores for k = 2..8
    elbow_scores, silhouette_scores = [], []
    k_range = range(2, 9)
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        elbow_scores.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    artifacts = {
        "features": features,
        "kmeans": kmeans,
        "scaler": scaler,
        "k": k,
        "k_range": list(k_range),
        "elbow_scores": elbow_scores,
        "silhouette_scores": silhouette_scores,
        "feature_cols": FEATURE_COLS,
        "X_scaled": X_scaled,
    }
    return artifacts
