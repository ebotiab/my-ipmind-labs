import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os

    import marimo as mo  # noqa: F401
    import polars as pl
    import sqlalchemy

    BATCH_ID = "Aggregated"
    return json, mo, os, pl, sqlalchemy


@app.cell
def _(json):
    with open("data/wifi6_claim_ids.json", "r") as _f:
        claim_ids_data = json.load(_f)

    # Flatten all claim IDs into a single list
    tp_ids = claim_ids_data.get("TRUE POSITIVES", [])
    tn_ids = claim_ids_data.get("TRUE NEGATIVES", [])
    all_claim_ids = tp_ids + tn_ids

    # Format for SQL IN clause
    claim_ids_str = ", ".join([f"'{cid}'" for cid in all_claim_ids])
    return claim_ids_str, tn_ids, tp_ids


@app.cell
def _(os, sqlalchemy):
    _password = os.environ.get("POSTGRES_PASSWORD")
    _username = os.environ.get("POSTGRES_USER")
    _host = os.environ.get("POSTGRES_SERVER")
    _database = os.environ.get("POSTGRES_DB")

    if all([_password, _username, _host, _database]):
        DATABASE_URL = f"postgresql://{_username}:{_password}@{_host}:6543/{_database}"
    else:
        # Fallback for local testing if env vars are not set directly
        DATABASE_URL = "postgresql://postgres:postgres@localhost:6543/postgres"

    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(claim_ids_str, engine, mo):
    results_df = mo.sql(
        f"""
        SELECT DISTINCT ON (j.claim_id)
            j.id::text AS job_id,
            j.claim_id::text,
            pr.essentiality_likelihood::float
        FROM 
            public.jobs j
        JOIN 
            public.prism_results pr ON pr.job_id = j.id
        WHERE 
            j.batch_id = BATCH_ID
            AND j.claim_id::text IN ({claim_ids_str})
        ORDER BY 
            j.claim_id, 
            j.created_at DESC;
        """,
        engine=engine,
    )
    return (results_df,)


@app.cell
def _(pl, results_df, tn_ids, tp_ids):

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    # ---------------------------------------------------------
    # PASO 1: Filtrar, Etiquetar y Predecir (Vectorizado)
    # ---------------------------------------------------------

    valid_ids = tp_ids + tn_ids
    df_processed = results_df.filter(pl.col("claim_id").is_in(valid_ids)).with_columns(
        label=pl.when(pl.col("claim_id").is_in(tp_ids)).then(1).otherwise(0),
        prediction=pl.when(pl.col("essentiality_likelihood").fill_null(0.0) > 0.7)
        .then(1)
        .otherwise(0),
    )
    output_data = df_processed.to_dicts()

    # ---------------------------------------------------------
    # PASO 2: Calcular Métricas con Scikit-Learn
    # ---------------------------------------------------------
    y_true = df_processed["label"].to_list()
    y_pred = df_processed["prediction"].to_list()

    # Extraemos la matriz de confusión. labels=[0, 1] previene errores si faltan clases.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculamos todas las métricas. zero_division=0.0 previene errores matemáticos.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics, output_data


@app.cell
def _(json, output_data):
    output_path = "data/wifi6_prism_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    return


@app.cell
def _(metrics, output_data):
    print(f"Total records: {len(output_data)}")
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
