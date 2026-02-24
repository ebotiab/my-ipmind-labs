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

    return json, mo, os, pl, sqlalchemy


@app.cell
def _(pl):
    # Read both sheets   # noqa: F401

    file_path = "../data/Wi-Fi 6 TRUE POSITIVE AND NEGATIVES.xlsx"
    df_tp = pl.read_excel(
        file_path, sheet_name="TRUE POSITIVES"
    ).with_columns(  # ty:ignore[possibly-missing-attribute]
        pl.lit("TRUE POSITIVES").alias("label")
    )
    df_tn = pl.read_excel(
        file_path, sheet_name="TRUE NEGATIVES"
    ).with_columns(  # ty:ignore[possibly-missing-attribute]
        pl.lit("TRUE NEGATIVES").alias("label")
    )
    df_combined = pl.concat(
        [
            df_tp.select(
                pl.col("publication_number"), pl.col("claim_number"), pl.col("label")
            ),
            df_tn.select(
                pl.col("publication_number"), pl.col("claim_number"), pl.col("label")
            ),
        ]
    )

    claims_list = df_combined.select(
        [
            pl.col("publication_number"),
            pl.col("claim_number").alias("claims_number"),
            pl.col("label"),
        ]
    ).to_dicts()

    values_str = ", ".join(
        [
            f"('{c['publication_number']}', {c['claims_number']}, '{c['label']}')"
            for c in claims_list
            if c["publication_number"] is not None and c["claims_number"] is not None
        ]
    )
    return (values_str,)


@app.cell
def _(os, sqlalchemy):
    _password = os.environ.get("POSTGRES_PASSWORD")
    _username = os.environ.get("POSTGRES_USER")
    _host = os.environ.get("POSTGRES_SERVER")
    _database = os.environ.get("POSTGRES_DB")
    DATABASE_URL = f"postgresql://{_username}:{_password}@{_host}:6543/{_database}"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(engine, mo, values_str):
    claim_ids_df = mo.sql(
        f"""
        SELECT 
            t.publication_number,
            t.claims_number,
            t.label,
            pc.id AS claim_id
        FROM 
            (VALUES {values_str}) AS t(publication_number, claims_number, label)
        JOIN 
            public.patents p ON p.publication_number = t.publication_number
        JOIN 
            public.patent_claims pc ON pc.patent_id = p.id AND pc.number_in_patent = t.claims_number;
        """,
        engine=engine
    )
    return (claim_ids_df,)


@app.cell
def _(claim_ids_df, json, pl):
    # claim_ids_df is typically a Pandas DF when sqlalchemy engine is used by marimo.sql
    df_res = pl.DataFrame(claim_ids_df)
    results = df_res.to_dicts()

    output_data = {
        "TRUE POSITIVES": [
            str(r["claim_id"]) for r in results if r["label"] == "TRUE POSITIVES"
        ],
        "TRUE NEGATIVES": [
            str(r["claim_id"]) for r in results if r["label"] == "TRUE NEGATIVES"
        ],
    }

    with open("../data/wifi6_claim_ids.json", "w") as f:
        json.dump(output_data, f, indent=2)
    return (output_data,)


@app.cell
def _(output_data):
    output_data["TRUE POSITIVES"][0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
