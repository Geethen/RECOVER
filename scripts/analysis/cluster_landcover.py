"""
Aggregate SANLC land cover class percentages per trajectory cluster.
Join key: cast both lat/lon to float32 before comparing.
"""
import duckdb

RESULTS = r"c:\Users\coach\myfiles\postdoc2\code\data\trajectory_results_v2.parquet"
RAW     = r"c:\Users\coach\myfiles\postdoc2\code\data\abandoned_ag_gpp_2000_2022_SA.parquet"

SANLC_LABELS = {
    1: "Natural/near-natural",
    2: "Secondary natural (revegetated)",
    3: "Artificial water body",
    4: "Built-up",
    5: "Cropland",
    6: "Mine",
    7: "Plantation",
}

con = duckdb.connect()
con.execute("SET memory_limit='4GB'; SET threads=4")

print("Running aggregation query...")
df = con.execute(f"""
    WITH joined AS (
        SELECT r.cluster, s.sanlc_2022
        FROM '{RESULTS}' r
        JOIN '{RAW}' s
          ON CAST(s.latitude  AS FLOAT) = CAST(r.latitude  AS FLOAT)
         AND CAST(s.longitude AS FLOAT) = CAST(r.longitude AS FLOAT)
        WHERE s.sanlc_2022 IS NOT NULL
    ),
    totals AS (
        SELECT cluster, COUNT(*) AS cluster_total
        FROM joined GROUP BY cluster
    )
    SELECT j.cluster,
           j.sanlc_2022,
           COUNT(*)                                        AS n,
           ROUND(COUNT(*) * 100.0 / t.cluster_total, 2)   AS pct,
           t.cluster_total
    FROM joined j
    JOIN totals t USING (cluster)
    GROUP BY j.cluster, j.sanlc_2022, t.cluster_total
    ORDER BY j.cluster, n DESC
""").df()

con.close()

df["land_cover"] = df["sanlc_2022"].map(SANLC_LABELS).fillna("Unknown")

# --- View 1: per cluster → % of each land cover ---
print(f"\nTotal rows: {len(df)}")
print(f"Clusters: {sorted(df['cluster'].unique())}\n")

for cluster_id, grp in df.groupby("cluster"):
    label = "noise" if cluster_id == -1 else str(cluster_id)
    total = grp["cluster_total"].iloc[0]
    print(f"Cluster {label}  ({total:,} pixels)")
    print(f"  {'Land cover':<35} {'n':>10}  {'%':>6}")
    print(f"  {'-'*55}")
    for _, row in grp.iterrows():
        print(f"  {row['land_cover']:<35} {int(row['n']):>10,}  {row['pct']:>5.1f}%")
    print()

# --- View 2: per land cover → % assigned to each cluster ---
print("\n" + "="*70)
print("LAND COVER -> CLUSTER DISTRIBUTION")
print("="*70)

lc_totals = df.groupby("sanlc_2022")["n"].sum().rename("lc_total")
df2 = df.join(lc_totals, on="sanlc_2022")
df2["pct_of_lc"] = (df2["n"] / df2["lc_total"] * 100).round(2)

cluster_labels = {c: ("noise" if c == -1 else str(c)) for c in df["cluster"].unique()}

for lc_id, grp in df2.groupby("sanlc_2022"):
    lc_name = SANLC_LABELS.get(lc_id, "Unknown")
    lc_total = grp["lc_total"].iloc[0]
    grp = grp.sort_values("n", ascending=False)
    print(f"\n{lc_name}  ({lc_total:,} pixels total)")
    print(f"  {'Cluster':<10} {'n':>12}  {'%':>6}")
    print(f"  {'-'*32}")
    for _, row in grp.iterrows():
        clabel = cluster_labels[row["cluster"]]
        print(f"  {clabel:<10} {int(row['n']):>12,}  {row['pct_of_lc']:>5.1f}%")

out_path = r"c:\Users\coach\myfiles\postdoc2\code\data\cluster_landcover.csv"
df.to_csv(out_path, index=False)
out_path2 = r"c:\Users\coach\myfiles\postdoc2\code\data\landcover_by_cluster.csv"
df2[["sanlc_2022","land_cover","cluster","n","lc_total","pct_of_lc"]].to_csv(out_path2, index=False)
print(f"\nSaved to {out_path} and {out_path2}")
