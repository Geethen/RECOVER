"""
Export 5 spatial subsets (100k pixels each) from trajectory_results.parquet
to a single GeoPackage for QGIS visualisation.

Each subset is independently reservoir-sampled so they cover different spatial
areas. All 5 land in one .gpkg file as separate layers (subset_1 … subset_5),
which QGIS loads directly with full symbology support.

Output: data/cluster_subsets.gpkg
"""

import os
import duckdb
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "cluster_subsets.gpkg")

N_SUBSETS   = 5
SAMPLE_SIZE = 100_000


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Results not found: {RESULTS_PATH}")

    con = duckdb.connect()

    # Quick sanity-check on cluster distribution
    dist = con.execute(f"""
        SELECT cluster, COUNT(*) AS n
        FROM '{RESULTS_PATH}'
        GROUP BY cluster ORDER BY cluster
    """).df()
    print("Cluster distribution:")
    print(dist.to_string(index=False))
    print()

    for i in range(1, N_SUBSETS + 1):
        seed = i * 137  # independent seeds
        print(f"Sampling subset {i}/{N_SUBSETS} (seed={seed})...")

        df = con.execute(f"""
            SELECT pixel_id,
                   latitude,
                   longitude,
                   cluster::INTEGER      AS cluster,
                   cluster_prob::FLOAT   AS cluster_prob
            FROM '{RESULTS_PATH}'
            USING SAMPLE reservoir({SAMPLE_SIZE} ROWS) REPEATABLE ({seed})
        """).df()

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        # Drop redundant lat/lon columns — geometry column carries them
        gdf = gdf.drop(columns=["latitude", "longitude"])

        layer = f"subset_{i}"
        gdf.to_file(OUTPUT_PATH, layer=layer, driver="GPKG", engine="pyogrio")
        print(f"  Saved layer '{layer}' ({len(gdf):,} points, {df['cluster'].value_counts().to_dict()})")

    con.close()

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nDone. {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("Open in QGIS: Layer > Add Layer > Add Vector Layer > select the .gpkg")
    print("Tip: symbolise by 'cluster' with categorised renderer to colour-code clusters.")


if __name__ == "__main__":
    main()
