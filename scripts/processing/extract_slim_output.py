"""
Post-process trajectory_results.parquet to produce a lean output file
containing only pixel_id, latitude, longitude, cluster, cluster_prob.

Run after trajectory_classifier.py has finished:
  python scripts/extract_slim_output.py
"""
import os
import duckdb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
SLIM_PATH = os.path.join(BASE_DIR, "data", "clustering_output.parquet")

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Results not found at {RESULTS_PATH}. Run trajectory_classifier.py first.")
        return

    con = duckdb.connect()

    total = con.execute(f"SELECT COUNT(*) FROM '{RESULTS_PATH}'").fetchone()[0]
    print(f"Total pixels in results: {total:,}")

    cluster_counts = con.execute(f"""
        SELECT cluster, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / {total}, 2) AS pct
        FROM '{RESULTS_PATH}'
        GROUP BY cluster
        ORDER BY cluster
    """).df()
    print("\nCluster distribution:")
    print(cluster_counts.to_string(index=False))

    print(f"\nWriting slim output to {SLIM_PATH} ...")
    con.execute(f"""
        COPY (
            SELECT pixel_id, latitude, longitude, cluster, cluster_prob
            FROM '{RESULTS_PATH}'
            ORDER BY pixel_id
        ) TO '{SLIM_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    slim_size = os.path.getsize(SLIM_PATH) / (1024 ** 2)
    print(f"Done. Slim output: {slim_size:.1f} MB  ({SLIM_PATH})")

if __name__ == "__main__":
    main()
