import duckdb

con = duckdb.connect()
total = con.execute("SELECT count(*) FROM 'data/abandoned_ag_gpp_2000_2022_SA.parquet'").fetchone()[0]
print(f"Total rows in raw data: {total}")

try:
    total_res = con.execute("SELECT count(*) FROM 'data/trajectory_results.parquet'").fetchone()[0]
    print(f"Total rows in processed data: {total_res}")
except Exception as e:
    print(e)
