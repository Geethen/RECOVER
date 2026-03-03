import duckdb
import pandas as pd
import os

data_dir = r"c:\Users\coach\myfiles\postdoc2\code\data"

f1 = os.path.join(data_dir, "trajectory_analysis_results.parquet")
f2 = os.path.join(data_dir, "abandoned_ag_gpp_2000_2022_SA.parquet")

print(f"Checking {f1}...")
try:
    cols1 = duckdb.sql(f"SELECT * FROM '{f1}' LIMIT 0").df().columns.tolist()
    print(cols1)
except Exception as e:
    print(f"Error: {e}")

print(f"\nChecking {f2}...")
try:
    cols2 = duckdb.sql(f"SELECT * FROM '{f2}' LIMIT 0").df().columns.tolist()
    print(cols2)
except Exception as e:
    print(f"Error: {e}")
