import duckdb

try:
    con = duckdb.connect()
    count_n = con.execute("SELECT COUNT(*) FROM 'data/dfsubsetNatural.parquet'").fetchone()[0]
    count_81 = con.execute("""
        SELECT COUNT(*) 
        FROM 'data/dfsubsetNatural.parquet' n 
        JOIN 'data/indices_gpp_svh_2000_2022.parquet' g 
          ON n.id = split_part(g.pixel_id, '_', 1) 
        WHERE g.eco_id = 81
    """).fetchone()[0]
    print(f"dfsubsetNatural total: {count_n}")
    print(f"eco_id 81 count: {count_81}")
except Exception as e:
    print(f"Error: {e}")
