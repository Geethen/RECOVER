import duckdb

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")

df = con.sql("""
    SELECT sanlc_2022, COUNT(pixel_id) as count 
    FROM 'data/abandoned_ag_gpp_2000_2022_SA.parquet' 
    GROUP BY sanlc_2022 
    ORDER BY count DESC
""").df()

print(df)
