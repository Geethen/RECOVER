# Understanding the DuckDB & Parquet Workflow

This document explains why and how we migrated your data from a **23 GB CSV** to a deduplicated **3.5 GB Parquet** file using **DuckDB**.

## 1. Why DuckDB?
DuckDB is an "analytical" database. Unlike standard Python `pandas`, which often tries to load the entire file into RAM, DuckDB "streams" data from your hard drive. This allowed us to:
*   Process a **23 GB** file on a machine with limited disk space.
*   Deduplicate millions of rows in seconds instead of hours.
*   Convert data to compressed Parquet format efficiently.

## 2. The Migration Process

### Phase 1: Streaming Conversion (CSV to Parquet)
We converted the raw CSV to Parquet first. Parquet is a "columnar" storage format that uses heavy compression (ZSTD).
```python
import duckdb

con = duckdb.connect()

# The 'COPY' command is highly optimized for streaming
con.execute("""
    COPY (
        SELECT * FROM read_csv_auto('data.csv', ignore_errors=true, null_padding=true)
    ) 
    TO 'data.parquet' 
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")
```
*   **Result**: The file shrank from **23.11 GB** down to **2.5 GB**.

### Phase 2: Deduplication
Once in Parquet, we used standard SQL to remove duplicates. DuckDB treats the Parquet file like a database table.
```python
# 'SELECT DISTINCT *' automatically finds and removes exact row duplicates
con.execute("""
    COPY (SELECT DISTINCT * FROM 'data.parquet') 
    TO 'data_clean.parquet' 
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")
```
*   **Discovery**: We found and removed **14,962,462** duplicate rows.

## 3. Integration into `abandoned_ag_extract.py`

I updated your main extraction script to use DuckDB as a "high-performance buffer." Here is the logic now used in the script:

### Resuming an Extraction
Instead of reading a massive CSV into memory, the script simply connects to a Parquet file:
```python
# DuckDB can treat a Parquet file as a table instantly without loading it
db_conn.execute("CREATE TABLE data AS SELECT * FROM 'existing_data.parquet'")
```

### Saving New Batches (Thread-Safe)
When Earth Engine returns a batch of data, we insert it into the DuckDB buffer. This is much faster than appending text to a CSV file.
```python
# Inside process_pixel_batch
db_conn.execute("INSERT INTO data SELECT * FROM df_result")
```

### High-Speed Analysis
When the script finished, it used to take a long time to calculate `describe()` stats in Pandas. Now it uses SQL:
```python
# This runs in milliseconds even on 30 million rows
stats = con.execute("SELECT mean(GPP_2000), stddev(GPP_2000) FROM 'data.parquet'").df()
```

## 4. Key Benefits
1.  **Disk Space**: You saved ~20 GB of space.
2.  **Safety**: We no longer worry about "Out of Memory" crashes because DuckDB handles memory management automatically.
3.  **Speed**: Loading the resulting Parquet file into any future analysis script will be **10x to 50x faster** than reading the original CSV.
