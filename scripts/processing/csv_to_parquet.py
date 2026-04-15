import duckdb
import os
import shutil
import time
import threading

# Paths
csv_path = r'c:\Users\coach\myfiles\postdoc2\code\data\abandoned_ag_gpp_2000_2022_SA.csv'
parquet_path = r'c:\Users\coach\myfiles\postdoc2\code\data\abandoned_ag_gpp_2000_2022_SA.parquet'

stop_monitoring = False

def get_free_space_gb():
    total, used, free = shutil.disk_usage("C:")
    return free / (1024**3)

def monitor_progress():
    """Check file size and disk space every 15 seconds"""
    print(f"{'Time':<10} | {'Parquet Size':<15} | {'Free Space':<15}")
    print("-" * 45)
    while not stop_monitoring:
        if os.path.exists(parquet_path):
            p_size = os.path.getsize(parquet_path) / (1024**3)
        else:
            p_size = 0.0
        
        f_space = get_free_space_gb()
        current_time = time.strftime("%H:%M:%S")
        
        print(f"{current_time:<10} | {p_size:>12.2f} GB | {f_space:>12.2f} GB", flush=True)
        
        if f_space < 0.2: # Emergency stop if space < 200MB
            print("\n!!! EMERGENCY STOP: Disk space critically low!")
            break
            
        time.sleep(15)

def convert():
    global stop_monitoring
    print(f"Initial free space: {get_free_space_gb():.2f} GB")
    
    if os.path.exists(parquet_path):
        os.remove(parquet_path)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false;") # Speeds up writing
    con.execute("SET enable_progress_bar=true;") # Internal progress bar
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print(f"Starting conversion...")
    try:
        start_time = time.time()
        # Using ZSTD for maximum compression
        # ignore_errors and null_padding handle malformed rows in large CSVs
        con.execute(f"""
            COPY (SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=true, null_padding=true)) 
            TO '{parquet_path}' 
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        stop_monitoring = True
        monitor_thread.join()

        end_time = time.time()
        duration = (end_time - start_time) / 60
        
        file_size = os.path.getsize(parquet_path) / (1024**3)
        print(f"\nFinal Parquet Size: {file_size:.2f} GB")
        print(f"Time taken: {duration:.2f} minutes")
        print(f"Final free space: {get_free_space_gb():.2f} GB")

    except Exception as e:
        stop_monitoring = True
        print(f"\nError: {e}")
        if os.path.exists(parquet_path):
            os.remove(parquet_path)

if __name__ == "__main__":
    convert()
