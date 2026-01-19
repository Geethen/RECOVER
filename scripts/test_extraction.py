import pandas as pd
from gee_extraction import extract_efficiently
import os
import ee

# Asset ID provided by user
ASSET_ID = "projects/ee-gsingh/assets/RECOVER/fscs_aef_samples"
OUTPUT_CSV = "extracted_indices.csv"

def run_test():
    if os.path.exists(OUTPUT_CSV):
        print(f"Removing existing test output: {OUTPUT_CSV}")
        os.remove(OUTPUT_CSV)

    print(f"Starting extraction for asset: {ASSET_ID}", flush=True)
    
    # Explicitly initialize as requested, though gee_extraction does it too.
    try:
        ee.Initialize(project='ee-gsingh')
        print("Explicit initialization successful.", flush=True)
    except Exception as e:
        print(f"Explicit initialization failed (might be handled in module): {e}", flush=True)
    
    try:
        # Run extraction
        extract_efficiently(ASSET_ID, OUTPUT_CSV)
        
        # Verify Output
        if os.path.exists(OUTPUT_CSV):
            res = pd.read_csv(OUTPUT_CSV)
            print("\nExtraction Results Summary:")
            print(res.head())
            print(f"Total rows: {len(res)}")
        else:
            print("\nWarning: Output CSV not created (might mean no points were processed or all failed).")
            
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
