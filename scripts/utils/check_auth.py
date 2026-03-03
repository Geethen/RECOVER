import ee
import sys

print("Checking GEE Authentication...", flush=True)
try:
    ee.Initialize(project='ee-gsingh')
    print("Authentication Successful!", flush=True)
    print(ee.Image("COPERNICUS/S2_SR_HARMONIZED").limit(1).getInfo(), flush=True)
except Exception as e:
    print(f"Authentication Failed: {e}", flush=True)
