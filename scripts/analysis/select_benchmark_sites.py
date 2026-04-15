import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from tqdm import tqdm
from scipy.interpolate import interp1d, PchipInterpolator
import geopandas as gpd
from shapely.geometry import Point

def extract_coords(geo_str):
    if not isinstance(geo_str, str):
        return [np.nan, np.nan]
    try:
        geo_dict = json.loads(geo_str.replace("'", '"'))
        return geo_dict['coordinates']
    except:
        return [np.nan, np.nan]

def half_cauchy_weight(distance, lam=2.0):
    """
    Half-Cauchy weighting formula.
    w = 1 / (1 + (dist/lam)^2)
    """
    return 1.0 / (1.0 + (distance / lam)**2)

def select_benchmarks(data_path, output_path, shp_path):
    print(f"--- HCAS v3.1 Style Benchmarking (Optimized) ---")
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 1. Feature Definition
    # Benchmark selection uses PREDICTED REFERENCE values (environmental potential)
    dist_cols = ['NBR_ref', 'NDMI_ref', 'NDWI_ref']
    # Scoring uses DEPARTURES (Difference between Actual and Predicted)
    score_cols = ['NBR_diff', 'NDMI_diff', 'NDWI_diff']
    
    # 2. Coordinate Extraction
    print("Extracting coordinates...")
    coords = df['geo_x'].apply(extract_coords).tolist()
    df[['lon', 'lat']] = pd.DataFrame(coords, index=df.index)
    
    # 3. Data Split
    df_nat = df[df['natural'] == 1].copy()
    df_trans = df[df['natural'] == 0].copy()
    
    print(f"Reference pool size: {len(df_nat)}")
    print(f"Test sites size: {len(df_trans)}")
    
    if len(df_trans) == 0:
        print("No transformed sites found. Sampling natural sites for demo.")
        df_trans = df_nat.sample(min(500, len(df_nat)), random_state=42)

    # 4. Standardization
    # Standardize values for distance calculation (Ref Values)
    print(f"\nStandardizing reference conditions {dist_cols} for benchmark selection...")
    scaler_dist = StandardScaler()
    X_nat_dist = df_nat[dist_cols].fillna(0).values
    X_trans_dist = df_trans[dist_cols].fillna(0).values
    X_nat_dist_scaled = scaler_dist.fit_transform(X_nat_dist)
    X_trans_dist_scaled = scaler_dist.transform(X_trans_dist)

    # Standardize values for scoring (Departures)
    # We use a separate scaler to ensure departue units are comparable
    print(f"Standardizing departures {score_cols} for scoring...")
    scaler_score = StandardScaler()
    X_nat_score = df_nat[score_cols].fillna(0).values
    X_trans_score = df_trans[score_cols].fillna(0).values
    X_nat_score_scaled = scaler_score.fit_transform(X_nat_score)
    X_trans_score_scaled = scaler_score.transform(X_trans_score)
    
    # 5. Spatial Index
    print("Building spatial index for reference sites...")
    nat_coords_rad = np.radians(df_nat[['lat', 'lon']].values)
    tree = BallTree(nat_coords_rad, metric='haversine')
    
    # HCAS v3.1 Parameters
    BENCHMARK_INCLUSION_KM = 200.0
    TOP_K_RS = 70
    PENALTY_FACTOR = 30.0
    FINAL_K = 10
    CAUCHY_LAMBDA = 2.0
    CALIB_POINTS_X = [0.0, 1.0]
    CALIB_POINTS_Y = [0.101, 0.944]
    
    spline_calib = PchipInterpolator(CALIB_POINTS_X, CALIB_POINTS_Y)

    # Pre-compute Reference-to-Reference Background Histogram (Density)
    print("Constructing background density distribution...")
    sample_size = 2000
    ref_sample_idx = np.random.choice(len(df_nat), sample_size, replace=False)
    ref_pairs_dists = []
    
    # Use 1000km zone for global density baseline
    radius_1000_rad = 1000.0 / 6371.0
    for idx_s in tqdm(ref_sample_idx[:200], desc="Sampling Reference Density"):
        pivot_feat = X_nat_dist_scaled[idx_s]
        pivot_coord = nat_coords_rad[idx_s:idx_s+1]
        neighbors_idx = tree.query_radius(pivot_coord, r=radius_1000_rad)[0]
        if len(neighbors_idx) > 1:
            n_sample = np.random.choice(neighbors_idx, min(100, len(neighbors_idx)), replace=False)
            neighbor_feats = X_nat_dist_scaled[n_sample]
            # Manhattan distance on Reference Conditions
            rs_manhattan = np.sum(np.abs(neighbor_feats - pivot_feat), axis=1)
            ref_pairs_dists.extend(rs_manhattan.tolist())
    
    # Histogram configuration: 400 bins, 0.05 width
    hist_counts, bin_edges = np.histogram(ref_pairs_dists, bins=400, range=(0, 20.0), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    density_interp = interp1d(bin_centers, hist_counts + 1e-9, kind='linear', fill_value=(1e-9, 1e-9), bounds_error=False)
    
    results = []
    print(f"Processing {len(df_trans)} test sites...")
    
    for i in tqdm(range(len(df_trans)), desc="Benchmarking"):
        test_site = df_trans.iloc[i]
        test_feat_dist = X_trans_dist_scaled[i]
        test_coord_rad = np.radians([[test_site['lat'], test_site['lon']]])
        
        # Step 1: Spatial filter (200km radius)
        radius_200_rad = BENCHMARK_INCLUSION_KM / 6371.0
        idx_spatial = tree.query_radius(test_coord_rad, r=radius_200_rad)[0]
        
        if len(idx_spatial) < FINAL_K:
            _, idx_spatial = tree.query(test_coord_rad, k=100)
            idx_spatial = idx_spatial[0]
            
        # Manhattan distance using Reference Conditions
        nat_ref_feats_filtered = X_nat_dist_scaled[idx_spatial]
        rs_distances = np.sum(np.abs(nat_ref_feats_filtered - test_feat_dist), axis=1)
        
        # Step 4: Filter top 70 closest in RS distance
        top_70_rank = np.argsort(rs_distances)[:min(len(rs_distances), TOP_K_RS)]
        idx_top_70 = idx_spatial[top_70_rank]
        rs_dist_top_70 = rs_distances[top_70_rank]
        
        # Step 5: Apply distance penalty
        top_70_coords = nat_coords_rad[idx_top_70]
        dlat = top_70_coords[:, 0] - test_coord_rad[0, 0]
        dlon = top_70_coords[:, 1] - test_coord_rad[0, 1]
        a = np.sin(dlat/2)**2 + np.cos(test_coord_rad[0, 0]) * np.cos(top_70_coords[:, 0]) * np.sin(dlon/2)**2
        geo_dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        rs_dist_adj = rs_dist_top_70 * (1.0 + geo_dist_km / PENALTY_FACTOR)
        
        # Step 6: Select top 10 with HIGHEST HISTOGRAM FREQUENCY
        rs_frequencies = density_interp(rs_dist_adj)
        freq_rank = np.argsort(-rs_frequencies)[:min(len(rs_frequencies), FINAL_K)]
        
        best_indices = idx_top_70[freq_rank]
        final_penalized_dists = rs_dist_adj[freq_rank]
        weights = half_cauchy_weight(final_penalized_dists, lam=CAUCHY_LAMBDA)
        weights /= np.sum(weights)
        
        # Step 7: Calculate SCORING departure
        # Instead of averaging the benchmark features (centroid), we calculate
        # the distance to each benchmark and THEN average the distances.
        # This prevents "averaging out" the departures and results in a more realistic (higher) departure.
        
        test_dep_vector = X_trans_score_scaled[i] # Shape (3,)
        bench_dep_vectors = X_nat_score_scaled[best_indices] # Shape (10, 3)
        
        # Calculate Manhattan distance for each benchmark pair
        # (|T_nbr - B_nbr| + |T_ndmi - B_ndmi| + ...)
        pair_departures = np.sum(np.abs(bench_dep_vectors - test_dep_vector), axis=1) # Shape (10,)
        
        # Weighted Average of the departures
        comp_departure = np.average(pair_departures, weights=weights)
        
        # Step 8: Calibrate to HCAS Score [0, 1]
        # Using a threshold of 2.0 (sum of standardized differences) for mapping
        uncalibrated_val = np.clip(1.0 - (comp_departure / 2.0), 0, 1)
        calibrated_hcas = float(spline_calib(uncalibrated_val))
        
        results.append({
            'id': test_site['id'],
            'comp_dep': comp_departure,
            'HCAS_score': calibrated_hcas
        })

    df_results = pd.DataFrame(results)
    df_final = df_trans.merge(df_results, on='id', how='left')
    
    # Save CSV
    print(f"\nSaving results to {output_path}...")
    df_final.to_parquet(output_path, index=False, compression="zstd", compression_level=3)
    
    # Save Shapefile
    print(f"Exporting HCAS scores to Shapefile: {shp_path}...")
    geometry = [Point(xy) for xy in zip(df_final.lon, df_final.lat)]
    # Export simple ID and score for transformed areas
    out_cols = ['id', 'comp_dep', 'HCAS_score']
    gdf = gpd.GeoDataFrame(df_final[out_cols], geometry=geometry, crs="EPSG:4326")
    gdf.to_file(shp_path)
    
    print("\n--- Summary ---")
    grand_mean = df_final['HCAS_score'].mean()
    print(f"Grand Mean HCAS Condition Score: {grand_mean:.4f}")

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\coach\myfiles\postdoc2\code"
    INPUT_FILE = os.path.join(BASE_DIR, "data", "reference_departure_with_intervals.parquet")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "benchmarked_condition.parquet")
    SHP_FILE = os.path.join(BASE_DIR, "data", "transformed_hcas.shp")
    
    if os.path.exists(INPUT_FILE):
        select_benchmarks(INPUT_FILE, OUTPUT_FILE, SHP_FILE)
    else:
        print(f"File not found: {INPUT_FILE}")

