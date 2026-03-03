import geopandas as gpd
import os
import pyogrio

geojson_path = r'c:\Users\coach\myfiles\postdoc2\code\data\spatial_subset_trajectories.geojson'
kml_path = r'c:\Users\coach\myfiles\postdoc2\code\data\spatial_subset_trajectories.kml'

def convert():
    if not os.path.exists(geojson_path):
        print(f"Error: {geojson_path} not found.")
        return

    print(f"Loading GeoJSON from {geojson_path} using pyogrio engine...")
    try:
        # Load GeoJSON with pyogrio
        gdf = gpd.read_file(geojson_path, engine='pyogrio')
        print(f"Loaded {len(gdf)} features.")
        
        # Ensure it has a reasonable CRS (KML requires WGS84)
        if gdf.crs is None:
            print("Warning: No CRS found in GeoJSON. Assuming EPSG:4326.")
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs != "EPSG:4326":
            print(f"Reprojecting from {gdf.crs} to EPSG:4326...")
            gdf = gdf.to_crs(epsg=4326)

        # Drop columns that KML driver might choke on (e.g. lists/dicts in properties)
        # KML driver in GDAL is sensitive to complex types.
        # Let's check columns for object types.
        for col in gdf.columns:
            if col != 'geometry':
                # Convert all columns to string to be safe for KML
                gdf[col] = gdf[col].astype(str)

        print(f"Writing KML to {kml_path}...")
        # Write using pyogrio explicitly
        try:
            # Note: valid drivers for pyogrio depend on GDAL build. 'KML' is standard.
            pyogrio.write_dataframe(gdf, kml_path, driver='KML')
            print("Success!")
        except Exception as e_pyogrio:
            print(f"Pyogrio write failed: {e_pyogrio}")
            print("Attempting fallback to geopandas to_file...")
            gdf.to_file(kml_path, driver='KML', engine='pyogrio')
            print("Success (fallback)!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert()
