import importlib.util

modules = ['fiona', 'pyogrio', 'simplekml']
print("Module check:")
for m in modules:
    spec = importlib.util.find_spec(m)
    print(f"{m}: {'FOUND' if spec else 'MISSING'}")
