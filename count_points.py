import glob, duckdb
import os
con = duckdb.connect()

total_sampled = 0
total_natural = 0

if os.path.exists('data/extracted_indices.parquet') and os.path.exists('data/dfsubsetNatural.parquet'):
    n_81_total = con.execute("SELECT count(*) FROM 'data/extracted_indices.parquet'").fetchone()[0]
    n_81_nat = con.execute("SELECT count(*) FROM 'data/dfsubsetNatural.parquet'").fetchone()[0]
    total_sampled += n_81_total
    total_natural += n_81_nat
    print(f'Eco Prototype: Total {n_81_total}, Natural {n_81_nat}')

ext_files = glob.glob('data/ref_samples_eco*.parquet')
ext_tot = 0
ext_nat = 0
for f in ext_files:
    n_tot = con.execute(f"SELECT count(*) FROM '{f}'").fetchone()[0]
    n_nat_f = con.execute(f'SELECT count(*) FROM \'{f}\' WHERE "natural"=1').fetchone()[0]
    ext_tot += n_tot
    ext_nat += n_nat_f
print(f'17 Ecos: Total {ext_tot}, Natural {ext_nat}')

total_sampled += ext_tot
total_natural += ext_nat
print(f'Grand Total: Sampled {total_sampled:,}, Natural {total_natural:,}')
