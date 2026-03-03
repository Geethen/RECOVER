"""
Test new features:
  - Landsat 5/7 C2 L2 for pre-2013 years
  - period_months aggregation (1=monthly, 4=seasonal, 12=annual)
"""
import ee, requests, calendar, sys
import matplotlib.image as mpimg
from io import BytesIO

ee.Initialize(project='ee-gsingh')

lat, lon = -28.85492, 22.04452
chip_km  = 3
point    = ee.Geometry.Point([lon, lat])
region   = point.buffer(chip_km * 1000 / 2).bounds()

_SPECTRAL = ['RED', 'GREEN', 'BLUE', 'NIR', 'SWIR1', 'SWIR2']

HLS_STD_IN = {
    'L30': ['B4', 'B3', 'B2', 'B5',  'B6',  'B7',  'Fmask'],
    'S30': ['B4', 'B3', 'B2', 'B8A', 'B11', 'B12', 'Fmask'],
}
HLS_STD_OUT = _SPECTRAL + ['Fmask']

RGB_PARAMS = {
    'SWIR1/NIR/GREEN': dict(bands=['SWIR1','NIR','GREEN'], min=0.0, max=0.45, gamma=1.0),
    'RED/GREEN/BLUE':  dict(bands=['RED','GREEN','BLUE'],  min=0.01, max=0.18, gamma=1.2),
}

# ── Date helpers ──────────────────────────────────────────────────────────────

def _period_end_dt(year, start_month, period_months):
    """Last day of the period (handles year-wrap)."""
    last_month = start_month + period_months - 1
    end_year   = year + (last_month - 1) // 12
    end_month  = (last_month - 1) % 12 + 1
    return f'{end_year}-{end_month:02d}-{calendar.monthrange(end_year, end_month)[1]}'

def _period_label(start_month, period_months):
    abbr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    if period_months == 1:
        return abbr[start_month - 1]
    last_month = (start_month + period_months - 2) % 12 + 1
    return f'{abbr[start_month-1]}-{abbr[last_month-1]}'

# ── Cloud masking ─────────────────────────────────────────────────────────────

def _mask_hls_clouds(img):
    fmask = img.select('Fmask')
    bits  = (1 << 1) | (1 << 2) | (1 << 3)
    return img.updateMask(fmask.bitwiseAnd(bits).eq(0))

def _mask_landsat_clouds(img):
    qa   = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(mask)

def _scale_rename_ls57(img):
    """Scale Landsat 5/7 C2 L2 SR bands and rename to standard names."""
    scaled = (img.select(['SR_B3','SR_B2','SR_B1','SR_B4','SR_B5','SR_B7'])
                 .multiply(0.0000275).add(-0.2)
                 .rename(_SPECTRAL))
    return scaled.addBands(img.select('QA_PIXEL'))

# ── Unified chip function ─────────────────────────────────────────────────────

def get_chip(lat, lon, year, month, period_months=1,
             vis_combo='SWIR1/NIR/GREEN', chip_km=3, hls_source='both'):
    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(chip_km * 1000 / 2).bounds()
    start_dt = f'{year}-{month:02d}-01'
    end_dt   = _period_end_dt(year, month, period_months)

    if year >= 2013:
        # ── HLS ──────────────────────────────────────────────────────────────
        parts = []
        if hls_source in ('L30', 'both'):
            parts.append(ee.ImageCollection('NASA/HLS/HLSL30/v002')
                         .filterBounds(point).filterDate(start_dt, end_dt)
                         .filter(ee.Filter.lt('CLOUD_COVERAGE', 30))
                         .select(HLS_STD_IN['L30'], HLS_STD_OUT))
        if hls_source in ('S30', 'both'):
            parts.append(ee.ImageCollection('NASA/HLS/HLSS30/v002')
                         .filterBounds(point).filterDate(start_dt, end_dt)
                         .filter(ee.Filter.lt('CLOUD_COVERAGE', 30))
                         .select(HLS_STD_IN['S30'], HLS_STD_OUT))
        merged = parts[0].merge(parts[1]) if len(parts) == 2 else parts[0]
        masked = merged.map(_mask_hls_clouds).select(_SPECTRAL)
    else:
        # ── Landsat 5 + 7 C2 L2 ─────────────────────────────────────────────
        def prep_ls(col_id):
            return (ee.ImageCollection(col_id)
                    .filterBounds(point).filterDate(start_dt, end_dt)
                    .filter(ee.Filter.lt('CLOUD_COVER', 30))
                    .map(_scale_rename_ls57)
                    .map(_mask_landsat_clouds)
                    .select(_SPECTRAL))
        merged = prep_ls('LANDSAT/LT05/C02/T1_L2').merge(
                 prep_ls('LANDSAT/LE07/C02/T1_L2'))
        masked = merged

    n = merged.size().getInfo()

    blank = ee.Image.constant([0] * 6).rename(_SPECTRAL)
    img   = ee.Image(ee.Algorithms.If(
        merged.size().gt(0),
        masked.median().unmask(0),
        blank,
    ))

    vp      = RGB_PARAMS[vis_combo]
    vis_img = img.select(vp['bands']).visualize(min=vp['min'], max=vp['max'], gamma=vp['gamma'])
    url     = vis_img.getThumbURL(dict(dimensions=256, region=region, format='png'))
    resp    = requests.get(url, timeout=60)
    return n, resp

# ── Run tests ────────────────────────────────────────────────────────────────

tests = [
    # Landsat monthly
    ('LS57 2000-06 monthly (p=1)',     dict(year=2000, month=6,  period_months=1)),
    ('LS57 2005-07 monthly (p=1)',     dict(year=2005, month=7,  period_months=1)),
    ('LS57 2012-09 monthly (p=1)',     dict(year=2012, month=9,  period_months=1)),
    # Landsat seasonal
    ('LS57 2005 Jun-Sep (p=4)',        dict(year=2005, month=6,  period_months=4)),
    ('LS57 2005 annual  (p=12)',       dict(year=2005, month=1,  period_months=12)),
    # Landsat edge cases
    ('LS57 2003-06 (SLC-off L7 era)', dict(year=2003, month=6,  period_months=1)),
    ('LS57 2000-01 empty? (p=1)',      dict(year=2000, month=1,  period_months=1)),
    # HLS with period_months > 1
    ('HLS 2018 Jun-Sep (p=4)',         dict(year=2018, month=6,  period_months=4)),
    ('HLS 2018 annual  (p=12)',        dict(year=2018, month=1,  period_months=12)),
    # Boundary year
    ('HLS 2013-06 (first HLS year)',   dict(year=2013, month=6,  period_months=1)),
    ('LS57 2012-06 (last LS year)',    dict(year=2012, month=6,  period_months=1)),
]

print(f"Location: lat={lat}, lon={lon}\n")
print(f"{'Test':<38} {'Size':>5} {'HTTP':>5} {'shape':>14}  Detail")
print("-" * 78)

for label, kwargs in tests:
    try:
        n, resp = get_chip(lat, lon, **kwargs)
        if resp.status_code == 200:
            arr    = mpimg.imread(BytesIO(resp.content), format='png')
            detail = str(arr.shape)
        else:
            detail = resp.text[:80]
        print(f"{label:<38} {n:>5} {resp.status_code:>5} {detail:>14}")
    except Exception as e:
        print(f"{label:<38} {'?':>5} {'ERR':>5} {'?':>14}  {type(e).__name__}: {e}")
    sys.stdout.flush()

print("\nDone.")
