// =============================================================================
// Naturalness Ranker — GEE App
// Click-to-rank 5 sites by perceived naturalness.
// Sites are displayed as Sentinel-2 thumbnails; a map shows ranked squares.
// =============================================================================

// ── Configuration ────────────────────────────────────────────────────────────

// Real examples from degree-of-recovery analysis (recovery_scores + ref_samples).
// Each has 1 abandoned-ag test site + 10 nearest natural reference sites (BallTree/Haversine).
var EXAMPLES = {
  'Low Recovery (Eco 41, score=42.6)': {
    batch_id: 'eco41_low_recovery',
    sites: [
      { id: 0,  name: 'Abandoned Ag (test)',  lon: 27.342787, lat: -32.651558 },
      { id: 1,  name: 'Natural Ref 1',        lon: 27.355901, lat: -32.641229 },
      { id: 2,  name: 'Natural Ref 2',        lon: 27.325179, lat: -32.646439 },
      { id: 3,  name: 'Natural Ref 3',        lon: 27.324550, lat: -32.646978 },
      { id: 4,  name: 'Natural Ref 4',        lon: 27.321406, lat: -32.646888 },
      { id: 5,  name: 'Natural Ref 5',        lon: 27.327335, lat: -32.635480 },
      { id: 6,  name: 'Natural Ref 6',        lon: 27.319250, lat: -32.643744 },
      { id: 7,  name: 'Natural Ref 7',        lon: 27.314938, lat: -32.645721 },
      { id: 8,  name: 'Natural Ref 8',        lon: 27.366232, lat: -32.634492 },
      { id: 9,  name: 'Natural Ref 9',        lon: 27.366052, lat: -32.670784 },
      { id: 10, name: 'Natural Ref 10',       lon: 27.312423, lat: -32.638624 }
    ]
  },
  'Medium Recovery (Eco 40, score=48.8)': {
    batch_id: 'eco40_medium_recovery',
    sites: [
      { id: 0,  name: 'Abandoned Ag (test)',  lon: 28.625042, lat: -32.247047 },
      { id: 1,  name: 'Natural Ref 1',        lon: 28.623244, lat: -32.240401 },
      { id: 2,  name: 'Natural Ref 2',        lon: 28.636719, lat: -32.237886 },
      { id: 3,  name: 'Natural Ref 3',        lon: 28.612644, lat: -32.237167 },
      { id: 4,  name: 'Natural Ref 4',        lon: 28.612375, lat: -32.231148 },
      { id: 5,  name: 'Natural Ref 5',        lon: 28.599259, lat: -32.247318 },
      { id: 6,  name: 'Natural Ref 6',        lon: 28.610488, lat: -32.223602 },
      { id: 7,  name: 'Natural Ref 7',        lon: 28.606446, lat: -32.225579 },
      { id: 8,  name: 'Natural Ref 8',        lon: 28.607704, lat: -32.223423 },
      { id: 9,  name: 'Natural Ref 9',        lon: 28.607434, lat: -32.221985 },
      { id: 10, name: 'Natural Ref 10',       lon: 28.594409, lat: -32.232765 }
    ]
  },
  'High Recovery (Eco 38, score=63.1)': {
    batch_id: 'eco38_high_recovery',
    sites: [
      { id: 0,  name: 'Abandoned Ag (test)',  lon: 28.431814, lat: -25.656828 },
      { id: 1,  name: 'Natural Ref 1',        lon: 28.474933, lat: -25.656199 },
      { id: 2,  name: 'Natural Ref 2',        lon: 28.464961, lat: -25.630417 },
      { id: 3,  name: 'Natural Ref 3',        lon: 28.476460, lat: -25.657546 },
      { id: 4,  name: 'Natural Ref 4',        lon: 28.479334, lat: -25.652516 },
      { id: 5,  name: 'Natural Ref 5',        lon: 28.484006, lat: -25.658175 },
      { id: 6,  name: 'Natural Ref 6',        lon: 28.443312, lat: -25.605893 },
      { id: 7,  name: 'Natural Ref 7',        lon: 28.489845, lat: -25.651707 },
      { id: 8,  name: 'Natural Ref 8',        lon: 28.436575, lat: -25.602659 },
      { id: 9,  name: 'Natural Ref 9',        lon: 28.423729, lat: -25.601222 },
      { id: 10, name: 'Natural Ref 10',       lon: 28.449870, lat: -25.601671 }
    ]
  }
};

// Active example — set from the dropdown
var FIRST_KEY = Object.keys(EXAMPLES)[0];
var SITES = EXAMPLES[FIRST_KEY].sites;
var BATCH_ID = EXAMPLES[FIRST_KEY].batch_id;

var BUFFER_M = 2000; // metres around each point for thumbnail & map square
var THUMB_DIM = '300'; // thumbnail pixel dimension
var START_DATE = '2023-01-01';
var END_DATE = '2024-01-01';
var CS_THRESHOLD = 0.6;

// Rank colours: index 0 = rank 1 (most natural, green) … index 10 = rank 11 (least, red)
var RANK_COLORS = [
  '00FF00', '33EE00', '66DD00', '99CC00', 'BBAA00',
  'DDAA00', 'FF9900', 'FF7700', 'FF5500', 'FF2200', 'FF0000'
];
var UNRANKED_COLOR = '888888';

var VIS_OPTIONS = {
  'True Color (B4/B3/B2)': { bands: ['B4', 'B3', 'B2'], min: 0, max: 3000 },
  'False Color (B11/B8/B3)': { bands: ['B11', 'B8', 'B3'], min: 0, max: 3000 }
};

// ── Google Sheets + Looker Studio integration ────────────────────────────────
// Setup:
// 1. Create a Google Sheet
// 2. Go to Extensions > Apps Script
// 3. Paste the contents of scripts/apps/sheets_backend.gs (replace Code.gs)
// 4. Run setupSheets() once from the Apps Script editor to create tabs
// 5. Deploy > New deployment > Web app > Execute as: Me, Who has access: Anyone
// 6. Copy the web app URL into SHEETS_WEBAPP_URL below
// 7. Set up a time-driven trigger: updateDashboard() every 5 min (optional,
//    dashboard also updates on each save)
//
// Looker Studio dashboard:
// 1. Go to lookerstudio.google.com > Create > Report
// 2. Add data source > Google Sheets > select your sheet
// 3. Add each tab as a separate data source (responses, leaderboard,
//    site_stats, agreement, progress)
// 4. Build pages:
//    - Progress:    time series chart from "progress" tab (x=date, y=cumulative)
//    - Leaderboard: table from "leaderboard" tab, sorted by submissions desc
//    - Agreement:   scorecard for kendalls_W, table for pairwise Spearman rho
//    - Spatial:     table/heatmap from "site_stats" (mean_rank, consensus_pct)
// 5. Copy the published Looker Studio URL into DASHBOARD_URL below

var SHEETS_WEBAPP_URL = 'https://script.google.com/macros/s/AKfycbxP4SgAjDOBYEzP5knuJoarFTVXmog1LP809J15c19ZghJSayNSdS_8u9S_JSApwhoH/exec'; // paste your Apps Script web app URL here
var DASHBOARD_URL = 'https://lookerstudio.google.com/reporting/e5b41f4c-9d91-451a-a74d-3831d6573684'; // paste your Looker Studio published URL here

// ── External viewer URLs ─────────────────────────────────────────────────────

function getEsriWaybackUrl(lat, lon) {
  // Esri World Imagery Wayback — browse historical satellite imagery
  return 'https://livingatlas.arcgis.com/wayback/#mapCenter='
    + lon + ',' + lat + ',16';
}

function getGoogleEarthUrl(lat, lon) {
  // Google Earth Web — 3D view at site location
  return 'https://earth.google.com/web/@' + lat + ',' + lon + ',0a,3000d,35y,0h,0t,0r';
}

// ── State ────────────────────────────────────────────────────────────────────

var rankings = [];       // ordered list of site indices (first = rank 1)
var nextRank = 1;        // next rank to assign (1–N)
var selectButtons = [];  // references to each site's Select button
var zoomButtons = [];    // references to each site's Zoom button
var rankLabels = [];     // references to each site's rank badge label
var thumbnailPanels = []; // references to each site's thumbnail panel
var sessionId = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
var labelerName = '';    // set by the labeler name text input
var appStarted = false;  // gates ranking until labeler enters their name

// ── Imagery helpers ──────────────────────────────────────────────────────────

// Compute a single S2 composite covering all sites in the active batch.
// Reused by all thumbnails — avoids separate collection filters per site.
var cachedComposite = null;
var cachedCompositeBatch = null;

function getBatchComposite() {
  if (cachedComposite && cachedCompositeBatch === BATCH_ID) {
    return cachedComposite;
  }
  // Bounding box covering all sites with 0.5° padding (~55 km) to ensure
  // full S2 tile coverage even for widely-spaced sites
  var lons = SITES.map(function (s) { return s.lon; });
  var lats = SITES.map(function (s) { return s.lat; });
  var bbox = ee.Geometry.Rectangle([
    Math.min.apply(null, lons) - 0.5,
    Math.min.apply(null, lats) - 0.5,
    Math.max.apply(null, lons) + 0.5,
    Math.max.apply(null, lats) + 0.5
  ]);

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(bbox)
    .filterDate(START_DATE, END_DATE);

  var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    .filterBounds(bbox)
    .filterDate(START_DATE, END_DATE);

  cachedComposite = s2.linkCollection(csPlus, ['cs'])
    .map(function (img) {
      return img.updateMask(img.select('cs').gte(CS_THRESHOLD));
    })
    .median();
  cachedCompositeBatch = BATCH_ID;
  return cachedComposite;
}

function getVisParams() {
  return VIS_OPTIONS[visSelect.getValue()];
}

// Build a square bounding box (same extent as thumbnail) for a site
function getSiteSquare(site) {
  var point = ee.Geometry.Point([site.lon, site.lat]);
  return point.buffer(BUFFER_M).bounds();
}

// ── Map helpers ──────────────────────────────────────────────────────────────

var mapPanel = ui.Map();
mapPanel.setOptions('SATELLITE');
mapPanel.setControlVisibility({ layerList: true, fullscreenControl: true, zoomControl: true });

// ── Natural mask reference layers (vis off by default) ──────────────────────
// These are the 4 datasets combined to define "natural" reference sites:
//   NATURAL = (SBTN == 1 OR NaturalForest_B0 >= 0.52) AND GHM <= 0.1 AND BII_All >= 0.7

var sbtnNatural = ee.Image('WRI/SBTN/naturalLands/v1_1/2020').select('natural');
mapPanel.addLayer(sbtnNatural.selfMask(),
  { min: 0, max: 1, palette: ['#2ecc71'] },
  'SBTN Natural Lands (2020)', false);

var ghm = ee.ImageCollection('projects/sat-io/open-datasets/GHM/HM_2022_90M')
  .map(function (img) { return img.select(0).rename('HM'); })
  .mosaic();
mapPanel.addLayer(ghm,
  { min: 0, max: 1, palette: ['#2ecc71', '#f1c40f', '#e74c3c'] },
  'Human Modification (GHM 2022)', false);
mapPanel.addLayer(ghm.lte(0.1).selfMask(),
  { min: 0, max: 1, palette: ['#27ae60'] },
  'GHM ≤ 0.1 (low modification)', false);

var biiMask = ee.Image('projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_Mask');
var bii1kmBands = ee.List([
  'Land Use', 'Land Use Intensity', 'BII All',
  'BII Amphibians', 'BII Birds', 'BII Forbs', 'BII Graminoids',
  'BII Mammals', 'BII All Plants', 'BII Reptiles', 'BII Trees',
  'BII All Vertebrates']);
var bii = ee.ImageCollection('projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_1km')
  .toBands().rename(bii1kmBands)
  .select('BII All').updateMask(biiMask);
mapPanel.addLayer(bii,
  { min: 0, max: 1, palette: ['#e74c3c', '#f39c12', '#2ecc71'] },
  'Biodiversity Intactness (BII)', false);
mapPanel.addLayer(bii.gte(0.7).selfMask(),
  { min: 0, max: 1, palette: ['#1abc9c'] },
  'BII ≥ 0.7 (high intactness)', false);

var naturalForest = ee.ImageCollection(
  'projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection'
).mosaic().select('B0');
mapPanel.addLayer(naturalForest,
  { min: 0, max: 1, palette: ['#f5f5f5', '#196f3d'] },
  'Natural Forest Probability', false);

// Combined natural mask (final product)
var combinedNatural = sbtnNatural.eq(1).or(naturalForest.gte(0.52))
  .and(ghm.lte(0.1))
  .and(bii.gte(0.7));
mapPanel.addLayer(combinedNatural.selfMask(),
  { min: 0, max: 1, palette: ['#00b894'] },
  'Combined Natural Mask', false);

// Track number of reference layers so updateMap can preserve them
var NUM_REF_LAYERS = mapPanel.layers().length();

function updateMap() {
  // Remove only the ranking overlay layers (keep reference layers at indices 0..NUM_REF_LAYERS-1)
  var layers = mapPanel.layers();
  while (layers.length() > NUM_REF_LAYERS) {
    layers.remove(layers.get(layers.length() - 1));
  }

  for (var i = 0; i < SITES.length; i++) {
    var site = SITES[i];
    var square = getSiteSquare(site);
    var rankIdx = rankings.indexOf(i);
    var color = rankIdx >= 0 ? RANK_COLORS[rankIdx] : UNRANKED_COLOR;
    var label = rankIdx >= 0 ? (site.name + ' (Rank ' + (rankIdx + 1) + ')') : site.name;

    // Draw square outline matching the thumbnail extent (no fill)
    var outline = ee.Image().byte().paint(
      ee.FeatureCollection([ee.Feature(square)]), 1, 3
    );
    mapPanel.addLayer(outline.visualize({ palette: [color] }), {}, label);
  }
}

function zoomToSites() {
  var coords = SITES.map(function (s) { return [s.lon, s.lat]; });
  var fc = ee.FeatureCollection(coords.map(function (c) {
    return ee.Feature(ee.Geometry.Point(c));
  }));
  mapPanel.centerObject(fc, 8);
}

function zoomToSite(site) {
  var square = getSiteSquare(site);
  mapPanel.centerObject(square, 14);
}

// ── Thumbnail builder ────────────────────────────────────────────────────────

var LINK_STYLE = { fontSize: '10px', margin: '0 4px', color: '#1a73e8' };

function buildThumbnailCard(site) {
  var point = ee.Geometry.Point([site.lon, site.lat]);
  var region = point.buffer(BUFFER_M);
  var composite = getBatchComposite();

  var visParams = getVisParams();
  var thumb = ui.Thumbnail({
    image: composite.visualize(visParams),
    params: { region: region, dimensions: THUMB_DIM, crs: 'EPSG:3857', format: 'PNG' }
  });

  var nameLabel = ui.Label({
    value: site.name,
    style: { fontWeight: 'bold', fontSize: '14px', margin: '4px 0 2px 0', textAlign: 'center' }
  });

  var coordLabel = ui.Label({
    value: site.lat.toFixed(2) + ', ' + site.lon.toFixed(2),
    style: { fontSize: '11px', color: '#666', margin: '0 0 2px 0', textAlign: 'center' }
  });

  // External viewer links
  var esriLink = ui.Label({
    value: 'Esri Wayback',
    style: LINK_STYLE,
    targetUrl: getEsriWaybackUrl(site.lat, site.lon)
  });

  var googleEarthLink = ui.Label({
    value: 'Google Earth',
    style: LINK_STYLE,
    targetUrl: getGoogleEarthUrl(site.lat, site.lon)
  });

  var linksRow = ui.Panel({
    widgets: [esriLink, googleEarthLink],
    layout: ui.Panel.Layout.flow('horizontal'),
    style: { stretch: 'horizontal', margin: '0', padding: '0' }
  });

  var rankBadge = ui.Label({
    value: '',
    style: {
      fontSize: '16px', fontWeight: 'bold', color: '#fff',
      backgroundColor: '#00000000', textAlign: 'center',
      margin: '0', padding: '2px 6px', shown: false
    }
  });

  var selectBtn = ui.Button({
    label: 'Select',
    style: { stretch: 'horizontal', margin: '2px 0 0 0' }
  });

  var zoomBtn = ui.Button({
    label: 'Zoom',
    style: { stretch: 'horizontal', margin: '2px 0 0 0' }
  });

  // Closures for click handlers
  (function (s, btn, badge) {
    btn.onClick(function () {
      handleSelect(s.id, btn, badge);
    });
  })(site, selectBtn, rankBadge);

  (function (s) {
    zoomBtn.onClick(function () {
      zoomToSite(s);
    });
  })(site);

  var buttonRow = ui.Panel({
    widgets: [selectBtn, zoomBtn],
    layout: ui.Panel.Layout.flow('horizontal'),
    style: { stretch: 'horizontal', margin: '0' }
  });

  var card = ui.Panel({
    widgets: [nameLabel, coordLabel, linksRow, thumb, rankBadge, buttonRow],
    style: {
      width: '190px', margin: '4px',
      border: '1px solid #ccc', padding: '4px'
    }
  });

  selectButtons.push(selectBtn);
  zoomButtons.push(zoomBtn);
  rankLabels.push(rankBadge);
  thumbnailPanels.push(card);

  return card;
}

// ── Event handlers ───────────────────────────────────────────────────────────

function handleSelect(siteIdx, btn, badge) {
  if (nextRank > SITES.length) return;
  if (!appStarted) {
    statusMessage.setValue('Enter your name in the left panel and click "Start ranking" first.');
    statusMessage.style().set('color', '#d93025');
    progressLabel.setValue('Please enter your name and click "Start" first.');
    progressLabel.style().set('color', '#d93025');
    return;
  }

  rankings.push(siteIdx);
  var rank = nextRank;
  nextRank++;

  // Update button and badge
  btn.setDisabled(true);
  btn.setLabel('Rank ' + rank);
  badge.setValue('# ' + rank);
  badge.style().set('shown', true);
  badge.style().set('backgroundColor', '#' + RANK_COLORS[rank - 1]);

  // Update progress
  if (nextRank <= SITES.length) {
    var msg = 'Click the site you rank #' + nextRank + ' of ' + SITES.length;
    progressLabel.setValue(msg);
    statusMessage.setValue(msg);
    statusMessage.style().set('color', '#1a73e8');
  } else {
    progressLabel.setValue('All ' + SITES.length + ' sites ranked!');
    statusMessage.setValue('All ' + SITES.length + ' ranked! Click "Save to Google Sheets" to save.');
    statusMessage.style().set('color', '#0d904f');
    statusSaveButton.style().set('shown', true);
    showResults();
  }

  // Update ranked list
  updateRankedList();

  // Update map markers
  updateMap();
}

function handleUndo() {
  if (rankings.length === 0) return;

  var lastIdx = rankings.pop();
  nextRank--;

  // Re-enable button and hide badge
  selectButtons[lastIdx].setDisabled(false);
  selectButtons[lastIdx].setLabel('Select');
  rankLabels[lastIdx].style().set('shown', false);

  // Hide results if they were shown
  resultsPanel.style().set('shown', false);
  exportButton.style().set('shown', false);
  statusSaveButton.style().set('shown', false);
  statusConfirmPanel.style().set('shown', false);
  saveStatus.style().set('shown', false);

  var msg = 'Click the site you rank #' + nextRank + ' of ' + SITES.length;
  progressLabel.setValue(msg);
  statusMessage.setValue(msg);
  statusMessage.style().set('color', '#1a73e8');
  updateRankedList();
  updateMap();
}

function handleReset() {
  rankings = [];
  nextRank = 1;
  sessionId = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);

  for (var i = 0; i < SITES.length; i++) {
    selectButtons[i].setDisabled(false);
    selectButtons[i].setLabel('Select');
    rankLabels[i].style().set('shown', false);
  }

  resultsPanel.style().set('shown', false);
  exportButton.style().set('shown', false);
  statusSaveButton.style().set('shown', false);
  statusConfirmPanel.style().set('shown', false);
  saveStatus.style().set('shown', false);
  progressLabel.setValue('Click the site you rank #1 of ' + SITES.length + ' (most natural)');
  progressLabel.style().set('color', '#1a73e8');
  statusMessage.setValue('Click "Select" on the most natural site first (Rank #1 of ' + SITES.length + ')');
  statusMessage.style().set('color', '#1a73e8');
  rankedListPanel.clear();
  updateMap();
}

function updateRankedList() {
  rankedListPanel.clear();
  for (var i = 0; i < rankings.length; i++) {
    var site = SITES[rankings[i]];
    rankedListPanel.add(ui.Label({
      value: (i + 1) + '. ' + site.name + ' (' + site.lat.toFixed(2) + ', ' + site.lon.toFixed(2) + ')',
      style: { fontSize: '13px', margin: '2px 0', color: '#' + RANK_COLORS[i] }
    }));
  }
}

function showResults() {
  resultsPanel.clear();
  resultsPanel.add(ui.Label({
    value: 'Final Ranking',
    style: { fontWeight: 'bold', fontSize: '16px', margin: '8px 0 4px 0' }
  }));

  var resultDict = {};
  for (var i = 0; i < rankings.length; i++) {
    var site = SITES[rankings[i]];
    resultsPanel.add(ui.Label({
      value: 'Rank ' + (i + 1) + ': ' + site.name +
        ' [' + site.lat.toFixed(4) + ', ' + site.lon.toFixed(4) + ']',
      style: { fontSize: '13px', margin: '2px 8px', color: '#' + RANK_COLORS[i] }
    }));
    resultDict['rank_' + (i + 1)] = {
      name: site.name, lon: site.lon, lat: site.lat
    };
  }

  resultsPanel.style().set('shown', true);
  exportButton.style().set('shown', true);

  // Print to console
  print('=== Naturalness Ranking ===');
  print(resultDict);
}

function handleExport() {
  var features = rankings.map(function (siteIdx, i) {
    var site = SITES[siteIdx];
    return ee.Feature(ee.Geometry.Point([site.lon, site.lat]), {
      rank: i + 1,
      site_name: site.name,
      longitude: site.lon,
      latitude: site.lat,
      labeler: labelerName,
      batch_id: BATCH_ID,
      session_id: sessionId
    });
  });

  var fc = ee.FeatureCollection(features);
  Export.table.toDrive({
    collection: fc,
    description: 'naturalness_ranking_' + BATCH_ID + '_' + labelerName.replace(/\s+/g, '_'),
    fileFormat: 'CSV'
  });

  print('Export task submitted — check the Tasks tab.');
}

function handleSaveToSheets() {
  if (!SHEETS_WEBAPP_URL) {
    print('ERROR: Set SHEETS_WEBAPP_URL at the top of the script first.');
    return;
  }

  // Capture ranking state before auto-advance resets it
  var saveBatchId = BATCH_ID;
  var saveRankings = rankings.slice();
  var saveSites = SITES.slice();
  var saveSession = sessionId;

  var params = 'batch_id=' + encodeURIComponent(saveBatchId) +
    '&labeler=' + encodeURIComponent(labelerName) +
    '&session_id=' + encodeURIComponent(saveSession) +
    '&timestamp=' + encodeURIComponent(new Date().toISOString());
  for (var i = 0; i < saveRankings.length; i++) {
    var site = saveSites[saveRankings[i]];
    params += '&rank' + (i + 1) + '=' + encodeURIComponent(
      site.name + '|' + site.lat + '|' + site.lon
    );
  }
  var fullUrl = SHEETS_WEBAPP_URL + '?' + params;

  // Show confirm link in the status bar (rebuilt with new targetUrl)
  statusSaveButton.style().set('shown', false);
  statusConfirmPanel.clear();
  statusConfirmPanel.add(ui.Label({
    value: 'CONFIRM SAVE: ' + saveBatchId + ' →',
    targetUrl: fullUrl,
    style: {
      fontSize: '14px', fontWeight: 'bold', color: '#fff',
      backgroundColor: '#343635ff', padding: '8px 16px', margin: '0',
      textAlign: 'center'
    }
  }));
  statusConfirmPanel.style().set('shown', true);
  statusMessage.setValue('Click the green link to confirm save.');
  statusMessage.style().set('color', '#333');

  // Auto-advance to the next example (thumbnails start loading immediately)
  var keys = Object.keys(EXAMPLES);
  var currentIdx = keys.indexOf(exampleSelect.getValue());
  if (currentIdx < keys.length - 1) {
    progressLabel.setValue('Loading next set...');
    progressLabel.style().set('color', '#0d904f');
    exampleSelect.setValue(keys[currentIdx + 1], true);
  } else {
    progressLabel.setValue('All examples done! Click confirm link above.');
    progressLabel.style().set('color', '#0d904f');
    resultsPanel.style().set('shown', false);
    exportButton.style().set('shown', false);
  }
}

function handleExampleChange(exampleName) {
  // Switch to a different site set
  var example = EXAMPLES[exampleName];
  SITES = example.sites;
  BATCH_ID = example.batch_id;
  batchLabel.setValue('Batch: ' + BATCH_ID);

  // Full reset of ranking state
  rankings = [];
  nextRank = 1;
  sessionId = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);

  resultsPanel.style().set('shown', false);
  exportButton.style().set('shown', false);
  statusSaveButton.style().set('shown', false);
  saveStatus.style().set('shown', false);
  rankedListPanel.clear();

  if (appStarted) {
    progressLabel.setValue('Click the site you rank #1 of ' + SITES.length + ' (most natural)');
    progressLabel.style().set('color', '#1a73e8');
    statusMessage.setValue('Click "Select" on the most natural site first (Rank #1 of ' + SITES.length + ')');
    statusMessage.style().set('color', '#1a73e8');
  }

  // Rebuild thumbnails for new sites
  thumbnailRow.clear();
  selectButtons = [];
  zoomButtons = [];
  rankLabels = [];
  thumbnailPanels = [];
  for (var i = 0; i < SITES.length; i++) {
    thumbnailRow.add(buildThumbnailCard(SITES[i]));
  }

  updateMap();
  zoomToSites();
}

function handleVisChange() {
  // Rebuild thumbnails with new vis params
  thumbnailRow.clear();
  selectButtons = [];
  zoomButtons = [];
  rankLabels = [];
  thumbnailPanels = [];

  for (var i = 0; i < SITES.length; i++) {
    var card = buildThumbnailCard(SITES[i]);
    thumbnailRow.add(card);
  }

  // Restore rank state on rebuilt buttons
  for (var r = 0; r < rankings.length; r++) {
    var idx = rankings[r];
    selectButtons[idx].setDisabled(true);
    selectButtons[idx].setLabel('Rank ' + (r + 1));
    rankLabels[idx].setValue('# ' + (r + 1));
    rankLabels[idx].style().set('shown', true);
    rankLabels[idx].style().set('backgroundColor', '#' + RANK_COLORS[r]);
  }
}

// ── UI Elements ──────────────────────────────────────────────────────────────

// Left control panel
var titleLabel = ui.Label({
  value: 'Naturalness Ranker',
  style: { fontSize: '22px', fontWeight: 'bold', margin: '8px', color: '#333' }
});

var instrLabel = ui.Label({
  value: 'Rank all sites by perceived naturalness (1 abandoned-ag test site + 10 natural references). Click "Select" on the most natural site first, then the next, and so on. Use "Zoom" to inspect on the map, or the Esri Wayback / Google Earth links for external views.',
  style: { fontSize: '13px', margin: '4px 8px', color: '#555' }
});

// Labeler identification
var labelerLabel = ui.Label({
  value: 'Your name:',
  style: { fontSize: '12px', fontWeight: 'bold', margin: '8px 8px 0 8px' }
});

var labelerInput = ui.Textbox({
  placeholder: 'Enter your name...',
  style: { stretch: 'horizontal', margin: '2px 8px' }
});

var startButton = ui.Button({
  label: 'Start ranking',
  style: { stretch: 'horizontal', margin: '2px 8px' },
  onClick: function () {
    var name = labelerInput.getValue().trim();
    if (!name) {
      statusMessage.setValue('Please enter your name in the left panel first.');
      statusMessage.style().set('color', '#d93025');
      progressLabel.setValue('Please enter your name first.');
      progressLabel.style().set('color', '#d93025');
      return;
    }
    labelerName = name;
    appStarted = true;
    labelerInput.setDisabled(true);
    startButton.setDisabled(true);
    startButton.setLabel('Ranking as: ' + labelerName);
    statusMessage.setValue('Click "Select" on the most natural site first (Rank #1 of ' + SITES.length + ')');
    statusMessage.style().set('color', '#1a73e8');
    progressLabel.setValue('Click the site you rank #1 of ' + SITES.length + ' (most natural)');
    progressLabel.style().set('color', '#1a73e8');
  }
});

var batchLabel = ui.Label({
  value: 'Batch: ' + BATCH_ID,
  style: { fontSize: '11px', color: '#888', margin: '0 8px 4px 8px' }
});

var progressLabel = ui.Label({
  value: 'Enter your name and click "Start ranking"',
  style: { fontSize: '14px', fontWeight: 'bold', margin: '8px', color: '#1a73e8' }
});

var visSelect = ui.Select({
  items: Object.keys(VIS_OPTIONS),
  value: 'True Color (B4/B3/B2)',
  style: { stretch: 'horizontal', margin: '4px 8px' },
  onChange: handleVisChange
});

var exampleLabel = ui.Label({ value: 'Site set:', style: { fontSize: '12px', margin: '8px 8px 0 8px' } });

var exampleSelect = ui.Select({
  items: Object.keys(EXAMPLES),
  value: FIRST_KEY,
  style: { stretch: 'horizontal', margin: '2px 8px' },
  onChange: handleExampleChange
});

var visLabel = ui.Label({ value: 'Imagery:', style: { fontSize: '12px', margin: '8px 8px 0 8px' } });

var rankedListPanel = ui.Panel({ style: { margin: '4px 8px' } });

var undoButton = ui.Button({
  label: 'Undo Last',
  style: { stretch: 'horizontal', margin: '4px 8px' },
  onClick: handleUndo
});

var resetButton = ui.Button({
  label: 'Reset All',
  style: { stretch: 'horizontal', margin: '4px 8px' },
  onClick: handleReset
});

var resultsPanel = ui.Panel({ style: { shown: false, margin: '4px 8px' } });

var exportButton = ui.Button({
  label: 'Export to Drive (CSV)',
  style: { stretch: 'horizontal', margin: '4px 8px', shown: false },
  onClick: handleExport
});

var saveStatus = ui.Label({
  value: '',
  style: { fontSize: '12px', color: '#666', margin: '2px 8px', shown: false }
});

// ── Status bar (above thumbnails) — instructions, errors, save actions ──────
var statusMessage = ui.Label({
  value: 'Enter your name in the left panel and click "Start ranking" to begin.',
  style: {
    fontSize: '14px', fontWeight: 'bold', color: '#1a73e8',
    margin: '0', padding: '0', stretch: 'horizontal'
  }
});

var statusSaveButton = ui.Button({
  label: 'Save to Google Sheets',
  style: { shown: false, margin: '0 8px', fontWeight: 'bold' },
  onClick: handleSaveToSheets
});

// Container for the confirm-save link (rebuilt each time with a new targetUrl)
var statusConfirmPanel = ui.Panel({
  style: { shown: false, margin: '0' }
});

var statusBar = ui.Panel({
  widgets: [statusMessage, statusSaveButton, statusConfirmPanel],
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    backgroundColor: '#f0f4f8', padding: '8px 12px',
    stretch: 'horizontal', border: '0 0 1px 0 solid #ddd'
  }
});

// Dashboard link (only shown if DASHBOARD_URL is set)
var dashboardLink = ui.Label({
  value: 'View live dashboard',
  style: {
    fontSize: '13px', color: '#1a73e8', margin: '8px',
    fontWeight: 'bold', shown: !!DASHBOARD_URL
  },
  targetUrl: DASHBOARD_URL || undefined
});

var controlPanel = ui.Panel({
  widgets: [
    titleLabel, instrLabel,
    labelerLabel, labelerInput, startButton, batchLabel,
    progressLabel,
    exampleLabel, exampleSelect,
    visLabel, visSelect,
    ui.Label({ value: 'Current ranking:', style: { fontSize: '13px', fontWeight: 'bold', margin: '8px 8px 0 8px' } }),
    rankedListPanel,
    undoButton, resetButton,
    resultsPanel, exportButton, saveStatus,
    dashboardLink
  ],
  style: { width: '320px', padding: '0' }
});

// Thumbnail row
var thumbnailRow = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal', true),
  style: { backgroundColor: '#f5f5f5', padding: '8px' }
});

// Build initial thumbnail cards
for (var i = 0; i < SITES.length; i++) {
  thumbnailRow.add(buildThumbnailCard(SITES[i]));
}

// Top area: status bar + thumbnails
var topPanel = ui.Panel({
  widgets: [statusBar, thumbnailRow],
  style: { stretch: 'both' }
});

// Right panel: top area on top, map on bottom
var rightPanel = ui.Panel(ui.SplitPanel({
  firstPanel: topPanel,
  secondPanel: mapPanel,
  orientation: 'vertical',
  wipe: false
}));

// ── App init ─────────────────────────────────────────────────────────────────

ui.root.clear();
ui.root.add(ui.SplitPanel(controlPanel, rightPanel));

updateMap();
zoomToSites();
