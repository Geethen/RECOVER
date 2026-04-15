// =============================================================================
// Naturalness Ranker — Google Apps Script Backend
// Paste this entire file into your Google Sheet's Apps Script editor
// (Extensions > Apps Script > replace Code.gs contents)
//
// After pasting:
//   1. Deploy > New deployment > Web app
//      - Execute as: Me
//      - Who has access: Anyone
//   2. Copy the web app URL into SHEETS_WEBAPP_URL in the GEE script
//   3. Run setupSheets() once from the Apps Script editor to create tabs
//   4. Set up a time-driven trigger for updateDashboard() (every 5 min)
//      - Edit > Triggers > Add trigger > updateDashboard > Time-driven > Minutes > 5
//
// Sheet tabs created by setupSheets():
//   "responses"   — raw ranking submissions (written by doGet)
//   "leaderboard" — submissions per labeler (computed by updateDashboard)
//   "site_stats"  — per-site mean rank, std dev, consensus (computed)
//   "agreement"   — pairwise + overall inter-rater agreement (computed)
//   "progress"    — cumulative submissions over time (computed)
// =============================================================================

// ── Web endpoint ─────────────────────────────────────────────────────────────

function doGet(e) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('responses') || ss.getActiveSheet();
  var params = e.parameter;

  // Deduplication: same batch + labeler + session = already saved
  var data = sheet.getDataRange().getValues();
  for (var d = 1; d < data.length; d++) {
    if (String(data[d][1]) === String(params.batch_id) &&
        String(data[d][2]) === String(params.labeler) &&
        String(data[d][3]) === String(params.session_id)) {
      return buildResponsePage_('already saved', params, true);
    }
  }

  // Build flat row: timestamp, batch_id, labeler, session_id, client_timestamp,
  //                 rank1, site1, lat1, lon1, ..., rankN, siteN, latN, lonN
  var row = [new Date(), params.batch_id, params.labeler,
             params.session_id, params.timestamp];
  for (var i = 1; i <= 11; i++) {
    var val = params['rank' + i];
    if (val) {
      var parts = val.split('|');
      row.push(i, parts[0], parseFloat(parts[1]), parseFloat(parts[2]));
    }
  }
  sheet.appendRow(row);
  SpreadsheetApp.flush(); // commit row before dashboard reads it

  // Trigger dashboard update after each submission
  updateDashboard();

  return buildResponsePage_('saved', params, false);
}

// Build a styled HTML confirmation page so the user sees feedback in the new tab
function buildResponsePage_(status, params, isDuplicate) {
  var labeler = params.labeler || '(unknown)';
  var batch = params.batch_id || '';
  var rankSummary = '';
  for (var i = 1; i <= 11; i++) {
    var val = params['rank' + i];
    if (val) {
      var parts = val.split('|');
      rankSummary += '<tr><td style="padding:2px 12px">' + i + '</td>'
        + '<td style="padding:2px 12px">' + parts[0] + '</td></tr>';
    }
  }

  var title = isDuplicate ? 'Already Saved' : 'Ranking Saved!';
  var color = isDuplicate ? '#f57c00' : '#0d904f';
  var icon = isDuplicate ? '⚠️' : '✅';

  var html = '<!DOCTYPE html><html><head><meta charset="utf-8">'
    + '<title>' + title + '</title>'
    + '<style>'
    + 'body{font-family:Google Sans,Roboto,Arial,sans-serif;display:flex;'
    + 'justify-content:center;align-items:center;min-height:100vh;margin:0;'
    + 'background:#f8f9fa;color:#333}'
    + '.card{background:#fff;border-radius:12px;padding:32px 40px;'
    + 'box-shadow:0 2px 8px rgba(0,0,0,.12);text-align:center;max-width:400px}'
    + 'h1{color:' + color + ';font-size:22px;margin:8px 0}'
    + '.meta{color:#666;font-size:13px;margin:4px 0}'
    + 'table{margin:12px auto;border-collapse:collapse;font-size:14px}'
    + '.hint{color:#999;font-size:12px;margin-top:16px}'
    + '</style></head><body>'
    + '<div class="card">'
    + '<div style="font-size:48px">' + icon + '</div>'
    + '<h1>' + title + '</h1>'
    + '<p class="meta">Labeler: <b>' + labeler + '</b></p>'
    + '<p class="meta">Batch: ' + batch + '</p>';

  if (rankSummary) {
    html += '<table><tr><th>Rank</th><th>Site</th></tr>' + rankSummary + '</table>';
  }

  html += '<p class="hint" id="hint">Closing this tab automatically...</p>'
    + '<button onclick="window.close();google.script.host.close();" '
    + 'style="margin-top:12px;padding:8px 24px;font-size:14px;border:none;'
    + 'border-radius:6px;background:' + color + ';color:#fff;cursor:pointer">'
    + 'Close tab</button>'
    + '<script>'
    + 'function tryClose(){'
    + '  try{window.close()}catch(e){}'
    + '  try{google.script.host.close()}catch(e){}'
    + '  try{window.open("","_self").close()}catch(e){}'
    + '}'
    + 'setTimeout(tryClose,1500);'
    + 'setTimeout(tryClose,3000);'
    + 'setTimeout(function(){'
    + '  document.getElementById("hint").textContent="Saved! You can close this tab.";'
    + '},4000);'
    + '</script>'
    + '</div></body></html>';

  return HtmlService.createHtmlOutput(html)
    .setTitle(title)
    .setSandboxMode(HtmlService.SandboxMode.IFRAME)
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

// ── One-time setup ───────────────────────────────────────────────────────────

function setupSheets() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();

  // Responses tab (raw data)
  var resp = ss.getSheetByName('responses') || ss.insertSheet('responses');
  if (resp.getLastRow() === 0) {
    var headers = ['server_timestamp', 'batch_id', 'labeler', 'session_id', 'client_timestamp'];
    for (var i = 1; i <= 11; i++) {
      headers.push('rank' + i, 'site' + i + '_name', 'site' + i + '_lat', 'site' + i + '_lon');
    }
    resp.appendRow(headers);
    resp.getRange(1, 1, 1, resp.getLastColumn()).setFontWeight('bold');
  }

  // Leaderboard tab
  var lb = ss.getSheetByName('leaderboard') || ss.insertSheet('leaderboard');
  lb.clear();
  lb.appendRow(['labeler', 'batch_id', 'submissions', 'first_submission', 'last_submission']);
  lb.getRange(1, 1, 1, 5).setFontWeight('bold');

  // Site stats tab
  var stats = ss.getSheetByName('site_stats') || ss.insertSheet('site_stats');
  stats.clear();
  var statsHeaders = ['batch_id', 'site_name', 'latitude', 'longitude',
    'mean_rank', 'std_dev', 'median_rank', 'n_rankings'];
  for (var i = 1; i <= 11; i++) { statsHeaders.push('rank' + i + '_count'); }
  statsHeaders.push('consensus_pct');
  stats.appendRow(statsHeaders);
  stats.getRange(1, 1, 1, statsHeaders.length).setFontWeight('bold');

  // Agreement tab
  var agree = ss.getSheetByName('agreement') || ss.insertSheet('agreement');
  agree.clear();
  agree.appendRow([
    'batch_id', 'metric', 'value', 'n_labelers', 'n_sites', 'interpretation'
  ]);
  agree.getRange(1, 1, 1, 6).setFontWeight('bold');

  // Progress tab
  var prog = ss.getSheetByName('progress') || ss.insertSheet('progress');
  prog.clear();
  prog.appendRow(['batch_id', 'date', 'cumulative_submissions', 'unique_labelers']);
  prog.getRange(1, 1, 1, 4).setFontWeight('bold');

  SpreadsheetApp.flush();
}

// ── Dashboard update ─────────────────────────────────────────────────────────

function updateDashboard() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var respSheet = ss.getSheetByName('responses');
  if (!respSheet || respSheet.getLastRow() <= 1) return;

  var data = respSheet.getDataRange().getValues();
  var headers = data[0];
  var rows = data.slice(1);

  if (rows.length === 0) return;

  // Group by batch_id
  var batches = {};
  rows.forEach(function(row) {
    var batchId = String(row[1]);
    if (!batches[batchId]) batches[batchId] = [];
    batches[batchId].push(row);
  });

  updateLeaderboard(ss, rows);
  updateSiteStats(ss, batches);
  updateAgreement(ss, batches);
  updateProgress(ss, batches);

  // Force-commit all pending writes so Looker Studio sees them immediately
  SpreadsheetApp.flush();
}

// ── Leaderboard ──────────────────────────────────────────────────────────────

function updateLeaderboard(ss, rows) {
  var sheet = ss.getSheetByName('leaderboard');
  sheet.getRange(2, 1, Math.max(sheet.getLastRow(), 1), 5).clearContent();

  var labelers = {};
  rows.forEach(function(row) {
    var key = row[2] + '||' + row[1]; // labeler||batch_id
    if (!labelers[key]) {
      labelers[key] = {
        labeler: row[2],
        batch_id: row[1],
        count: 0,
        first: row[0],
        last: row[0]
      };
    }
    labelers[key].count++;
    if (row[0] < labelers[key].first) labelers[key].first = row[0];
    if (row[0] > labelers[key].last) labelers[key].last = row[0];
  });

  var outRows = [];
  Object.keys(labelers).forEach(function(key) {
    var l = labelers[key];
    outRows.push([l.labeler, l.batch_id, l.count, l.first, l.last]);
  });

  // Sort by count descending
  outRows.sort(function(a, b) { return b[2] - a[2]; });

  if (outRows.length > 0) {
    sheet.getRange(2, 1, outRows.length, 5).setValues(outRows);
  }
}

// ── Site stats ───────────────────────────────────────────────────────────────

function updateSiteStats(ss, batches) {
  var sheet = ss.getSheetByName('site_stats');
  var NUM_RANKS = 11;
  var numCols = 8 + NUM_RANKS + 1; // 8 base cols + rank counts + consensus
  sheet.getRange(2, 1, Math.max(sheet.getLastRow(), 1), numCols).clearContent();

  var outRows = [];

  Object.keys(batches).forEach(function(batchId) {
    var batchRows = batches[batchId];

    // Collect all sites and their ranks across all labelers
    var sites = {}; // site_name -> {ranks: [], lat, lon}
    batchRows.forEach(function(row) {
      for (var i = 0; i < NUM_RANKS; i++) {
        var rankCol = 5 + i * 4;       // rank value column
        var nameCol = rankCol + 1;      // site name column
        var latCol = rankCol + 2;
        var lonCol = rankCol + 3;

        var rank = row[rankCol];
        var name = row[nameCol];
        if (!name || !rank) continue;

        if (!sites[name]) {
          sites[name] = { ranks: [], lat: row[latCol], lon: row[lonCol] };
        }
        sites[name].ranks.push(Number(rank));
      }
    });

    Object.keys(sites).forEach(function(siteName) {
      var s = sites[siteName];
      var n = s.ranks.length;
      var mean = s.ranks.reduce(function(a, b) { return a + b; }, 0) / n;

      var variance = s.ranks.reduce(function(sum, r) {
        return sum + (r - mean) * (r - mean);
      }, 0) / n;
      var stdDev = Math.sqrt(variance);

      var sorted = s.ranks.slice().sort(function(a, b) { return a - b; });
      var median = n % 2 === 0
        ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        : sorted[Math.floor(n / 2)];

      // Rank distribution
      var rankCounts = [];
      for (var rc = 0; rc < NUM_RANKS; rc++) { rankCounts.push(0); }
      s.ranks.forEach(function(r) { if (r >= 1 && r <= NUM_RANKS) rankCounts[r - 1]++; });

      // Consensus: % of labelers who assigned the most common rank
      var maxCount = Math.max.apply(null, rankCounts);
      var consensusPct = n > 0 ? Math.round(maxCount / n * 100) : 0;

      var outRow = [
        batchId, siteName, s.lat, s.lon,
        Math.round(mean * 100) / 100,
        Math.round(stdDev * 100) / 100,
        median, n
      ].concat(rankCounts).concat([consensusPct]);
      outRows.push(outRow);
    });
  });

  // Sort by batch, then mean rank
  outRows.sort(function(a, b) {
    if (a[0] !== b[0]) return a[0] < b[0] ? -1 : 1;
    return a[4] - b[4];
  });

  if (outRows.length > 0) {
    sheet.getRange(2, 1, outRows.length, numCols).setValues(outRows);
  }
}

// ── Agreement metrics ────────────────────────────────────────────────────────

function updateAgreement(ss, batches) {
  var sheet = ss.getSheetByName('agreement');
  sheet.getRange(2, 1, Math.max(sheet.getLastRow(), 1), 6).clearContent();

  var outRows = [];

  Object.keys(batches).forEach(function(batchId) {
    var batchRows = batches[batchId];

    // Build ranking matrix: labelers x sites
    // Each row = one labeler's ranking (site_name -> rank)
    var labelerRankings = {};
    var allSites = {};

    batchRows.forEach(function(row) {
      var labeler = String(row[2]);
      if (!labelerRankings[labeler]) labelerRankings[labeler] = {};

      for (var i = 0; i < 11; i++) {
        var rankCol = 5 + i * 4;
        var nameCol = rankCol + 1;
        var rank = row[rankCol];
        var name = row[nameCol];
        if (!name || !rank) continue;

        labelerRankings[labeler][name] = Number(rank);
        allSites[name] = true;
      }
    });

    var labelerNames = Object.keys(labelerRankings);
    var siteNames = Object.keys(allSites).sort();
    var k = labelerNames.length; // number of raters
    var n = siteNames.length;    // number of items

    if (k < 2 || n < 2) {
      outRows.push([batchId, 'kendalls_W', 'N/A', k, n,
        'Need at least 2 labelers and 2 sites']);
      return;
    }

    // Build matrix: k rows x n columns (ranks)
    var matrix = [];
    labelerNames.forEach(function(labeler) {
      var row = [];
      siteNames.forEach(function(site) {
        row.push(labelerRankings[labeler][site] || 0);
      });
      matrix.push(row);
    });

    // Kendall's W (coefficient of concordance)
    // W = 12 * S / (k^2 * (n^3 - n))
    // where S = sum of squared deviations of column rank sums from mean rank sum

    // Column sums (sum of ranks for each site across all labelers)
    var colSums = [];
    for (var j = 0; j < n; j++) {
      var sum = 0;
      for (var i = 0; i < k; i++) {
        sum += matrix[i][j];
      }
      colSums.push(sum);
    }

    var meanColSum = colSums.reduce(function(a, b) { return a + b; }, 0) / n;

    var S = colSums.reduce(function(sum, cs) {
      return sum + (cs - meanColSum) * (cs - meanColSum);
    }, 0);

    var W = (12 * S) / (k * k * (n * n * n - n));
    W = Math.round(W * 1000) / 1000;

    var interpretation;
    if (W >= 0.9) interpretation = 'Very strong agreement';
    else if (W >= 0.7) interpretation = 'Strong agreement';
    else if (W >= 0.5) interpretation = 'Moderate agreement';
    else if (W >= 0.3) interpretation = 'Weak agreement';
    else interpretation = 'Little to no agreement';

    outRows.push([batchId, 'kendalls_W', W, k, n, interpretation]);

    // Mean Spearman rho between all labeler pairs
    var rhoSum = 0;
    var pairCount = 0;
    for (var a = 0; a < k; a++) {
      for (var b = a + 1; b < k; b++) {
        var rho = spearmanRho(matrix[a], matrix[b]);
        rhoSum += rho;
        pairCount++;

        outRows.push([
          batchId,
          'spearman_' + labelerNames[a] + '_vs_' + labelerNames[b],
          Math.round(rho * 1000) / 1000,
          2, n,
          rho >= 0.8 ? 'Strong' : rho >= 0.5 ? 'Moderate' : 'Weak'
        ]);
      }
    }

    if (pairCount > 0) {
      var meanRho = Math.round((rhoSum / pairCount) * 1000) / 1000;
      outRows.push([batchId, 'mean_spearman_rho', meanRho, k, n,
        meanRho >= 0.8 ? 'Strong' : meanRho >= 0.5 ? 'Moderate' : 'Weak']);
    }
  });

  if (outRows.length > 0) {
    sheet.getRange(2, 1, outRows.length, 6).setValues(outRows);
  }
}

// Spearman rank correlation between two rank arrays
function spearmanRho(ranks1, ranks2) {
  var n = ranks1.length;
  if (n === 0) return 0;

  var dSquaredSum = 0;
  for (var i = 0; i < n; i++) {
    var d = ranks1[i] - ranks2[i];
    dSquaredSum += d * d;
  }

  return 1 - (6 * dSquaredSum) / (n * (n * n - 1));
}

// ── Progress ─────────────────────────────────────────────────────────────────

function updateProgress(ss, batches) {
  var sheet = ss.getSheetByName('progress');
  sheet.getRange(2, 1, Math.max(sheet.getLastRow(), 1), 4).clearContent();

  var outRows = [];

  Object.keys(batches).forEach(function(batchId) {
    var batchRows = batches[batchId];

    // Sort by timestamp
    batchRows.sort(function(a, b) { return new Date(a[0]) - new Date(b[0]); });

    // Group by date
    var dailyStats = {};
    var cumulativeCount = 0;
    var uniqueLabelers = {};

    batchRows.forEach(function(row) {
      var date = Utilities.formatDate(new Date(row[0]), 'UTC', 'yyyy-MM-dd');
      cumulativeCount++;
      uniqueLabelers[String(row[2])] = true;

      dailyStats[date] = {
        cumulative: cumulativeCount,
        uniqueLabelers: Object.keys(uniqueLabelers).length
      };
    });

    Object.keys(dailyStats).sort().forEach(function(date) {
      var s = dailyStats[date];
      outRows.push([batchId, date, s.cumulative, s.uniqueLabelers]);
    });
  });

  if (outRows.length > 0) {
    sheet.getRange(2, 1, outRows.length, 4).setValues(outRows);
  }
}

// ── Demo data ────────────────────────────────────────────────────────────────
// Run this once from the Apps Script editor to populate the sheet with
// 3 example labeler submissions. Useful for testing the dashboard.

function seedDemoData() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('responses');
  if (!sheet) {
    setupSheets();
    sheet = ss.getSheetByName('responses');
  }

  // 3 demo labelers with different ranking preferences
  var demoRows = [
    // Labeler 1: Alice — ranks Site C highest (grassland), Site D lowest
    [new Date('2026-03-20T09:15:00Z'), 'highveld_batch_01', 'Alice',
     'demo_alice_01', '2026-03-20T09:15:00.000Z',
     1, 'Site C', -26.85, 27.80,
     2, 'Site E', -26.30, 29.50,
     3, 'Site A', -26.50, 28.25,
     4, 'Site B', -27.20, 29.10,
     5, 'Site D', -27.60, 28.90],

    // Labeler 2: Bob — mostly agrees with Alice, swaps A and E
    [new Date('2026-03-21T14:30:00Z'), 'highveld_batch_01', 'Bob',
     'demo_bob_01', '2026-03-21T14:30:00.000Z',
     1, 'Site C', -26.85, 27.80,
     2, 'Site A', -26.50, 28.25,
     3, 'Site E', -26.30, 29.50,
     4, 'Site B', -27.20, 29.10,
     5, 'Site D', -27.60, 28.90],

    // Labeler 3: Carol — disagrees on top pick, agrees on bottom
    [new Date('2026-03-22T11:00:00Z'), 'highveld_batch_01', 'Carol',
     'demo_carol_01', '2026-03-22T11:00:00.000Z',
     1, 'Site E', -26.30, 29.50,
     2, 'Site C', -26.85, 27.80,
     3, 'Site B', -27.20, 29.10,
     4, 'Site A', -26.50, 28.25,
     5, 'Site D', -27.60, 28.90]
  ];

  demoRows.forEach(function(row) {
    sheet.appendRow(row);
  });

  // Update dashboard tabs with the demo data
  updateDashboard();

  SpreadsheetApp.getUi().alert(
    'Demo data seeded!\n\n' +
    '3 labelers added: Alice, Bob, Carol\n' +
    'Batch: highveld_batch_01\n\n' +
    'Check the leaderboard, site_stats, agreement, and progress tabs.'
  );
}
