import { state } from '../state.js';
import { escapeHtml, featureIntersectsPolygon, isPointInPolygon, ratingLabel, ratingColor } from '../utils.js';
import { stateForPoint } from '../services/jurisdiction.js';
import { scanMatchesPolygon } from '../services/scan_flow.js';
import { modelSelectionFingerprint, packModelSession } from '../model3d_geometry.js';
import { clearAnalysisPolygon, clearManualPools, startDrawing, startPoolMarking, stopDrawing } from './drawing.js';
import {
  EMPTY_FC,
  boundsFromMap,
  featureCollection,
  geojsonBounds,
  pointFeature,
  setSourceData,
  setView,
  uiZoom
} from '../geo.js';

let osirisSummaryInFlight = false;

function resizeMapSoon(delay = 350) {
  setTimeout(() => { try { state.map.resize(); } catch (e) {} }, delay);
}

export function openPanel() {
  document.getElementById('side-panel').classList.add('open');
  resizeMapSoon();
}

export function closePanel() {
  document.getElementById('side-panel').classList.remove('open');
  resizeMapSoon();
}

export function clearAll() {
  stopDrawing();
  closePanel();
  if (state.mapReady) {
    setSourceData(state.map, 'selected-property-source', EMPTY_FC);
    setSourceData(state.map, 'locate-source', EMPTY_FC);
  }
  clearAnalysisPolygon();
  state.currentData = null;
  try { sessionStorage.removeItem('current3DData'); } catch (e) {}
  try { localStorage.removeItem('abi_current_3d_selection'); } catch (e) {}
  clearManualPools({ reanalyze: false });
  import('./osint.js').then(m => m.setSelectedPropertyPoint(null)).catch(() => {});

  const sv = document.getElementById('sv-wrap');
  sv.classList.remove('open', 'has-view');
  sv.style.height = '';
  document.getElementById('sv-frame').src = '';

  resizeMapSoon(400);
}

export function loadStreetView(lat, lng) {
  const wrap = document.getElementById('sv-wrap');
  document.getElementById('sv-frame').src = `https://maps.google.com/maps?cbll=${lat},${lng}&layer=c&cbp=11,0,0,0,0&output=svembed`;
  wrap.classList.add('open', 'has-view');
  resizeMapSoon();
}

export function renderAnalysis(d) {
  try {
    const packedModel = packModelSession(d);
    sessionStorage.setItem('current3DData', packedModel);
    if (d.property_id) localStorage.removeItem('abi_current_3d_selection');
    else localStorage.setItem('abi_current_3d_selection', packedModel);
  } catch (e) {}

  // Area Analysis is always a suitability report. Existing-pool detection is
  // intentionally not part of this workflow; property ranking views may still
  // display their recorded pool status through the property-specific payload.
  const isAreaAnalysis = !d.property_id;
  const isExistPool = !isAreaAnalysis && (
    state.hasExistingPool || d.has_existing_pool || (d.features && d.features.has_existing_pool)
  );
  const warningText = Array.isArray(d.data_warnings) && d.data_warnings.length
    ? `\n\nData notes: ${d.data_warnings.join(' ')}`
    : '';
  const explanation = d.explanation || '';
  const alreadyMentionsPool = /^existing swimming pool|^existing pool|^property already/i.test(explanation.trim());
  const explanationText = `${isExistPool && !alreadyMentionsPool ? 'Existing swimming pool detected. ' : ''}${explanation}${warningText}`;
  
  const poolWarning = document.getElementById('pool-warning');
  if (isExistPool) {
    poolWarning.style.display = 'block';
  } else {
    poolWarning.style.display = 'none';
  }

  if (d.property_id) {
    document.getElementById('sp-title').textContent = `Property: ${d.address}`;
    let ownerHtml = '';
    if (d.owner && d.owner.name) {
      ownerHtml = `
        <div style="font-size:.85rem;line-height:1.7;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,.08);">
          <div><strong>Owner:</strong> ${escapeHtml(d.owner.name)}</div>
          ${d.owner.phone ? `<div>Phone: ${escapeHtml(d.owner.phone)}</div>` : ''}
          ${d.owner.email ? `<div>Email: <a href="mailto:${encodeURIComponent(d.owner.email)}" style="color:var(--accent);text-decoration:none;">${escapeHtml(d.owner.email)}</a></div>` : ''}
        </div>
      `;
    }
    document.getElementById('sp-details').innerHTML = `
      <div style="font-size:.85rem;line-height:1.7;">
        <div><strong>${escapeHtml(d.suburb)}, ${escapeHtml(d.lga)}</strong></div>
        <div style="color:var(--muted);font-size:.76rem;">Property ID: ${escapeHtml(d.property_id)} · Rank: #${escapeHtml(d.rank || 'N/A')}</div>
      </div>
      ${ownerHtml}
    `;
  } else {
    const area = Number(d.area_sqm) || 0;
    document.getElementById('sp-title').textContent = `Area Analysis — ${area} sqm`;
    document.getElementById('sp-details').innerHTML = `
      <div style="font-size:.85rem;line-height:1.7;">
        <div><strong>${escapeHtml(area)} sqm</strong> (${(area/10000).toFixed(3)} ha)</div>
        <div style="color:var(--muted);font-size:.76rem;">Perimeter: ${escapeHtml(d.perimeter_m)}m · ${escapeHtml(d.shape?.n_vertices ?? 0)} vertices</div>
      </div>
      <div class="shape-chips">
        <span class="shape-chip">Compactness: ${Math.round(Number(d.shape?.compactness || 0)*100)}%</span>
        <span class="shape-chip">Fill: ${Math.round(Number(d.shape?.fill_ratio || 0)*100)}%</span>
        <span class="shape-chip">Aspect: ${escapeHtml(d.shape?.aspect_ratio ?? 0)}:1</span>
      </div>`;
  }

  const modules = [
    { key: 'pool_3d', label: 'Swimming Pool' },
    { key: 'pool_2d', label: 'Pool (Flat)' },
    { key: 'fence', label: 'Fencing' },
    { key: 'deck', label: 'Deck / Patio' },
    { key: 'landscaping', label: 'Landscaping' },
  ];
  let html = '';
  modules.forEach(m => {
    const candidate = Number(d.scores?.[m.key]);
    let v = Number.isFinite(candidate) ? Math.max(0, Math.min(100, candidate)) : 0;
    let r = ratingLabel(v);
    let col = ratingColor(v);
    let badgeValue = String(Math.round(v));
    let fillWidth = v;
    if (isExistPool && (m.key === 'pool_3d' || m.key === 'pool_2d')) {
      r = { text: 'Existing Pool', cls: 'existing' };
      col = '#a855f7';
      badgeValue = 'N/A';
      fillWidth = 100;
    }
    html += `<div class="rating-card"><div class="rating-top"><span class="rating-name">${m.label}</span><span class="rating-badge ${r.cls}">${r.text} · ${badgeValue}</span></div><div class="rating-bar"><div class="rating-fill" style="width:${fillWidth}%;background:${col}"></div></div></div>`;
  });
  document.getElementById('sp-ratings').innerHTML = html;

  const f = d.features || {};
  const hasValue = value => value !== undefined && value !== null && value !== '';
  const metrics = [
    { label: 'Total Area', val: hasValue(f.total_area_sqm) ? `${f.total_area_sqm} sqm` : (hasValue(d.area_sqm) ? `${d.area_sqm} sqm` : '—') },
    { label: 'Backyard', val: hasValue(f.backyard_area_sqm) ? `${f.backyard_area_sqm} sqm` : '—' },
    { label: 'Depth', val: hasValue(f.backyard_depth_m) ? `${f.backyard_depth_m} m` : '—' },
    { label: 'Frontage', val: hasValue(f.frontage_m) ? `${f.frontage_m} m` : (hasValue(f.road_frontage_m) ? `${f.road_frontage_m} m` : '—') },
    { label: 'Side Access', val: hasValue(f.max_side_clearance_m) ? `${f.max_side_clearance_m} m` : '—' },
    { label: 'Slope', val: f.slope_pct !== undefined ? `${f.slope_pct}%` : '—' },
    { label: 'Relief', val: f.relief_m !== undefined ? `${f.relief_m} m` : '—' },
    { label: 'Solar Access', val: f.solar_access_pct !== undefined ? `${f.solar_access_pct}%` : '—' },
    { label: 'Vegetation', val: f.vegetation_pct !== undefined ? `${f.vegetation_pct}%` : (d.vegetation_pct !== undefined ? `${d.vegetation_pct}%` : '—') },
    { label: 'Flood Risk', val: f.flood_risk_pct !== undefined ? `${f.flood_risk_pct}%` : (d.flood_risk !== undefined ? (d.flood_risk ? 'High' : 'Low') : '—') },
  ];
  let metricsHtml = '<div class="metrics-grid">';
  metrics.forEach(m => {
    metricsHtml += `<div class="metric-box"><div class="lbl">${escapeHtml(m.label)}</div><div class="val">${escapeHtml(m.val)}</div></div>`;
  });
  metricsHtml += '</div>';
  document.getElementById('sp-metrics').innerHTML = metricsHtml;

  document.getElementById('sp-explain').textContent = explanationText;

  // Wire up QGIS GeoPackage download via JS fetch+blob (no page navigation)
  const spQgis = document.getElementById('sp-qgis');
  if (spQgis) {
    const urlParams = new URLSearchParams(window.location.search);
    const moduleParam = urlParams.get('module') || 'pool_3d';
    const lgaParam = urlParams.get('lga') || '';
    let qgisUrl = `/api/export/qgis?module=${moduleParam}`;
    if (lgaParam) qgisUrl += `&lga=${encodeURIComponent(lgaParam)}`;
    spQgis._qgisUrl = qgisUrl;  // Store for click handler
  }

  const centroid = d.centroid || (d.polygon ? d.polygon[0] : null);
  const spSv = document.getElementById('sp-sv');
  const spEarth = document.getElementById('sp-earth');
  const spBird = document.getElementById('sp-bird');
  
  if (centroid) {
    spSv.href = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${centroid[1]},${centroid[0]}`;
    spSv.onclick = (e) => { e.preventDefault(); loadStreetView(centroid[1], centroid[0]); };

    spEarth.href = `https://earth.google.com/web/@${centroid[1]},${centroid[0]},50a,200d,35y,0h,45t,0r`;
    spEarth.onclick = null;

    spBird.href = `https://www.google.com/maps/@?api=1&map_action=map&center=${centroid[1]},${centroid[0]}&zoom=19`;
    spBird.onclick = null;

    // Feed the selected location to the tactical link graph so nearby OSINT
    // nodes draw semantic connection lines to it.
    import('./osint.js').then(m => m.setSelectedPropertyPoint({ lat: centroid[1], lon: centroid[0] })).catch(() => {});
  } else {
    spSv.href = '#';
    spSv.onclick = null;
    spEarth.href = '#';
    spEarth.onclick = null;
    spBird.href = '#';
    spBird.onclick = null;
  }
  
  const modelLink = document.getElementById('sp-3d');
  modelLink.href = '#';
  modelLink.classList.add('disabled');
  modelLink.classList.remove('needs-property');
  modelLink.setAttribute('aria-disabled', 'true');
  modelLink.onclick = event => event.preventDefault();
  modelLink.textContent = d.property_id ? '3D Property Model' : 'Choose Property for 3D';
  if (!d.property_id) {
    modelLink.classList.remove('disabled');
    modelLink.classList.add('needs-property');
    modelLink.setAttribute('aria-disabled', 'false');
    modelLink.title = 'A property model needs a selected ranked property, not only a drawn area.';
    modelLink.onclick = event => {
      event.preventDefault();
      showToast('For a 3D property model, run the suburb Scan and click a property parcel/lead first.', 'info');
    };
    return;
  }
  const selectionFingerprint = modelSelectionFingerprint(d);
  const parcelBounds = d.geometries?.parcel_geom ? geojsonBounds(d.geometries.parcel_geom) : null;
  const buildingBounds = d.geometries?.building_geom ? geojsonBounds(d.geometries.building_geom) : null;
  if (parcelBounds && buildingBounds) {
    let href = `/3d?south=${parcelBounds.getSouth()}&north=${parcelBounds.getNorth()}&west=${parcelBounds.getWest()}&east=${parcelBounds.getEast()}`;
    href += `&propertyId=${encodeURIComponent(d.property_id)}`;
    if (selectionFingerprint) href += `&selection=${encodeURIComponent(selectionFingerprint)}`;
    modelLink.href = href;
    modelLink.classList.remove('disabled');
    modelLink.setAttribute('aria-disabled', 'false');
    modelLink.onclick = null;
    modelLink.title = `Open the selected-property model for ${d.address || d.property_id}`;
  } else {
    modelLink.textContent = '3D Geometry Unavailable';
    modelLink.title = 'This property record has no usable building footprint.';
  }
}

function setPanelText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = String(value);
}

function clampScanPasses(value) {
  const candidate = Number(value || 12);
  if (!Number.isFinite(candidate)) return 12;
  return Math.max(3, Math.min(40, Math.round(candidate)));
}

function poolFeatureCenter(feature) {
  if (!feature) return null;
  if (feature.type === 'node') {
    const lat = Number(feature.lat);
    const lon = Number(feature.lon);
    return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
  }
  const geometry = Array.isArray(feature.geometry) ? feature.geometry : [];
  const points = geometry
    .map(point => ({ lat: Number(point.lat), lon: Number(point.lon) }))
    .filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lon));
  if (!points.length) return null;
  return {
    lat: points.reduce((sum, point) => sum + point.lat, 0) / points.length,
    lon: points.reduce((sum, point) => sum + point.lon, 0) / points.length
  };
}

function featureInCurrentView(feature) {
  if (!state.map) return true;
  const center = poolFeatureCenter(feature);
  return Boolean(center && boundsFromMap(state.map).contains(center));
}

function isSatellitePool(feature) {
  const tags = feature?.tags || {};
  const source = String(tags['detector:source'] || '');
  const method = String(tags['detector:method'] || '');
  return (
    source === 'satellite-cv' ||
    source === 'satellite-yolo' ||
    source === 'yolo-cv' ||
    method.includes('satellite-water') ||
    method.includes('pool-trained-yolo')
  );
}

function poolConfidenceLabel(feature) {
  const tags = feature?.tags || {};
  const raw = tags['confidence:yolo'] || tags['confidence:cv'] || tags['confidence:tag'];
  const confidence = Number(raw);
  const pct = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : 'cached';
  // Scan-voted pools earned their place through cross-pass agreement — say so, otherwise
  // a voted pool with a moderate raw frame score reads like an unvetted weak detection.
  if (tags['scan:verdict'] === 'confirmed') {
    const seen = Number(tags['scan:passes_seen']) || 0;
    return seen > 1 ? `verified ${seen}× · ${pct}` : `verified · ${pct}`;
  }
  return pct;
}

function poolMapsUrl(center) {
  if (!center) return '#';
  return `https://www.google.com/maps/@?api=1&map_action=map&center=${center.lat},${center.lon}&zoom=20`;
}

function visiblePoolCacheItems() {
  const pools = Array.from(state.globalPools?.values?.() || []);
  const visiblePools = pools.filter(featureInCurrentView);
  const satellitePools = visiblePools.filter(isSatellitePool);
  return (satellitePools.length ? satellitePools : visiblePools).slice(-8).reverse();
}

function exportDetectionsCsv() {
  const centroid = (f) => {
    if (f.type === 'node' && typeof f.lat === 'number') return [f.lat, f.lon];
    const g = Array.isArray(f.geometry) ? f.geometry : [];
    if (!g.length) return [null, null];
    return [
      g.reduce((s, p) => s + p.lat, 0) / g.length,
      g.reduce((s, p) => s + p.lon, 0) / g.length
    ];
  };
  const rows = [['type', 'state', 'lat', 'lon', 'confidence', 'area_sqm', 'scan_verdict', 'source', 'times_seen', 'google_maps']];
  state.globalPools.forEach(f => {
    const [lat, lon] = centroid(f);
    if (lat === null) return;
    const t = f.tags || {};
    rows.push([
      t['detector:source'] ? 'pool (AI detected)' : 'pool (OSM mapped)',
      stateForPoint(lat, lon).toUpperCase(),
      lat.toFixed(7), lon.toFixed(7),
      t['scan:confidence'] || t['confidence:yolo'] || '',
      t['pool:area_sqm'] || '',
      t['scan:verdict'] || '',
      t['detector:source'] || 'openstreetmap',
      t['scan:passes_seen'] || '',
      `https://www.google.com/maps?q=${lat.toFixed(7)},${lon.toFixed(7)}`
    ]);
  });
  state.globalSolarPanels.forEach(f => {
    const [lat, lon] = centroid(f);
    if (lat === null) return;
    const t = f.tags || {};
    rows.push([
      'solar array (AI detected)', stateForPoint(lat, lon).toUpperCase(),
      lat.toFixed(7), lon.toFixed(7),
      t['confidence:yolo'] || '', t['solar:area_sqm'] || '', '',
      'satellite-yolo-solar', '',
      `https://www.google.com/maps?q=${lat.toFixed(7)},${lon.toFixed(7)}`
    ]);
  });
  if (rows.length === 1) {
    showToast('Nothing to export yet — run a scan or browse with detection on first.', 'error');
    return;
  }
  const csv = rows.map(r => r.map(v => `"${String(v ?? '').replace(/"/g, '""')}"`).join(',')).join('\r\n');
  // ﻿ BOM so Excel opens the file as UTF-8.
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `detections-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  showToast(`Exported ${rows.length - 1} detections to CSV.`, 'success');
}

export function updateOsirisIntelPanel(message = '') {
  const panel = document.getElementById('osiris-intel-panel');
  if (!panel) return;

  const inSelection = feature => !state.polygon || featureIntersectsPolygon(feature, state.polygon);
  const pools = Array.from(state.globalPools?.values?.() || []).filter(inSelection);
  const structures = Array.from(state.globalBuildings?.values?.() || []);
  const solarPanels = Array.from(state.globalSolarPanels?.values?.() || []).filter(inSelection);
  setPanelText('osiris-pool-count', pools.length);
  setPanelText('osiris-solar-count', solarPanels.length);
  setPanelText('osiris-structure-count', structures.length);

  const status = document.getElementById('osiris-intel-status');
  if (status && message) status.textContent = message;

  const poolList = document.getElementById('osiris-pool-list');
  const showPools = document.getElementById('osiris-layer-pools')?.checked ?? true;
  if (!poolList) return;
  if (!showPools) {
    poolList.innerHTML = '<div class="osiris-empty">Mapped pool cache hidden.</div>';
    return;
  }

  const items = visiblePoolCacheItems();
  if (!items.length) {
    poolList.innerHTML = '<div class="osiris-empty">No pools loaded for the selected area yet.</div>';
    return;
  }

  poolList.innerHTML = items.map((feature, index) => {
    const center = poolFeatureCenter(feature);
    const source = isSatellitePool(feature) ? 'Satellite detection' : 'Mapped/recorded pool';
    const mapsUrl = poolMapsUrl(center);
    const lat = center ? center.lat.toFixed(5) : 'unknown';
    const lon = center ? center.lon.toFixed(5) : 'unknown';
    return `
      <div class="osiris-pool-item" data-pool-index="${index}">
        <strong>${source} · ${poolConfidenceLabel(feature)}</strong>
        <span>${lat}, ${lon}</span>
        <div class="osiris-pool-actions">
          <button type="button" data-action="fly-pool" data-index="${index}">Map</button>
          <a href="${escapeHtml(mapsUrl)}" target="_blank" rel="noopener">Google Maps</a>
        </div>
      </div>
    `;
  }).join('');

  poolList.querySelectorAll('[data-action="fly-pool"]').forEach(button => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.index);
      const feature = items[index];
      const center = poolFeatureCenter(feature);
      if (center && state.map) {
        setView(state.map, center.lat, center.lon, Math.max(18, uiZoom(state.map)));
      }
    });
  });
}

async function refreshOsirisAgentSummary() {
  if (osirisSummaryInFlight) return;
  osirisSummaryInFlight = true;
  try {
    const [scansResponse, leadsResponse] = await Promise.all([
      fetch('/api/agent/scans?limit=3'),
      fetch('/api/agent/leads?limit=25')
    ]);
    const scansPayload = await scansResponse.json().catch(() => ({}));
    const leadsPayload = await leadsResponse.json().catch(() => ({}));
    const allLeads = Array.isArray(leadsPayload.items) ? leadsPayload.items : [];
    const allScans = Array.isArray(scansPayload.items) ? scansPayload.items : [];
    const coords = state.polygon?.vertices?.map(point => [point.lng, point.lat]) || [];
    const scans = coords.length >= 3
      ? allScans.filter(scan => scanMatchesPolygon(scan, coords))
      : [];
    const leads = state.polygon
      ? allLeads.filter(lead => isPointInPolygon(
          { lat: Number(lead.lat), lng: Number(lead.lon) },
          state.polygon
        ))
      : [];
    setPanelText('osiris-leads-count', leads.length);
    const running = scans.find(scan => scan.status === 'running');
    if (running) {
      const done = running.passes_completed || 0;
      const total = running.passes_total || running.passes_requested || 0;
      const blocks = running.blocks || 1;
      const sweeps = running.sweeps || running.passes_requested || 0;
      const progress = blocks > 1
        ? `${done}/${total} block-scans (${blocks} blocks × ${sweeps} sweeps)`
        : `${done}/${total} refinement sweeps`;
      updateOsirisIntelPanel(`${running.label || 'Area scan'}: ${progress}, ${running.leads?.length || 0} leads.`);
    } else if (scans[0]) {
      updateOsirisIntelPanel(scans[0].summary || 'Latest scan completed.');
    }
  } catch (error) {
    updateOsirisIntelPanel(`Lead summary unavailable: ${error.message}`);
  } finally {
    osirisSummaryInFlight = false;
  }
}

async function scanOsirisPlace() {
  const areaInput = document.getElementById('osiris-area-input');
  const passInput = document.getElementById('osiris-pass-input');
  const scanButton = document.getElementById('osiris-place-scan-btn');
  const area = (areaInput?.value || '').trim();
  if (!area) return;
  if (scanButton) scanButton.disabled = true;
  updateOsirisIntelPanel(`Scanning ${area} for leads...`);
  try {
    // Run the SAME scan pipeline as the polygon "Run area scan": start, poll, ingest,
    // and render detected pools as map polygons. Only the input differs (suburb → bbox).
    const { startPlaceScan } = await import('../services/api.js');
    await startPlaceScan(area, { passes: clampScanPasses(passInput?.value) });
    refreshOsirisAgentSummary();
  } catch (error) {
    updateOsirisIntelPanel(`Place scan failed: ${error.message}`);
    showToast(`Place scan failed: ${error.message}`, 'error');
  } finally {
    if (scanButton) scanButton.disabled = false;
  }
}

// Google-Maps-style place autocomplete for the area search box. Suggestions come
// from the existing /api/geocode (Nominatim) endpoint; selecting one fills the box
// and pans the map there. Enter with no active suggestion runs the lead scan.
function initAreaAutocomplete(input) {
  if (!input) return;
  const list = document.getElementById('osiris-area-suggestions');
  if (!list) return;

  let items = [];
  let activeIndex = -1;
  let debounceTimer = null;
  let requestSeq = 0;
  let suppressInput = false;

  const close = () => {
    list.hidden = true;
    list.innerHTML = '';
    items = [];
    activeIndex = -1;
    input.setAttribute('aria-expanded', 'false');
  };

  const splitLabel = (displayName) => {
    const parts = String(displayName).split(',').map(s => s.trim()).filter(Boolean);
    const primary = parts.shift() || displayName;
    return { primary, secondary: parts.join(', ') };
  };

  const render = () => {
    if (!items.length) { close(); return; }
    list.innerHTML = items.map((it, i) => {
      const { primary, secondary } = splitLabel(it.display_name);
      return `<li class="osiris-suggestion${i === activeIndex ? ' active' : ''}" role="option" data-index="${i}" aria-selected="${i === activeIndex}"><strong>${escapeHtml(primary)}</strong>${secondary ? `<span>${escapeHtml(secondary)}</span>` : ''}</li>`;
    }).join('');
    list.hidden = false;
    input.setAttribute('aria-expanded', 'true');
    list.querySelectorAll('.osiris-suggestion').forEach(el => {
      // mousedown (not click) so it fires before the input's blur closes the list.
      el.addEventListener('mousedown', (event) => {
        event.preventDefault();
        choose(Number(el.dataset.index));
      });
    });
  };

  const choose = (index) => {
    const item = items[index];
    if (!item) return;
    suppressInput = true;             // updating .value must not reopen suggestions
    input.value = item.display_name;
    const lat = Number(item.lat);
    const lon = Number(item.lon);
    if (state.map && Number.isFinite(lat) && Number.isFinite(lon)) {
      setView(state.map, lat, lon, 16);
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));  // keep dashboard link in sync
    close();
    input.focus();
  };

  const fetchSuggestions = async (queryText) => {
    const seq = ++requestSeq;
    try {
      const response = await fetch(`/api/geocode?q=${encodeURIComponent(queryText)}`);
      const payload = await response.json().catch(() => ({}));
      if (seq !== requestSeq) return;  // a newer keystroke superseded this request
      items = (response.ok && Array.isArray(payload.items)) ? payload.items.slice(0, 5) : [];
      activeIndex = -1;
      render();
    } catch (error) {
      if (seq === requestSeq) close();
    }
  };

  input.addEventListener('input', () => {
    if (suppressInput) { suppressInput = false; return; }
    const queryText = input.value.trim();
    window.clearTimeout(debounceTimer);
    if (queryText.length < 2) { close(); return; }
    debounceTimer = window.setTimeout(() => fetchSuggestions(queryText), 250);
  });

  input.addEventListener('keydown', (event) => {
    const open = !list.hidden && items.length > 0;
    if (event.key === 'ArrowDown' && open) {
      event.preventDefault();
      activeIndex = (activeIndex + 1) % items.length;
      render();
    } else if (event.key === 'ArrowUp' && open) {
      event.preventDefault();
      activeIndex = (activeIndex - 1 + items.length) % items.length;
      render();
    } else if (event.key === 'Enter') {
      if (open && activeIndex >= 0) {
        event.preventDefault();
        choose(activeIndex);
      } else {
        close();
        scanOsirisPlace();
      }
    } else if (event.key === 'Escape') {
      close();
    }
  });

  // Close when focus leaves; small delay lets a suggestion mousedown register first.
  input.addEventListener('blur', () => window.setTimeout(close, 120));
}

export function initUIElements() {
  document.getElementById('sp-close').addEventListener('click', clearAll);
  document.getElementById('sp-clear').addEventListener('click', clearAll);
  document.getElementById('draw-area-btn').addEventListener('click', startDrawing);
  document.getElementById('mark-pool-btn').addEventListener('click', startPoolMarking);
  document.getElementById('clear-pools-btn').addEventListener('click', () => clearManualPools());

  // QGIS Export button — JS-driven download with loading state and toast
  document.getElementById('sp-qgis').addEventListener('click', async () => {
    const btn = document.getElementById('sp-qgis');
    const originalText = btn.textContent;
    btn.textContent = 'Exporting...';
    btn.disabled = true;
    btn.style.opacity = '0.6';

    try {
      // A drawn area exports the selected polygon + its detected features; a ranked
      // property/module export uses the GET endpoint. Pick based on what's loaded.
      const data = state.currentData;
      const isAreaAnalysis = data && Array.isArray(data.polygon) && !data.property_id;
      const res = isAreaAnalysis
        ? await fetch('/api/export/qgis/area', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
          })
        : await fetch(btn._qgisUrl || '/api/export/qgis');

      if (!res.ok) {
        let errMsg = 'Export failed';
        try {
          const errData = await res.json();
          errMsg = errData.error || errMsg;
        } catch (e) { /* ignore parse errors */ }
        showToast(errMsg, 'error');
        return;
      }

      const blob = await res.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      const disposition = res.headers.get('Content-Disposition') || '';
      const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
      a.download = filenameMatch ? filenameMatch[1] : 'qgis_export.gpkg';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(downloadUrl);
      showToast('QGIS GeoPackage exported.', 'success');
    } catch (err) {
      console.error('QGIS export error:', err);
      showToast(`Export failed: ${err.message}`, 'error');
    } finally {
      btn.textContent = originalText;
      btn.disabled = false;
      btn.style.opacity = '1';
    }
  });

  const svWrap = document.getElementById('sv-wrap');
  const svResizer = document.getElementById('sv-resizer');
  let isResizing = false;

  svResizer.addEventListener('mousedown', () => {
    isResizing = true;
    document.body.style.cursor = 'ns-resize';
    svWrap.style.transition = 'none';
    document.getElementById('sv-frame').style.pointerEvents = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    let newHeight = window.innerHeight - e.clientY;
    if (newHeight < 100) newHeight = 100;
    if (newHeight > window.innerHeight * 0.8) newHeight = window.innerHeight * 0.8;
    svWrap.style.height = newHeight + 'px';
    try { state.map.resize(); } catch (err) {}
  });

  document.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      document.body.style.cursor = '';
      svWrap.style.transition = '';
      document.getElementById('sv-frame').style.pointerEvents = 'auto';
      try { state.map.resize(); } catch (err) {}
    }
  });

  // Location Search setup
  const searchInput = document.getElementById('search-input');
  const searchBtn = document.getElementById('search-btn');
  const locateBtn = document.getElementById('locate-btn');
  const searchResults = document.getElementById('search-results');

  async function performSearch() {
    const q = searchInput.value.trim();
    if (!q) return;
    searchResults.style.display = 'flex';
    searchResults.innerHTML = '<div class="search-item">Searching...</div>';
    try {
      const res = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.error || `Search failed (${res.status})`);
      const data = Array.isArray(payload.items) ? payload.items : [];
      searchResults.innerHTML = '';
      if (data.length === 0) {
        searchResults.innerHTML = '<div class="search-item" style="color:var(--muted)">No results found</div>';
        return;
      }
      data.forEach(item => {
        const div = document.createElement('div');
        div.className = 'search-item';
        div.textContent = item.display_name;
        div.onclick = () => {
          setView(state.map, Number(item.lat), Number(item.lon), 18);
          searchResults.style.display = 'none';
          searchInput.value = item.display_name.split(',')[0];
          const osirisAreaInput = document.getElementById('osiris-area-input');
          if (osirisAreaInput) {
            osirisAreaInput.value = item.display_name.split(',')[0];
            osirisAreaInput.dispatchEvent(new Event('input'));
          }
        };
        searchResults.appendChild(div);
      });
    } catch (e) {
      searchResults.innerHTML = '<div class="search-item" style="color:#ef4444">Search failed</div>';
    }
  }

  searchBtn.addEventListener('click', performSearch);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') performSearch();
  });

  locateBtn.addEventListener('click', () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser.");
      return;
    }
    
    locateBtn.textContent = 'Locating...';
    locateBtn.style.color = '#fbbf24';
    
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        
        locateBtn.textContent = 'Locate';
        locateBtn.style.color = '#10b981';

        state.map.flyTo({ center: [lon, lat], zoom: 17, duration: 2000 });  // style z17 = ui z18

        if (state.mapReady) {
          setSourceData(state.map, 'locate-source', featureCollection([
            pointFeature({ lat, lon }, { tooltip: 'You are here (Live Location)' })
          ]));
        }
      },
      (error) => {
        locateBtn.textContent = 'Locate';
        locateBtn.style.color = '#ef4444';
        let msg = "Could not retrieve your location.";
        if (error.code === error.PERMISSION_DENIED) {
          msg = "Location access denied by browser permissions.";
        }
        alert(msg);
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }
    );
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('#search-box')) {
      searchResults.style.display = 'none';
    }
  });



  const basemapSelect = document.getElementById('map-basemap');

  const BASEMAP_LABELS = {
    'au-aerial': 'Hybrid AU Aerial (Esri + state gov imagery)',
    'esri': 'Esri World Imagery',
    'google-satellite': 'Google Satellite tiles',
    'google': 'Google Maps tiles'
  };

  basemapSelect.addEventListener('change', () => {
    const val = basemapSelect.value;

    import('./map.js').then(m => {
      m.switchBasemap(val);
      m.showGoogleOverlayMap();
      m.renderAILayers();
    });

    showToast(`${BASEMAP_LABELS[val] || 'Basemap'} is active with synchronized analysis overlays.`, 'info');
  });

  const runAIDetectionButton = document.getElementById('run-ai-detection-btn');
  runAIDetectionButton?.addEventListener('click', async () => {
    try {
      const api = await import('../services/api.js');
      await api.scanSelectedAreaForAIDetections({ forceRefresh: true });
    } catch (error) {
      console.error(error);
    }
  });

  const runAgentScanButton = document.getElementById('run-agent-scan-btn');
  if (runAgentScanButton) {
    runAgentScanButton.addEventListener('click', async () => {
      runAgentScanButton.disabled = true;
      try {
        const api = await import('../services/api.js');
        await api.startAreaScanFromCurrentView({ source: 'manual-button' });
      } catch (error) {
        showToast(`Area scan failed: ${error.message}`, 'error');
      } finally {
        if (!state.agentScanActiveId && !state.agentScanStartInFlight) {
          runAgentScanButton.disabled = false;
        }
      }
    });
  }

  const osirisAreaInput = document.getElementById('osiris-area-input');
  const osirisScanButton = document.getElementById('osiris-place-scan-btn');
  const osirisPoolsToggle = document.getElementById('osiris-layer-pools');

  osirisScanButton?.addEventListener('click', scanOsirisPlace);
  document.getElementById('export-csv-btn')?.addEventListener('click', exportDetectionsCsv);
  initAreaAutocomplete(osirisAreaInput);
  osirisPoolsToggle?.addEventListener('change', () => updateOsirisIntelPanel());
  updateOsirisIntelPanel();
  refreshOsirisAgentSummary();
  window.setInterval(refreshOsirisAgentSummary, 10000);

  const dashboardLink = document.getElementById('leads-dashboard-link');
  if (dashboardLink) {
    const urlParams = new URLSearchParams(window.location.search);
    const moduleParam = urlParams.get('module') || 'pool_3d';
    const updateDashboardLink = () => {
      const areaParam = urlParams.get('area') || osirisAreaInput?.value || 'Sydney NSW';
      dashboardLink.href = `/dashboard?module=${encodeURIComponent(moduleParam)}&area=${encodeURIComponent(areaParam)}`;
    };
    updateDashboardLink();
    osirisAreaInput?.addEventListener('input', updateDashboardLink);
  }

  // ---- Dev mode: evaluation harness + detection feedback tools ----
  initDevMode();

  // Trigger initial basemap setup once loaded
  setTimeout(() => {
    basemapSelect.dispatchEvent(new Event('change'));
  }, 50);
}

function renderEvalReport(report, statusEl) {
  if (!report || report.error) {
    statusEl.textContent = report?.error || 'No evaluation report yet.';
    statusEl.dataset.tone = 'idle';
    return;
  }
  const pct = value => (value === null || value === undefined ? 'n/a' : `${Math.round(value * 100)}%`);
  const ci = report.recall_ci_95;
  const ciText = ci ? ` (95% CI ${pct(ci.low)}–${pct(ci.high)})` : '';
  const eligible = report.marketing_audit?.pool_national_claim_eligible === true;
  const scope = eligible ? 'HOLDOUT' : 'DEVELOPMENT ONLY — NO NATIONAL CLAIM';
  const blocker = report.marketing_audit?.blockers?.[0];
  statusEl.textContent =
    `${scope} · Recall ${pct(report.recall)}${ciText} · FP-rate ${pct(report.fp_rate)} · precision~${pct(report.precision_proxy)} ` +
    `(${report.tp}/${report.tp + report.fn} pools found, ${report.fp} FPs on ${report.fp + report.tn} negatives, ` +
    `${report.cells_scanned} cells) · ${report.created_at}${report.build ? ' · ' + report.build : ''}` +
    (blocker ? ` · Blocked: ${blocker}` : '');
  statusEl.dataset.tone = eligible ? 'success' : 'error';
}

function initDevMode() {
  const toggle = document.getElementById('dev-mode-toggle');
  const tools = document.getElementById('dev-tools');
  const statusEl = document.getElementById('eval-status');
  const bootstrapBtn = document.getElementById('eval-bootstrap-btn');
  const runBtn = document.getElementById('eval-run-btn');
  if (!toggle || !tools) return;

  let saved = false;
  // abi_* is the current key; qldr_* is read once for migration from pre-rename builds.
  try { saved = (localStorage.getItem('abi_dev_mode') ?? localStorage.getItem('qldr_dev_mode')) === '1'; } catch (e) {}
  toggle.checked = saved;
  tools.hidden = !saved;

  // ---- Automatic data pipeline (no manual feeding while dev mode is on) ----
  // 1) every new ~900 m cell the user settles on gets OSM-bootstrapped once;
  // 2) completed scans harvest verified detections server-side;
  // 3) the pipeline line shows the dataset growing.
  const pipelineEl = document.getElementById('pipeline-status');
  const bootstrappedCells = new Set(
    (() => { try { return JSON.parse(localStorage.getItem('abi_auto_bootstrap_cells') || localStorage.getItem('qldr_auto_bootstrap_cells') || '[]'); } catch (e) { return []; } })()
  );
  let autoBootstrapBusy = false;
  let autoBootstrapTimer = null;

  async function refreshPipelineStatus() {
    if (!toggle.checked || !pipelineEl) return;
    try {
      const status = await fetch('/api/eval/pipeline').then(r => r.json());
      const l = status.labels || {};
      const t = status.training_candidates || {};
      pipelineEl.textContent =
        `Auto-pipeline: ${l.total || 0} eval labels (${l.from_feedback || 0} human) · ` +
        `${t.total || 0} training candidates (${t.pool || 0} pool / ${t.solar || 0} solar) · ` +
        `${status.feedback_total || 0} feedback clicks`;
      pipelineEl.dataset.tone = 'success';
    } catch (e) { /* server briefly busy — next tick catches up */ }
  }

  async function autoBootstrapViewport() {
    if (!toggle.checked || autoBootstrapBusy || !state.map) return;
    if (uiZoom(state.map) < 16) return;
    const c = state.map.getCenter();
    const cellKey = `${Math.floor(c.lat / 0.008)}:${Math.floor(c.lng / 0.008)}`;
    if (bootstrappedCells.has(cellKey)) return;
    autoBootstrapBusy = true;
    try {
      const bounds = state.map.getBounds();
      const response = await fetch('/api/eval/bootstrap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bounds: {
            south: bounds.getSouth(), west: bounds.getWest(),
            north: bounds.getNorth(), east: bounds.getEast()
          },
          area: 'auto-pipeline'
        })
      });
      if (response.ok) {
        bootstrappedCells.add(cellKey);
        try { localStorage.setItem('abi_auto_bootstrap_cells', JSON.stringify([...bootstrappedCells].slice(-400))); } catch (e) {}
        await refreshPipelineStatus();
      }
    } catch (e) { /* transient — retried on the next settle */ }
    finally { autoBootstrapBusy = false; }
  }

  function scheduleAutoBootstrap() {
    if (!toggle.checked) return;
    window.clearTimeout(autoBootstrapTimer);
    autoBootstrapTimer = window.setTimeout(autoBootstrapViewport, 3500);
  }
  state.map?.on?.('moveend', scheduleAutoBootstrap);
  window.setInterval(() => { if (toggle.checked) refreshPipelineStatus(); }, 30000);
  if (saved) { scheduleAutoBootstrap(); refreshPipelineStatus(); }

  toggle.addEventListener('change', () => {
    tools.hidden = !toggle.checked;
    try { localStorage.setItem('abi_dev_mode', toggle.checked ? '1' : '0'); } catch (e) {}
    if (toggle.checked) {
      fetch('/api/eval/report').then(r => (r.ok ? r.json() : null)).then(rep => {
        if (rep && statusEl) renderEvalReport(rep, statusEl);
      }).catch(() => {});
      scheduleAutoBootstrap();
      refreshPipelineStatus();
    }
  });
  if (saved && statusEl) {
    fetch('/api/eval/report').then(r => (r.ok ? r.json() : null)).then(rep => {
      if (rep) renderEvalReport(rep, statusEl);
    }).catch(() => {});
  }

  bootstrapBtn?.addEventListener('click', async () => {
    bootstrapBtn.disabled = true;
      statusEl.textContent = 'Bootstrapping development labels from OpenStreetMap for the current viewport...';
    statusEl.dataset.tone = 'running';
    try {
      const bounds = state.map.getBounds();
      const response = await fetch('/api/eval/bootstrap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bounds: {
            south: bounds.getSouth(), west: bounds.getWest(),
            north: bounds.getNorth(), east: bounds.getEast()
          },
          area: document.getElementById('osiris-area-input')?.value || ''
        })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Bootstrap failed (${response.status})`);
      const c = payload.counts || {};
      statusEl.textContent =
        `Added ${payload.added_pool_labels} pool + ${payload.added_no_pool_labels} no-pool labels. ` +
        `Set now: ${c.pool || 0} pool / ${c.no_pool || 0} no-pool (${c.from_feedback || 0} from feedback). ` +
        `OSM bootstrap is development data only and cannot support a national accuracy claim.`;
      statusEl.dataset.tone = 'success';
    } catch (error) {
      statusEl.textContent = `Bootstrap failed: ${error.message}`;
      statusEl.dataset.tone = 'error';
    } finally {
      bootstrapBtn.disabled = false;
    }
  });

  runBtn?.addEventListener('click', async () => {
    runBtn.disabled = true;
    statusEl.textContent = 'Running detector over labelled cells (can take a few minutes)...';
    statusEl.dataset.tone = 'running';
    try {
      const buildTag = document.getElementById('build-badge')?.textContent?.trim().split('·')[0].trim() || '';
      const response = await fetch('/api/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_cells: 12, build: buildTag })
      });
      const report = await response.json();
      if (!response.ok) throw new Error(report.error || `Evaluation failed (${response.status})`);
      renderEvalReport(report, statusEl);
    } catch (error) {
      statusEl.textContent = `Evaluation failed: ${error.message}`;
      statusEl.dataset.tone = 'error';
    } finally {
      runBtn.disabled = false;
    }
  });
}

// Toast notification utility — slides up from bottom, auto-dismisses
export function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.style.cssText = `
    padding: 10px 20px;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #fff;
    pointer-events: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
    animation: toastSlideUp 0.3s ease forwards;
    max-width: 400px;
    text-align: center;
  `;
  toast.style.background = type === 'error'
    ? 'linear-gradient(135deg, rgba(239,68,68,0.9), rgba(185,28,28,0.9))'
    : 'linear-gradient(135deg, rgba(16,185,129,0.9), rgba(5,150,105,0.9))';
  toast.textContent = message;

  // Add keyframe animation if not yet injected
  if (!document.getElementById('toast-keyframes')) {
    const style = document.createElement('style');
    style.id = 'toast-keyframes';
    style.textContent = `
      @keyframes toastSlideUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
      @keyframes toastFadeOut { from { opacity:1; } to { opacity:0; transform:translateY(-10px); } }
    `;
    document.head.appendChild(style);
  }

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastFadeOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
