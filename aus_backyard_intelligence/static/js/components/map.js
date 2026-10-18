// MapLibre GL viewer core (build r22): Nearmap-grade WebGL rendering on a fully
// keyless FOSS stack — Esri World Imagery underlay + Australian state government
// aerial services + OSM/Google raster options, true 3D camera, fill-extrusion
// buildings, and GeoJSON feature pipelines for every analysis overlay.
import { state } from '../state.js';
import { fetchAIDataInView, fetchTerrainGridInView, loadPersistentPoolCache, scanSelectedAreaForAIDetections } from '../services/api.js';
import { escapeHtml } from '../utils.js';
import { poolRegisterLinkHtml } from '../services/jurisdiction.js';
import {
  GeoBounds,
  boundsFromMap,
  circlePolygonFeature,
  featureCollection,
  EMPTY_FC,
  lineFeature,
  pointFeature,
  polygonFeature,
  setLayersVisibility,
  setSourceData,
  setView,
  normalizeStartupZoom,
  uiZoom
} from '../geo.js';
import {
  buildElevationContours,
  estimateSolarScore,
  geometryAreaSqm,
  geometryCentroid,
  geometryExtent,
  inferBuildingHeight,
  inferCanopyRadius,
  isValidBuildingGeometry,
  isValidPoolGeometry,
  normalizeGeometry,
  poolConfidence
} from '../services/feature_algorithms.js';

// ---------------------------------------------------------------------------
// Basemap registry (100% keyless). Each entry lists the raster layer ids that
// make up that basemap; switching = toggling layer visibility, never rebuilding
// the style (so vector overlays and the camera are untouched).
const BASEMAP_GROUPS = {
  'au-aerial': ['base-esri', 'gov-nsw', 'gov-sa', 'gov-qld', 'gov-vic', 'gov-tas'],
  'esri': ['base-esri'],
  'google-satellite': ['base-google-satellite'],
  'google': ['base-google-roadmap']
};
const ALL_BASEMAP_LAYERS = [...new Set(Object.values(BASEMAP_GROUPS).flat())];

// Australian state government open aerial services, expressed as bounded raster
// sources. MapLibre only requests tiles inside `bounds`, which reproduces the
// old per-tile lat/lon router: outside a service's coverage the Esri underlay
// simply shows through (404s and transparent blanks self-heal the same way).
// Priority (top of stack wins): TAS > VIC > QLD > SA > NSW.
function rasterSource(tiles, { maxzoom = 19, attribution = '', bounds = null, tileSize = 256 } = {}) {
  const source = { type: 'raster', tiles, tileSize, maxzoom, attribution };
  if (bounds) source.bounds = bounds;
  return source;
}

const GOV_SOURCES = {
  'gov-tas-src': rasterSource(
    ['https://services.thelist.tas.gov.au/arcgis/rest/services/Basemaps/Orthophoto/MapServer/tile/{z}/{y}/{x}'],
    { bounds: [143.6, -44.0, 148.9, -39.2], maxzoom: 19, attribution: 'TAS (theLIST)' }
  ),
  'gov-vic-src': rasterSource(
    ['https://base.maps.vic.gov.au/service?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=AERIAL_WM_256&STYLE=&TILEMATRIXSET=EPSG:3857:256&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image%2Fpng'],
    { bounds: [140.96, -39.25, 150.05, -34.0], maxzoom: 20, attribution: 'VIC (Vicmap)' }
  ),
  'gov-qld-src': rasterSource(
    ['https://spatial-img.information.qld.gov.au/arcgis/rest/services/Basemaps/LatestStateProgram_AllUsers/ImageServer/tile/{z}/{y}/{x}'],
    { bounds: [137.999, -29.2, 153.65, -8.9], maxzoom: 20, attribution: 'QLD State Program' }
  ),
  'gov-sa-src': rasterSource(
    ['https://imagemap.geohub.sa.gov.au/mapproxy/wmts/PublicMosaic/webmercator_22/{z}/{x}/{y}.png'],
    { bounds: [129.001, -38.2, 140.999, -26.001], maxzoom: 21, attribution: 'SA Geohub' }
  ),
  'gov-nsw-src': rasterSource(
    ['https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer/tile/{z}/{y}/{x}'],
    { bounds: [141.0, -37.6, 153.7, -28.15], maxzoom: 20, attribution: 'NSW/ACT (SIX)' }
  )
};

function buildBaseStyle() {
  const sources = {
    'esri-src': rasterSource(
      ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      { maxzoom: 19, attribution: 'Esri, Maxar, Earthstar Geographics' }
    ),
    'google-satellite-src': rasterSource(
      ['https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'],
      { maxzoom: 21, attribution: '&copy; Google' }
    ),
    'google-roadmap-src': rasterSource(
      ['https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'],
      { maxzoom: 21, attribution: '&copy; Google' }
    )
  };
  Object.assign(sources, GOV_SOURCES);

  const layers = [
    { id: 'background', type: 'background', paint: { 'background-color': '#0b1220' } },
    { id: 'base-esri', type: 'raster', source: 'esri-src' },
    { id: 'gov-nsw', type: 'raster', source: 'gov-nsw-src' },
    { id: 'gov-sa', type: 'raster', source: 'gov-sa-src' },
    { id: 'gov-qld', type: 'raster', source: 'gov-qld-src' },
    { id: 'gov-vic', type: 'raster', source: 'gov-vic-src' },
    { id: 'gov-tas', type: 'raster', source: 'gov-tas-src' },
    { id: 'base-google-satellite', type: 'raster', source: 'google-satellite-src', layout: { visibility: 'none' } },
    { id: 'base-google-roadmap', type: 'raster', source: 'google-roadmap-src', layout: { visibility: 'none' } }
  ];
  return {
    version: 8, name: 'abi-foss-hybrid', sources, layers,
    // Keyless community PBF glyph server. Any symbol layer using `text-field`
    // (e.g. OSINT flight callsigns) needs this to resolve fonts — without it
    // MapLibre fails the WHOLE symbol layer (icon included, not just the
    // text), so aircraft were fetched but silently never drawn on the map.
    glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf'
  };
}

function buildStartupStyle() {
  const params = new URLSearchParams(window.location.search);
  const isLocalHarness = ['localhost', '127.0.0.1'].includes(window.location.hostname) &&
    params.get('e2e_rendered_layers') === '1';
  if (!isLocalHarness) return buildBaseStyle();
  // The rendered-layer regression exercises only application overlays. A
  // background-only base keeps it deterministic and prevents remote imagery
  // availability from delaying MapLibre's initial `load` event in CI.
  return {
    version: 8,
    name: 'abi-rendered-layer-test',
    sources: {},
    layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#0b1220' } }]
  };
}

// ---------------------------------------------------------------------------
// Vector overlay sources & layers.

const GEOJSON_SOURCES = [
  'contours-source', 'canopy-source', 'trees-source', 'solar-suit-source',
  'solar-panels-source', 'buildings-source', 'ranked-parcels-source',
  'ranked-markers-source', 'selected-property-source', 'manual-pools-source',
  'pools-source', 'pools-points-source', 'draw-source', 'draw-guide-source',
  'locate-source'
];

function addOverlaySourcesAndLayers(map) {
  GEOJSON_SOURCES.forEach(id => {
    if (!map.getSource(id)) map.addSource(id, { type: 'geojson', data: EMPTY_FC });
  });

  map.addLayer({
    id: 'contours-line', type: 'line', source: 'contours-source',
    paint: { 'line-color': '#60a5fa', 'line-width': 1.35, 'line-opacity': 0.6, 'line-dasharray': [4, 5] }
  });
  map.addLayer({
    id: 'canopy-fill', type: 'fill', source: 'canopy-source',
    paint: { 'fill-color': '#10b981', 'fill-opacity': 0.5, 'fill-outline-color': '#047857' }
  });
  map.addLayer({
    id: 'trees-fill', type: 'fill', source: 'trees-source',
    paint: { 'fill-color': '#10b981', 'fill-opacity': 0.4, 'fill-outline-color': '#047857' }
  });
  map.addLayer({
    id: 'solar-suit-fill', type: 'fill', source: 'solar-suit-source',
    paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.42 }
  });
  map.addLayer({
    id: 'solar-panels-fill', type: 'fill', source: 'solar-panels-source',
    paint: { 'fill-color': '#facc15', 'fill-opacity': 0.28 }
  });
  map.addLayer({
    id: 'solar-panels-line', type: 'line', source: 'solar-panels-source',
    paint: { 'line-color': '#facc15', 'line-width': 2 }
  });
  map.addLayer({
    id: 'buildings-fill', type: 'fill', source: 'buildings-source',
    paint: { 'fill-color': '#ec4899', 'fill-opacity': 0.32 }
  });
  map.addLayer({
    id: 'buildings-outline', type: 'line', source: 'buildings-source',
    paint: { 'line-color': '#db2777', 'line-width': 1.5 }
  });
  // True 3D extrusions from OSM footprints (replaces the old CSS-skew fake).
  map.addLayer({
    id: 'buildings-3d-extrusion',
    type: 'fill-extrusion',
    source: 'buildings-source',
    layout: { visibility: 'none' },
    paint: {
      'fill-extrusion-color': '#db2777',
      'fill-extrusion-height': ['coalesce', ['get', 'height'], ['*', ['get', 'levels'], 3.2], 5.5],
      'fill-extrusion-base': 0,
      'fill-extrusion-opacity': 0.85
    }
  });
  map.addLayer({
    id: 'ranked-parcels-fill', type: 'fill', source: 'ranked-parcels-source',
    paint: { 'fill-color': '#10b981', 'fill-opacity': 0.08 }
  });
  map.addLayer({
    id: 'ranked-parcels-line', type: 'line', source: 'ranked-parcels-source',
    paint: { 'line-color': '#10b981', 'line-width': 2, 'line-dasharray': [2, 2] }
  });
  map.addLayer({
    id: 'ranked-markers-circle', type: 'circle', source: 'ranked-markers-source',
    paint: {
      'circle-radius': 8, 'circle-color': '#10b981',
      'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2, 'circle-opacity': 0.85
    }
  });
  map.addLayer({
    id: 'selected-property-fill', type: 'fill', source: 'selected-property-source',
    paint: {
      'fill-color': ['match', ['get', 'role'], 'building', '#64748b', 'backyard', '#4ade80', '#10b981'],
      'fill-opacity': ['match', ['get', 'role'], 'building', 0.6, 'backyard', 0.3, 0.15]
    }
  });
  map.addLayer({
    id: 'selected-property-line', type: 'line', source: 'selected-property-source',
    paint: {
      'line-color': ['match', ['get', 'role'], 'building', '#475569', 'backyard', '#22c55e', '#059669'],
      'line-width': ['match', ['get', 'role'], 'building', 1, 'backyard', 2, 3]
    }
  });
  map.addLayer({
    id: 'manual-pools-fill', type: 'fill', source: 'manual-pools-source',
    paint: { 'fill-color': '#0ea5e9', 'fill-opacity': 0.32 }
  });
  map.addLayer({
    id: 'manual-pools-line', type: 'line', source: 'manual-pools-source',
    paint: { 'line-color': '#0ea5e9', 'line-width': 3 }
  });
  map.addLayer({
    id: 'pools-fill', type: 'fill', source: 'pools-source',
    paint: { 'fill-color': '#ef4444', 'fill-opacity': 0.5 }
  });
  // Red→amber "target lock" outline: a soft amber glow beneath a crisp red edge.
  map.addLayer({
    id: 'pools-outline-glow', type: 'line', source: 'pools-source',
    paint: { 'line-color': '#f59e0b', 'line-width': 6, 'line-opacity': 0.35, 'line-blur': 3 }
  });
  map.addLayer({
    id: 'pools-outline', type: 'line', source: 'pools-source',
    paint: { 'line-color': '#ef4444', 'line-width': 2.5 }
  });
  map.addLayer({
    id: 'pools-points', type: 'circle', source: 'pools-points-source',
    paint: {
      'circle-radius': 5,
      'circle-color': '#ef4444',
      'circle-opacity': 0.9,
      'circle-stroke-color': ['case', ['get', 'isDetected'], '#f59e0b', '#f97316'],
      'circle-stroke-width': 2
    }
  });
  map.addLayer({
    id: 'draw-polygon-fill', type: 'fill', source: 'draw-source',
    filter: ['==', ['get', 'role'], 'polygon'],
    paint: { 'fill-color': '#10b981', 'fill-opacity': 0.15 }
  });
  map.addLayer({
    id: 'draw-polygon-line', type: 'line', source: 'draw-source',
    filter: ['==', ['get', 'role'], 'polygon'],
    paint: { 'line-color': '#10b981', 'line-width': 3 }
  });
  map.addLayer({
    id: 'draw-preview-line', type: 'line', source: 'draw-source',
    filter: ['==', ['get', 'role'], 'preview'],
    paint: { 'line-color': '#22d3ee', 'line-width': 2, 'line-dasharray': [3, 2] }
  });
  map.addLayer({
    id: 'draw-guide-line', type: 'line', source: 'draw-guide-source',
    paint: { 'line-color': '#38bdf8', 'line-width': 1.75, 'line-dasharray': [2, 2], 'line-opacity': 0.9 }
  });
  map.addLayer({
    id: 'draw-vertices', type: 'circle', source: 'draw-source',
    filter: ['==', ['get', 'role'], 'vertex'],
    paint: {
      'circle-radius': 5, 'circle-color': '#ffffff',
      'circle-stroke-color': '#10b981', 'circle-stroke-width': 2
    }
  });
  map.addLayer({
    id: 'locate-circle', type: 'circle', source: 'locate-source',
    paint: {
      'circle-radius': 9, 'circle-color': '#3b82f6',
      'circle-stroke-color': '#ffffff', 'circle-stroke-width': 3, 'circle-opacity': 0.95
    }
  });
}

// ---------------------------------------------------------------------------
// Popups (hover tooltip + click intelligence card for pools).

let hoverPopup = null;
let clickPopup = null;

function wireInteractions(map) {
  hoverPopup = new maplibregl.Popup({
    closeButton: false, closeOnClick: false, className: 'ml-tooltip', maxWidth: '320px', offset: 10
  });
  clickPopup = new maplibregl.Popup({ className: 'ml-card', maxWidth: '300px', offset: 10 });

  const tooltipLayers = ['pools-fill', 'pools-points', 'buildings-fill', 'solar-panels-fill', 'manual-pools-fill', 'ranked-markers-circle', 'ranked-parcels-fill'];
  tooltipLayers.forEach(layerId => {
    map.on('mousemove', layerId, event => {
      if (state.drawingMode) return;
      const feature = event.features && event.features[0];
      const tooltip = feature?.properties?.tooltip;
      if (!tooltip) return;
      map.getCanvas().style.cursor = 'pointer';
      hoverPopup.setLngLat(event.lngLat).setHTML(`<div class="ml-tooltip-body">${escapeHtml(tooltip)}</div>`).addTo(map);
    });
    map.on('mouseleave', layerId, () => {
      map.getCanvas().style.cursor = '';
      hoverPopup.remove();
    });
  });

  const poolClick = event => {
    if (state.drawingMode || state.isPegmanActive) return;
    const feature = event.features && event.features[0];
    if (!feature) return;
    const cached = state.renderedPoolIndex.get(feature.properties.uid);
    if (!cached) return;
    openPoolPopup(map, event.lngLat, cached, feature.properties.sourceLabel, feature.properties.confidence);
  };
  map.on('click', 'pools-fill', poolClick);
  map.on('click', 'pools-points', poolClick);

  const rankedClick = event => {
    if (state.drawingMode) return;
    const feature = event.features && event.features[0];
    const propertyId = feature?.properties?.property_id;
    if (propertyId === undefined || propertyId === null) return;
    if (state.isPegmanActive) {
      import('./ui.js').then(m => m.loadStreetView(event.lngLat.lat, event.lngLat.lng));
      import('./controls.js').then(m => m.togglePegmanMode(false));
      return;
    }
    import('../services/api.js').then(m => m.selectPropertyById(propertyId));
  };
  map.on('click', 'ranked-markers-circle', rankedClick);
  map.on('click', 'ranked-parcels-fill', rankedClick);
}

function openPoolPopup(map, lngLat, feature, sourceLabel, confidence) {
  const center = featureCenter(feature);
  if (!center) return;
  const href = `https://www.google.com/maps/@?api=1&map_action=map&center=${center.lat},${center.lon}&zoom=20`;
  // Jurisdiction-correct compliance link — pool registration is state law.
  const registerLink = poolRegisterLinkHtml(center.lat, center.lon);
  const devMode = document.getElementById('dev-mode-toggle')?.checked;
  const feedback = devMode
    ? `<div class="pool-feedback">
         <button type="button" data-fb="correct" title="Confirm this is a real pool">✓ Correct</button>
         <button type="button" data-fb="wrong" title="Not a pool — remove and store as a negative label">✗ Not a pool</button>
       </div>`
    : '';
  const html = `
    <div class="pool-popup">
      <strong>${escapeHtml(sourceLabel)}</strong><br>
      ${escapeHtml(String(confidence))}% confidence<br>
      <a href="${href}" target="_blank" rel="noopener">Open in Google Maps</a>${registerLink}
      ${feedback}
    </div>
  `;
  clickPopup.setLngLat(lngLat).setHTML(html).addTo(map);
  const root = clickPopup.getElement();
  if (!root) return;
  root.querySelectorAll('[data-fb]').forEach(button => {
    button.onclick = () => {
      button.disabled = true;
      import('../services/api.js')
        .then(m => m.sendPoolFeedback(button.dataset.fb, feature))
        .then(() => { try { clickPopup.remove(); } catch (e) {} })
        .catch(err => { button.disabled = false; console.error('feedback failed', err); });
    };
  });
}

// ---------------------------------------------------------------------------

export function initMap() {
  if (!window.maplibregl) {
    throw new Error('MapLibre GL failed to load — check network access to unpkg.com');
  }

  const select = document.getElementById('map-basemap');
  const initialBasemap = select?.value || 'au-aerial';
  const startupParams = new URLSearchParams(window.location.search);
  const requestedArea = (startupParams.get('area') || startupParams.get('lga') || '').trim();
  // NOTE: Number(null) === 0 and Number('') === 0 (NOT NaN). Parsing absent params
  // directly would make a paramless URL look like a request for lat=0,lon=0,zoom=0,
  // which centers the map on "null island" in the Atlantic (dark ocean → black map).
  // Only parse when the param is actually present and non-empty.
  const numParam = (...keys) => {
    for (const k of keys) {
      const raw = startupParams.get(k);
      if (raw !== null && raw.trim() !== '') return Number(raw);
    }
    return NaN;
  };
  const requestedLat = numParam('lat');
  const requestedLon = numParam('lon', 'lng');
  const requestedZoom = numParam('zoom');
  // URL zoom is a navigation input, not a detector setting. Honour the full map
  // range; detector calls independently enforce their own close-zoom threshold.
  const requestedZoomLevel = normalizeStartupZoom(requestedZoom);
  // A real target is finite, in-range, and not "null island" (0,0).
  const isSaneLatLng = (lat, lng) =>
    Number.isFinite(lat) && Number.isFinite(lng) &&
    Math.abs(lat) <= 90 && Math.abs(lng) <= 180 &&
    !(Math.abs(lat) < 0.5 && Math.abs(lng) < 0.5);
  let deferredStartupFetch = false;
  let deferredAreaQuery = '';

  state.map = new maplibregl.Map({
    container: 'map',
    style: buildStartupStyle(),
    center: [153.025, -27.248],
    zoom: 16,                       // ui zoom 17 in the app's Leaflet-calibrated terms
    maxZoom: 21,                    // ui zoom 22
    minZoom: 2,
    pitch: 0,
    bearing: 0,
    maxPitch: 70,
    antialias: true,                // smooth fill-extrusion edges
    doubleClickZoom: false,
    attributionControl: { compact: true }
  });
  window.__abiMap = state.map;      // console access for diagnostics

  // Restore map location from localStorage or default to Geolocation on startup
  let savedCenter = null;
  let savedZoom = null;
  try {
    // abi_* is the current key; qld_backyard_* is read for migration from pre-rename builds.
    savedCenter = localStorage.getItem('abi_map_center') || localStorage.getItem('qld_backyard_map_center');
    savedZoom = localStorage.getItem('abi_map_zoom') || localStorage.getItem('qld_backyard_map_zoom');
  } catch (e) {
    console.warn('[MAP] localStorage unavailable:', e);
  }
  let restored = false;

  if (Number.isFinite(requestedLat) && Number.isFinite(requestedLon)) {
    setView(state.map, requestedLat, requestedLon, requestedZoomLevel);
    restored = true;
    deferredStartupFetch = true;
  } else if (requestedArea) {
    restored = true;
    deferredAreaQuery = requestedArea;
  }

  if (!restored && savedCenter && savedZoom) {
    try {
      const center = JSON.parse(savedCenter);
      const zoom = parseInt(savedZoom, 10);
      if (Array.isArray(center) && center.length === 2 && !isNaN(zoom) && isSaneLatLng(center[0], center[1])) {
        setView(state.map, center[0], center[1], zoom);
        restored = true;
        console.log(`[MAP] Restored map location from localStorage:`, center, zoom);
      } else if (Array.isArray(center) && !isSaneLatLng(center[0], center[1])) {
        console.warn('[MAP] Ignoring invalid saved map location:', center);
        try {
          localStorage.removeItem('abi_map_center');
          localStorage.removeItem('abi_map_zoom');
          localStorage.removeItem('qld_backyard_map_center');
          localStorage.removeItem('qld_backyard_map_zoom');
        } catch (e) {}
      }
    } catch (e) {
      console.warn("Failed to parse saved map location from localStorage:", e);
    }
  }

  if (!restored) {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          console.log(`[MAP] Startup Geolocation success via Browser: ${lat}, ${lon}`);
          setView(state.map, lat, lon, 17);
          try {
            localStorage.setItem('abi_map_center', JSON.stringify([lat, lon]));
            localStorage.setItem('abi_map_zoom', '17');
          } catch (e) {}
          updateAILayers();
        },
        (err) => {
          console.warn("[MAP] Browser Geolocation failed/denied, using default view:", err);
          setView(state.map, -27.248, 153.025, 17);
          window.setTimeout(updateAILayers, 0);
        },
        { enableHighAccuracy: false, timeout: 3000, maximumAge: 600000 }
      );
    } else {
      setView(state.map, -27.248, 153.025, 17);
      window.setTimeout(updateAILayers, 0);
    }
  } else if (!deferredStartupFetch && !deferredAreaQuery) {
    updateAILayers();
  }

  async function applyStartupAreaQuery(area) {
    try {
      const res = await fetch(`/api/geocode?q=${encodeURIComponent(area)}`);
      const payload = await res.json().catch(() => ({}));
      const item = Array.isArray(payload.items) ? payload.items[0] : null;
      const lat = Number(item?.lat);
      const lon = Number(item?.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        setView(state.map, lat, lon, requestedZoomLevel);
        try {
          localStorage.setItem('abi_map_center', JSON.stringify([lat, lon]));
          localStorage.setItem('abi_map_zoom', String(requestedZoomLevel));
        } catch (e) {}
        updateAILayers();
        return;
      }
    } catch (error) {
      console.warn("[MAP] Startup area geocoding failed:", error);
    }
    updateAILayers();
  }

  state.map.on('load', () => {
    addOverlaySourcesAndLayers(state.map);
    wireInteractions(state.map);
    applyBasemapVisibility(initialBasemap);
    state.mapReady = true;
    loadPersistentPoolCache();
    state.lastAIRenderKey = null;
    renderAILayers();
    console.log('[MAP] MapLibre style loaded — vector pipeline live');
  });

  state.map.on('error', event => {
    // Raster tiles 404 routinely outside gov coverage — only surface real errors.
    const message = event?.error?.message || '';
    if (message && !/status.*(404|400)/i.test(message)) {
      console.warn('[MAP] MapLibre error:', message);
    }
  });

  // --- One-shot rendering diagnostic. Run __poolDebug() in the browser console. ---
  window.__poolDebug = function () {
    try {
      const cached = (state.cachedPoolsAndTrees && state.cachedPoolsAndTrees.pools) || [];
      const poolsSource = state.map.getSource('pools-source');
      const info = {
        zoom: uiZoom(state.map),
        styleZoom: state.map.getZoom(),
        pitch: state.map.getPitch(),
        bearing: state.map.getBearing(),
        globalPools: state.globalPools.size,
        cachedPools: cached.length,
        renderedPoolIndex: state.renderedPoolIndex.size,
        solarPanels: state.globalSolarPanels.size,
        layerPoolsChecked: !!(document.getElementById('layer-pools') || {}).checked,
        basemap: (document.getElementById('map-basemap') || {}).value,
        isTilted: !!state.isTilted,
        mapReady: state.mapReady,
        hasPoolsSource: Boolean(poolsSource)
      };
      const c = state.map.getCenter();
      setSourceData(state.map, 'locate-source', featureCollection([
        pointFeature({ lat: c.lat, lon: c.lng }, { tooltip: 'debug marker' })
      ]));
      console.log('[POOL DEBUG]', info);
      return info;
    } catch (e) {
      console.error('[POOL DEBUG ERROR]', e);
    }
  };

  state.map.on('moveend', () => {
    const zoom = uiZoom(state.map);
    state.lastFetchZoom = zoom;

    // Save map center and zoom to localStorage for default open (skip insane
    // centers like [0,0] so we never persist a dark-ocean location).
    try {
      const center = state.map.getCenter();
      if (isSaneLatLng(center.lat, center.lng)) {
        localStorage.setItem('abi_map_center', JSON.stringify([center.lat, center.lng]));
        localStorage.setItem('abi_map_zoom', String(Math.round(zoom)));
      }
    } catch (e) {
      console.warn("Failed to save map position to localStorage:", e);
    }
  });

  state.map.on('zoomend', () => {
    import('../services/api.js').then(m => m.updateVisibilityByZoom());
  });

  // Wire up AI Layer Checkbox interactions
  document.getElementById('layer-pools').addEventListener('change', handleCheckboxChange);
  document.getElementById('layer-buildings').addEventListener('change', handleCheckboxChange);
  document.getElementById('layer-trees').addEventListener('change', handleCheckboxChange);
  document.getElementById('layer-3d').addEventListener('change', handleCheckboxChange);
  document.getElementById('feature-layer-retry').addEventListener('click', () => {
    const coords = state.polygon?.vertices?.map(point => [point.lng, point.lat]) || [];
    if (coords.length >= 3) {
      scanSelectedAreaForAIDetections({ forceRefresh: true });
    }
  });

  if (deferredAreaQuery) {
    window.setTimeout(() => applyStartupAreaQuery(deferredAreaQuery), 0);
  } else if (deferredStartupFetch) {
    window.setTimeout(updateAILayers, 0);
  }

  // Overlay toggles only render data from the active selected-area analysis.
  const overlayToggled = () => { renderAILayers(); };
  document.getElementById('overlay-canopy').addEventListener('change', overlayToggled);
  document.getElementById('overlay-solar').addEventListener('change', overlayToggled);
  document.getElementById('overlay-solar-panels')?.addEventListener('change', overlayToggled);
  document.getElementById('overlay-drainage').addEventListener('change', event => {
    if (event.currentTarget.checked) fetchTerrainGridInView();
    overlayToggled();
  });

  // View perspective: a true WebGL camera move, not a CSS skew.
  document.getElementById('aerial-timeline').addEventListener('change', () => {
    const val = document.getElementById('aerial-timeline').value;
    if (val === 'oblique') {
      state.map.easeTo({ pitch: 30, duration: 800, easing: t => t * (2 - t) });
    } else if (!state.isTilted) {
      state.map.easeTo({ pitch: 0, bearing: 0, duration: 800, easing: t => t * (2 - t) });
    }
  });

  updateAILayers();
}

// ---------------------------------------------------------------------------
// Basemap switching (visibility toggles on the single persistent style).

function applyBasemapVisibility(value) {
  if (!state.mapReady) return;
  const group = BASEMAP_GROUPS[value] || BASEMAP_GROUPS['au-aerial'];
  setLayersVisibility(state.map, ALL_BASEMAP_LAYERS, false);
  setLayersVisibility(state.map, group, true);
}

export function showGoogleOverlayMap() {
  const mapEl = document.getElementById('map');
  document.querySelectorAll('.google-map-frame').forEach(frame => {
    if (frame.dataset.streetView !== 'true') {
      frame.style.display = 'none';
      frame.src = '';
      delete frame.dataset.mapUrl;
    }
  });
  mapEl.style.display = 'block';
  mapEl.style.visibility = '';
  try { state.map.resize(); } catch (e) {}
}

export function switchBasemap(val) {
  if (!state.map) return;
  if (state.mapReady) {
    applyBasemapVisibility(val);
  } else {
    state.map.once('load', () => applyBasemapVisibility(val));
  }

  // Hide background iframes if they are visible (unless in street view)
  document.querySelectorAll('.google-map-frame').forEach(frame => {
    if (frame.dataset.streetView !== 'true') {
      frame.style.display = 'none';
      frame.src = '';
      delete frame.dataset.mapUrl;
    }
  });
}

export function fetchAIData(options = {}) {
  if (document.getElementById('overlay-drainage')?.checked) fetchTerrainGridInView();
  if (options.source !== 'selected-area-ai-scan') return;
  if (uiZoom(state.map) < 15) {
    updateAILayers();
    return;
  }
  fetchAIDataInView(options);
}

const AI_SOURCE_IDS = [
  'pools-source', 'pools-points-source', 'buildings-source', 'trees-source',
  'canopy-source', 'solar-suit-source', 'solar-panels-source', 'contours-source'
];

export function updateAILayers() {
  if (uiZoom(state.map) < 15) {
    if (state.mapReady) {
      AI_SOURCE_IDS.forEach(id => setSourceData(state.map, id, EMPTY_FC));
    }
    state.renderedPoolIndex.clear();
    state.cachedPoolsAndTrees = null;
    state.cachedBuildings = null;
    state.lastAIRenderKey = null;
    return;
  }
  renderAILayers();
}

export function handleCheckboxChange() {
  // Every explicit selected-area analysis loads all core categories. Checkboxes
  // therefore only change visibility and never launch out-of-band inference.
  renderAILayers();
  if (uiZoom(state.map) < 15) {
    const center = state.map.getCenter();
    state.map.flyTo({ center: [center.lng, center.lat], zoom: 15 });  // style z15 = ui z16
    return;
  }
}

// True for pools produced by any detector (satellite YOLO/CV, property records) rather
// than traced in OpenStreetMap. Used to guarantee detected pools always render.
function isDetectedPoolTags(tags = {}) {
  const source = String(tags?.['detector:source'] || '');
  const yoloRole = String(tags?.['detector:yolo'] || '');
  const method = String(tags?.['detector:method'] || '');
  return source.startsWith('satellite') || source === 'yolo-cv' || source === 'local-property-record' ||
    yoloRole === 'yes' || yoloRole === 'validator' || method.includes('pool-trained-yolo');
}

function featureCenter(feature) {
  if (!feature) return null;
  if (feature._center) return feature._center;

  let center = null;
  if (feature.type === 'node') {
    const lat = Number(feature.lat);
    const lon = Number(feature.lon);
    center = Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
  } else {
    center = geometryCentroid(feature.geometry);
  }

  if (center) {
    Object.defineProperty(feature, '_center', {
      value: center,
      writable: true,
      enumerable: false,
      configurable: true
    });
  }
  return center;
}

function buildTreeGrid(trees) {
  const grid = {};
  const cellSize = 0.0005; // ~50m cells

  for (const tree of trees) {
    const center = featureCenter(tree);
    if (!center) continue;

    const latBin = Math.floor(center.lat / cellSize);
    const lonBin = Math.floor(center.lon / cellSize);
    const key = `${latBin},${lonBin}`;

    if (!grid[key]) {
      grid[key] = [];
    }
    grid[key].push(tree);
  }
  return { grid, cellSize };
}

function nearbyTreeCountWithGrid(center, treeGridData, radiusMeters = 25) {
  if (!center || !treeGridData) return 0;
  const { grid, cellSize } = treeGridData;
  const latitudeScale = 111320;
  const longitudeScale = latitudeScale * Math.cos(center.lat * Math.PI / 180);

  const latBin = Math.floor(center.lat / cellSize);
  const lonBin = Math.floor(center.lon / cellSize);

  let count = 0;
  const radiusSq = radiusMeters * radiusMeters;

  for (let dLat = -1; dLat <= 1; dLat++) {
    for (let dLon = -1; dLon <= 1; dLon++) {
      const key = `${latBin + dLat},${lonBin + dLon}`;
      const cellTrees = grid[key];
      if (!cellTrees) continue;

      for (const tree of cellTrees) {
        const treeCenter = featureCenter(tree);
        if (!treeCenter) continue;

        const northing = (treeCenter.lat - center.lat) * latitudeScale;
        const easting = (treeCenter.lon - center.lon) * longitudeScale;
        if (northing * northing + easting * easting <= radiusSq) {
          count += 1;
        }
      }
    }
  }
  return count;
}

function collectVisibleFeatures(features, bounds, cap, validator = null) {
  const mapCenter = state.map.getCenter();
  const candidates = [];
  const south = bounds.getSouth();
  const west = bounds.getWest();
  const north = bounds.getNorth();
  const east = bounds.getEast();

  for (const feature of features || []) {
    const center = featureCenter(feature);
    if (!center) continue;

    let visible = false;
    if (feature.type === 'node') {
      visible = center.lat >= south && center.lat <= north && center.lon >= west && center.lon <= east;
    } else {
      const extent = geometryExtent(feature.geometry);
      visible = extent && extent.east >= west && extent.west <= east &&
                extent.north >= south && extent.south <= north;
    }

    if (!visible || (validator && !validator(feature))) continue;
    const distance = (center.lat - mapCenter.lat) ** 2 + (center.lon - mapCenter.lng) ** 2;
    candidates.push({ feature, distance });
  }
  candidates.sort((left, right) => left.distance - right.distance);
  return candidates.slice(0, cap).map(item => item.feature);
}

function renderSignature(bounds) {
  const ids = ['layer-pools', 'layer-buildings', 'layer-trees', 'layer-3d', 'overlay-canopy', 'overlay-solar', 'overlay-solar-panels', 'overlay-drainage'];
  const flags = ids.map(id => document.getElementById(id)?.checked ? '1' : '0').join('');
  const basemap = document.getElementById('map-basemap')?.value || 'au-aerial';
  return [
    state.aiFeatureRevision,
    Math.round(uiZoom(state.map) * 10) / 10,
    bounds.getSouth().toFixed(4), bounds.getWest().toFixed(4),
    bounds.getNorth().toFixed(4), bounds.getEast().toFixed(4),
    flags, basemap, state.isTilted ? 'T' : 'F'
  ].join(':');
}

export function renderAILayers() {
  if (state.aiRenderFrame !== null) return;
  const schedule = window.requestAnimationFrame || (callback => window.setTimeout(callback, 0));
  state.aiRenderFrame = schedule(() => {
    state.aiRenderFrame = null;
    renderAILayersNow();
  });
}

function poolSourceLabel(feature) {
  const tags = feature.tags || {};
  const detectorSource = tags['detector:source'] || '';
  const detectorMethod = tags['detector:method'] || '';
  const yoloRole = tags['detector:yolo'] || '';
  const isYolo = yoloRole === 'yes' || yoloRole === 'validator' || detectorSource === 'satellite-yolo' || detectorMethod === 'pool-trained-yolo';
  const currentModel = String(tags['detector:model'] || 'pool-trained weights');
  return detectorSource === 'osm-verified' || detectorSource === 'public-mapped'
    ? 'Mapped pool'
    : detectorSource === 'satellite-yolo'
      ? `Pool-trained YOLO (${currentModel})`
    : detectorSource === 'satellite-cv'
      ? 'Satellite pool detection'
    : detectorSource === 'local-property-record'
      ? 'Recorded existing pool'
      : isYolo
        ? `Detected pool · ${currentModel}`
        : 'Detected pool';
}

// Fixed-size ~12 m box around a point (used for degenerate detections so a
// detected pool can never be invisible).
function centerBoxRing(center) {
  const latOffset = 0.00006;
  const lonOffset = 0.00007;
  return [
    { lat: center.lat - latOffset, lon: center.lon - lonOffset },
    { lat: center.lat - latOffset, lon: center.lon + lonOffset },
    { lat: center.lat + latOffset, lon: center.lon + lonOffset },
    { lat: center.lat + latOffset, lon: center.lon - lonOffset }
  ];
}

function renderAILayersNow() {
  if (!state.mapReady) return;

  const visibleBounds = boundsFromMap(state.map);
  const signature = renderSignature(visibleBounds);
  if (signature === state.lastAIRenderKey) return;
  state.lastAIRenderKey = signature;

  const renderBounds = visibleBounds.pad(0.12);
  const zoom = uiZoom(state.map);
  // A real pool footprint is ~10–18 m across, which shrinks below one pixel once you
  // zoom out far enough to view a large area scan — so detections "disappear" even though
  // they're loaded. Below this zoom we draw each detection as a constant screen-size dot
  // so it stays visible; at closer zoom we draw the true footprint polygon.
  const drawPoolsAsDots = zoom < 17;
  const pools = state.cachedPoolsAndTrees?.pools || [];
  const trees = state.cachedPoolsAndTrees?.trees || [];
  const buildings = state.cachedBuildings?.buildings || [];

  // Detected pools ALWAYS pass — a satellite detection with odd/degenerate geometry
  // still gets drawn (worst case as a small centroid box) instead of silently
  // vanishing. Only mapped OSM ways still require a sane traced polygon.
  const visiblePools = collectVisibleFeatures(
    pools,
    renderBounds,
    3000,
    feature => feature.type === 'node' ||
      isDetectedPoolTags(feature.tags) ||
      isValidPoolGeometry(feature.geometry, feature.tags)
  );
  const visibleBuildings = collectVisibleFeatures(buildings, renderBounds, zoom >= 18 ? 800 : 500, feature => isValidBuildingGeometry(feature.geometry));
  const visibleTrees = collectVisibleFeatures(trees, renderBounds, zoom >= 18 ? 1200 : 800);

  let treeGrid = null;
  if (document.getElementById('overlay-solar').checked) {
    treeGrid = buildTreeGrid(visibleTrees);
  }

  // ---- Pools → polygon + point FeatureCollections ----
  const poolPolygons = [];
  const poolPoints = [];
  state.renderedPoolIndex.clear();
  if (document.getElementById('layer-pools').checked) {
    visiblePools.forEach((feature, index) => {
      const confidence = Math.round(poolConfidence(feature) * 100);
      const detectorSource = feature.tags?.['detector:source'] || '';
      const isDetected = isDetectedPoolTags(feature.tags) || detectorSource === 'satellite-cv';
      const sourceLabel = poolSourceLabel(feature);
      const uid = `pool-${index}`;
      state.renderedPoolIndex.set(uid, feature);

      const center = featureCenter(feature);
      const baseProps = { uid, sourceLabel, confidence, isDetected };

      if (isDetected && drawPoolsAsDots && center) {
        poolPoints.push(pointFeature(center, {
          ...baseProps,
          tooltip: `${sourceLabel} · ${confidence}% confidence`
        }));
        return;
      }

      if (feature.type === 'node') {
        const tooltip = isDetected
          ? `${sourceLabel} · ${confidence}% confidence`
          : `Approximate mapped aquatic point · ${confidence}% tag confidence`;
        if (isDetected && center) {
          poolPolygons.push(polygonFeature(centerBoxRing(center), { ...baseProps, tooltip }));
        } else if (center) {
          poolPoints.push(pointFeature(center, { ...baseProps, tooltip }));
        }
        return;
      }

      const geometry = normalizeGeometry(feature.geometry);
      const area = geometryAreaSqm(geometry);
      if (isDetected) {
        const tooltip = area > 0
          ? `${sourceLabel} · ${confidence}% confidence · ${area.toFixed(0)} m²`
          : `${sourceLabel} · ${confidence}% confidence`;
        if (geometry.length >= 3) {
          // Segmented outline: draw the detector's actual pool shape.
          poolPolygons.push(polygonFeature(geometry, { ...baseProps, tooltip }));
        } else {
          const extent = geometryExtent(geometry);
          if (extent) {
            poolPolygons.push(polygonFeature([
              { lat: extent.south, lon: extent.west },
              { lat: extent.south, lon: extent.east },
              { lat: extent.north, lon: extent.east },
              { lat: extent.north, lon: extent.west }
            ], { ...baseProps, tooltip }));
          } else if (center) {
            poolPolygons.push(polygonFeature(centerBoxRing(center), { ...baseProps, tooltip }));
          }
        }
      } else if (geometry.length >= 3) {
        const tooltip = `Mapped swimming pool · ${area.toFixed(0)} m² · ${confidence}% tag confidence`;
        poolPolygons.push(polygonFeature(geometry, { ...baseProps, tooltip }));
      }
    });
  }
  setSourceData(state.map, 'pools-source', featureCollection(poolPolygons));
  setSourceData(state.map, 'pools-points-source', featureCollection(poolPoints));

  // ---- Buildings (2D fill + 3D extrusion share one source) ----
  const buildingFeatures = [];
  const buildingsChecked = document.getElementById('layer-buildings').checked;
  const meshChecked = document.getElementById('layer-3d').checked;
  if (buildingsChecked || meshChecked || state.isTilted) {
    visibleBuildings.forEach(feature => {
      const geometry = normalizeGeometry(feature.geometry);
      if (geometry.length < 3) return;
      // building tag value is community-edited OSM data — escaped at tooltip render.
      const kind = feature.tags?.building || 'structure';
      const area = geometryAreaSqm(geometry);
      const levels = Number.parseFloat(feature.tags?.['building:levels']);
      buildingFeatures.push(polygonFeature(geometry, {
        height: inferBuildingHeight(feature.tags),
        levels: Number.isFinite(levels) ? levels : null,
        tooltip: `${kind} footprint · ${area.toFixed(0)} m²`
      }));
    });
  }
  setSourceData(state.map, 'buildings-source', featureCollection(buildingFeatures));
  // Flat fill only when the 2D layer is on; extrusion when meshes/3D camera are on.
  const extrusionOn = (meshChecked || state.isTilted) && zoom >= 16;
  setLayersVisibility(state.map, ['buildings-fill'], buildingsChecked && !extrusionOn);
  setLayersVisibility(state.map, ['buildings-outline'], buildingsChecked);
  setLayersVisibility(state.map, ['buildings-3d-extrusion'], extrusionOn);

  // ---- Trees / canopy ----
  const treeFeatures = [];
  if (document.getElementById('layer-trees').checked) {
    visibleTrees.forEach(feature => {
      if (feature.type === 'node') {
        treeFeatures.push(circlePolygonFeature(
          { lat: feature.lat, lon: feature.lon }, inferCanopyRadius(feature.tags), {}, 14
        ));
      } else {
        const geometry = normalizeGeometry(feature.geometry);
        if (geometry.length >= 3 && geometryAreaSqm(geometry) >= 4) {
          treeFeatures.push(polygonFeature(geometry, {}));
        }
      }
    });
  }
  setSourceData(state.map, 'trees-source', featureCollection(treeFeatures));

  const canopyFeatures = [];
  if (document.getElementById('overlay-canopy').checked) {
    visibleTrees.forEach(feature => {
      if (feature.type === 'node') {
        canopyFeatures.push(circlePolygonFeature(
          { lat: feature.lat, lon: feature.lon }, inferCanopyRadius(feature.tags) * 1.15, {}, 14
        ));
      } else {
        const geometry = normalizeGeometry(feature.geometry);
        if (geometry.length >= 3 && geometryAreaSqm(geometry) >= 4) {
          canopyFeatures.push(polygonFeature(geometry, {}));
        }
      }
    });
  }
  setSourceData(state.map, 'canopy-source', featureCollection(canopyFeatures));

  // ---- Roof solar suitability heuristic ----
  const solarFeatures = [];
  if (document.getElementById('overlay-solar').checked) {
    visibleBuildings.forEach(feature => {
      const geometry = normalizeGeometry(feature.geometry);
      if (geometry.length < 3) return;
      const center = geometryCentroid(geometry);
      const count = nearbyTreeCountWithGrid(center, treeGrid);
      const score = estimateSolarScore(feature, count);
      const color = score >= 75 ? '#facc15' : score >= 55 ? '#f97316' : '#ef4444';
      solarFeatures.push(polygonFeature(geometry, { color, score }));
    });
  }
  setSourceData(state.map, 'solar-suit-source', featureCollection(solarFeatures));

  // ---- Real detected solar arrays (multi-class satellite model) ----
  const panelFeatures = [];
  if (document.getElementById('overlay-solar-panels')?.checked) {
    const detectedPanels = collectVisibleFeatures(
      Array.from(state.globalSolarPanels.values()), renderBounds, 600,
      feature => Array.isArray(feature.geometry) && feature.geometry.length >= 3
    );
    detectedPanels.forEach(feature => {
      const geometry = normalizeGeometry(feature.geometry);
      if (geometry.length < 3) return;
      const conf = Math.round(parseFloat(feature.tags['confidence:yolo'] || '0') * 100);
      panelFeatures.push(polygonFeature(geometry, {
        tooltip: `Solar array (detected) · ${conf}% · ~${feature.tags['solar:area_sqm'] || '?'} m²`
      }));
    });
  }
  setSourceData(state.map, 'solar-panels-source', featureCollection(panelFeatures));

  // ---- Drainage contours (computed in a background Web Worker) ----
  if (document.getElementById('overlay-drainage').checked) {
    if (state.terrainGrid && state.lastTerrainGridKey !== state.terrainGridKey) {
      state.lastTerrainGridKey = state.terrainGridKey;
      requestContourBuild(state.terrainGrid, state.terrainGridKey);
    }
  } else {
    setSourceData(state.map, 'contours-source', EMPTY_FC);
    state.lastTerrainGridKey = null;
  }

  renderManualPools();
}

// Manual precise pool traces (sky blue) live in their own source so drawing
// updates don't force a full AI re-render.
export function renderManualPools() {
  if (!state.mapReady) return;
  const features = state.manualPools.map(feature =>
    polygonFeature(feature.geometry, { tooltip: 'Precise existing pool trace' })
  );
  setSourceData(state.map, 'manual-pools-source', featureCollection(features));
}

// ---------------------------------------------------------------------------
// Elevation contours in a Web Worker: the grid marching-squares pass runs off
// the UI thread and returns a ready-to-use GeoJSON FeatureCollection.

function contoursToFeatureCollection(contours) {
  return featureCollection(contours.map(contour =>
    lineFeature(contour.points, { level: Math.round(contour.level * 10) / 10 })
  ));
}

function requestContourBuild(grid, key) {
  if (!state.contourWorkerBroken && typeof Worker !== 'undefined') {
    try {
      if (!state.contourWorker) {
        state.contourWorker = new Worker('/static/js/workers/elevation-worker.js', { type: 'module' });
        state.contourWorker.onmessage = event => {
          const { key: doneKey, featureCollection: fc } = event.data || {};
          if (doneKey !== state.terrainGridKey) return;  // superseded grid
          setSourceData(state.map, 'contours-source', fc || EMPTY_FC);
        };
        state.contourWorker.onerror = err => {
          console.warn('[MAP] contour worker failed — falling back to main thread:', err?.message || err);
          state.contourWorkerBroken = true;
          state.contourWorker = null;
          if (state.terrainGrid) {
            const contours = buildElevationContours(state.terrainGrid, 7);
            setSourceData(state.map, 'contours-source', contoursToFeatureCollection(contours));
          }
        };
      }
      state.contourWorker.postMessage({ grid: {
        latitudes: grid.latitudes, longitudes: grid.longitudes, elevations: grid.elevations
      }, levels: 7, key });
      return;
    } catch (e) {
      state.contourWorkerBroken = true;
      state.contourWorker = null;
    }
  }
  const contours = buildElevationContours(grid, 7);
  setSourceData(state.map, 'contours-source', contoursToFeatureCollection(contours));
}
