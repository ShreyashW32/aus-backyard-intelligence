// Palantir-style tactical OSINT data-fusion layer for MapLibre GL.
//
// Toggling the Tactical Intelligence layer polls the keyless backend feeds
// (/api/osint/*) and paints them with tactical symbology: live aircraft with
// trailing vectors, wildfire danger polygons, glowing CCTV nodes, pulsing
// incident/earthquake rings, and dotted semantic link lines from active nodes
// to the selected property. Everything is inverted to [lon, lat] at the source.
import { state } from '../state.js';
import { escapeHtml } from '../utils.js';
import {
  EMPTY_FC,
  addSvgImage,
  boundsFromMap,
  featureCollection,
  haversineMeters,
  lineFeature,
  pointFeature,
  setLayersVisibility,
  setSourceData
} from '../geo.js';

const OSINT_SOURCES = [
  'osint-fires-source', 'osint-incidents-source', 'osint-quakes-source',
  'osint-cameras-source', 'osint-flights-source', 'osint-trails-source',
  'osint-links-source'
];
const OSINT_LAYERS = [
  'osint-fires-fill', 'osint-fires-line', 'osint-fires-outline', 'osint-fires-point',
  'osint-incidents-ripple', 'osint-incidents-core',
  'osint-quakes-ripple', 'osint-quakes-core',
  'osint-links-line',
  'osint-cameras-glow', 'osint-cameras-icon',
  'osint-trails-line', 'osint-flights-icon'
];

const FLIGHT_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
  <path d="M12 2c-.7 0-1.2 1-1.2 2.4v4.2L2.5 13v1.8l8.3-2.4v4.6l-2.3 1.6v1.4L12 19l3.5 1v-1.4l-2.3-1.6v-4.6l8.3 2.4V13l-8.3-4.4V4.4C13.2 3 12.7 2 12 2z"
    fill="#38bdf8" stroke="#0f172a" stroke-width="0.6"/>
</svg>`;
const CAMERA_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22">
  <circle cx="12" cy="12" r="11" fill="rgba(16,185,129,0.18)"/>
  <path d="M4 8h9l2-2 5 2.2V16l-5 2.2-2-2H4z" fill="#10b981" stroke="#052e1b" stroke-width="0.6"/>
  <circle cx="8.5" cy="12" r="2.4" fill="#052e1b"/>
  <circle cx="8.5" cy="12" r="1.1" fill="#a7f3d0"/>
</svg>`;

let osintEnabled = false;
let selectedPropertyPoint = null;   // {lat, lon} for semantic link lines
let cameraClickPopup = null;
let flightClickPopup = null;
let fireClickPopup = null;
let incidentClickPopup = null;
let quakeClickPopup = null;

// ---------------------------------------------------------------------------
// Style setup: sources + tactical layers, installed once on demand.

async function ensureOsintStyle(map) {
  if (map.getSource('osint-fires-source')) return;

  OSINT_SOURCES.forEach(id => {
    if (!map.getSource(id)) map.addSource(id, { type: 'geojson', data: EMPTY_FC });
  });

  await Promise.all([
    addSvgImage(map, 'osint-flight', FLIGHT_ICON, 28),
    addSvgImage(map, 'osint-camera', CAMERA_ICON, 26)
  ]);

  // Wildfire danger polygons: translucent red fill, glowing orange stroke.
  map.addLayer({
    id: 'osint-fires-fill', type: 'fill', source: 'osint-fires-source',
    filter: ['==', ['geometry-type'], 'Polygon'],
    paint: { 'fill-color': '#ef4444', 'fill-opacity': 0.28 }
  });
  map.addLayer({
    id: 'osint-fires-outline', type: 'line', source: 'osint-fires-source',
    filter: ['==', ['geometry-type'], 'Polygon'],
    paint: { 'line-color': '#f97316', 'line-width': 6, 'line-opacity': 0.4, 'line-blur': 4 }
  });
  map.addLayer({
    id: 'osint-fires-line', type: 'line', source: 'osint-fires-source',
    paint: { 'line-color': '#f59e0b', 'line-width': 2 }
  });
  // Most state-feed fire reports are Points, not perimeter polygons — without
  // this layer they were fetched but silently invisible (fill/outline are
  // Polygon-only). Drawn as a glowing flame dot so it reads distinctly from
  // incidents (amber ring) and quakes (red ring).
  map.addLayer({
    id: 'osint-fires-point', type: 'circle', source: 'osint-fires-source',
    filter: ['==', ['geometry-type'], 'Point'],
    paint: {
      'circle-radius': 8, 'circle-color': '#f97316', 'circle-opacity': 0.85,
      'circle-stroke-color': '#7c2d12', 'circle-stroke-width': 1.5,
      'circle-blur': 0.3
    }
  });

  // Semantic link lines sit beneath the node markers.
  map.addLayer({
    id: 'osint-links-line', type: 'line', source: 'osint-links-source',
    paint: { 'line-color': '#a855f7', 'line-width': 1.4, 'line-dasharray': [2, 2], 'line-opacity': 0.85 }
  });

  // Incident + earthquake pulsing rings (radius/opacity animated in tick()).
  map.addLayer({
    id: 'osint-incidents-ripple', type: 'circle', source: 'osint-incidents-source',
    filter: ['==', ['geometry-type'], 'Point'],
    paint: { 'circle-radius': 10, 'circle-color': '#fbbf24', 'circle-opacity': 0.25, 'circle-stroke-color': '#fbbf24', 'circle-stroke-width': 1 }
  });
  map.addLayer({
    id: 'osint-incidents-core', type: 'circle', source: 'osint-incidents-source',
    filter: ['==', ['geometry-type'], 'Point'],
    paint: { 'circle-radius': 4, 'circle-color': '#fbbf24', 'circle-stroke-color': '#78350f', 'circle-stroke-width': 1 }
  });
  map.addLayer({
    id: 'osint-quakes-ripple', type: 'circle', source: 'osint-quakes-source',
    paint: {
      'circle-radius': ['+', 8, ['*', 3, ['coalesce', ['get', 'mag'], 1]]],
      'circle-color': '#ef4444', 'circle-opacity': 0.2,
      'circle-stroke-color': '#ef4444', 'circle-stroke-width': 1
    }
  });
  map.addLayer({
    id: 'osint-quakes-core', type: 'circle', source: 'osint-quakes-source',
    paint: {
      'circle-radius': ['+', 3, ['*', 1.4, ['coalesce', ['get', 'mag'], 1]]],
      'circle-color': '#ef4444', 'circle-stroke-color': '#450a0a', 'circle-stroke-width': 1
    }
  });

  // CCTV nodes: soft green glow + camera glyph.
  map.addLayer({
    id: 'osint-cameras-glow', type: 'circle', source: 'osint-cameras-source',
    paint: { 'circle-radius': 12, 'circle-color': '#10b981', 'circle-opacity': 0.18, 'circle-blur': 0.6 }
  });
  map.addLayer({
    id: 'osint-cameras-icon', type: 'symbol', source: 'osint-cameras-source',
    layout: { 'icon-image': 'osint-camera', 'icon-size': 0.85, 'icon-allow-overlap': true }
  });

  // Flight trails then live aircraft (rotated to heading).
  map.addLayer({
    id: 'osint-trails-line', type: 'line', source: 'osint-trails-source',
    paint: { 'line-color': '#38bdf8', 'line-width': 1.3, 'line-opacity': 0.5 }
  });
  map.addLayer({
    id: 'osint-flights-icon', type: 'symbol', source: 'osint-flights-source',
    layout: {
      'icon-image': 'osint-flight',
      'icon-size': 0.9,
      'icon-rotate': ['coalesce', ['get', 'track'], 0],
      'icon-rotation-alignment': 'map',
      // Keep the aircraft marker independent of remote font glyphs. A failed
      // callsign glyph request used to suppress the entire symbol layer, icon
      // included. Callsign remains available in the click card.
      'icon-allow-overlap': true
    }
  });

  wireOsintInteractions(map);
}

function wireOsintInteractions(map) {
  cameraClickPopup = new maplibregl.Popup({ className: 'osint-card', maxWidth: '260px', offset: 14 });
  flightClickPopup = new maplibregl.Popup({ className: 'osint-card', maxWidth: '260px', offset: 14 });

  map.on('click', 'osint-cameras-icon', event => {
    const feature = event.features && event.features[0];
    if (!feature) return;
    const p = feature.properties || {};
    const lat = event.lngLat.lat.toFixed(5);
    const lon = event.lngLat.lng.toFixed(5);
    const label = p.name || p.camera_type || p.surveillance || 'Surveillance camera';
    cameraClickPopup.setLngLat(event.lngLat).setHTML(`
      <div class="osint-cam-card">
        <div class="osint-cam-frame">
          <div class="osint-cam-scan"></div>
          <span class="osint-cam-rec">● REC</span>
          <span class="osint-cam-coords">${lat}, ${lon}</span>
          <span class="osint-cam-label">SIMULATED FEED · NO LIVE VIDEO</span>
        </div>
        <strong>${escapeHtml(label)}</strong>
        <div class="osint-cam-meta">
          <span>Type: ${escapeHtml(p.camera_type || p.surveillance || 'unknown')}</span>
          ${p.operator ? `<span>Operator: ${escapeHtml(p.operator)}</span>` : ''}
          ${p.zone ? `<span>Zone: ${escapeHtml(p.zone)}</span>` : ''}
          <span>Jurisdiction: ${escapeHtml(stateHint(event.lngLat.lat, event.lngLat.lng))}</span>
          <span class="osint-card-source">Source: OpenStreetMap surveillance node${p.id ? ` #${escapeHtml(String(p.id))}` : ''}</span>
        </div>
        ${p.id ? `<a class="osint-card-link" href="https://www.openstreetmap.org/node/${encodeURIComponent(p.id)}" target="_blank" rel="noopener noreferrer">View on OpenStreetMap ↗</a>` : ''}
      </div>
    `).addTo(map);
  });

  map.on('click', 'osint-flights-icon', event => {
    const feature = event.features && event.features[0];
    if (!feature) return;
    const p = feature.properties || {};
    const alt = p.altitude_m != null && p.altitude_m !== '' ? `${Math.round(Number(p.altitude_m))} m` : 'n/a';
    const vel = p.velocity_ms != null && p.velocity_ms !== '' ? `${Math.round(Number(p.velocity_ms) * 3.6)} km/h` : 'n/a';
    flightClickPopup.setLngLat(event.lngLat).setHTML(`
      <div class="osint-flight-card">
        <strong>${escapeHtml(p.callsign || p.icao24 || 'Unknown aircraft')}</strong>
        <div class="osint-flight-meta">
          <span>ICAO24: ${escapeHtml(p.icao24 || '—')}</span>
          ${p.aircraft_type ? `<span>Type: ${escapeHtml(p.aircraft_type)}</span>` : ''}
          <span>Altitude: ${escapeHtml(alt)}</span>
          <span>Velocity: ${escapeHtml(vel)}</span>
          <span>Heading: ${p.track != null && p.track !== '' ? Math.round(Number(p.track)) + '°' : 'n/a'}</span>
          <span>Squawk: ${escapeHtml(p.squawk || '—')}</span>
          ${p.origin_country ? `<span>Origin: ${escapeHtml(p.origin_country)}</span>` : ''}
          <span class="osint-card-source">Source: community ADS-B (adsb.lol / OpenSky)</span>
        </div>
        ${p.icao24 ? `<a class="osint-card-link" href="https://globe.adsbexchange.com/?icao=${encodeURIComponent(p.icao24)}" target="_blank" rel="noopener noreferrer">Track live on ADS-B Exchange ↗</a>` : ''}
      </div>
    `).addTo(map);
  });

  fireClickPopup = new maplibregl.Popup({ className: 'osint-card', maxWidth: '280px', offset: 14 });
  incidentClickPopup = new maplibregl.Popup({ className: 'osint-card', maxWidth: '280px', offset: 14 });
  quakeClickPopup = new maplibregl.Popup({ className: 'osint-card', maxWidth: '260px', offset: 14 });

  // NSW RFS / VIC emergency descriptions use `<br />` as a field separator
  // (e.g. "ALERT LEVEL: ... <br />LOCATION: ..."). escapeHtml neutralises the
  // whole string first (the feed is external, so anything else stays inert
  // text), then only that specific already-escaped `<br>` pattern is turned
  // back into a real line break — never the raw input.
  function formatFeedDescription(text) {
    return escapeHtml(String(text ?? '').slice(0, 240)).replace(/&lt;br\s*\/?&gt;/gi, '<br>');
  }

  function emergencyCardHtml(p, defaultKind) {
    const kind = p.kind === 'fire' ? 'Fire / burn' : (defaultKind || 'Incident');
    return `
      <div class="osint-emergency-card">
        <strong>${escapeHtml(p.title || kind)}</strong>
        <div class="osint-cam-meta">
          <span>Kind: ${escapeHtml(kind)}</span>
          ${p.category ? `<span>Category: ${escapeHtml(p.category)}</span>` : ''}
          ${p.status ? `<span>Status: ${escapeHtml(p.status)}</span>` : ''}
          ${p.updated ? `<span>Updated: ${escapeHtml(p.updated)}</span>` : ''}
          <span>Source: ${escapeHtml(p.source || 'State emergency feed')} (${escapeHtml((p.jurisdiction || '').toUpperCase())})</span>
        </div>
        ${p.description ? `<p class="osint-card-desc">${formatFeedDescription(p.description)}</p>` : ''}
        ${p.link ? `<a class="osint-card-link" href="${escapeHtml(p.link)}" target="_blank" rel="noopener noreferrer">Official incident page ↗</a>` : ''}
      </div>
    `;
  }

  ['osint-fires-fill', 'osint-fires-point'].forEach(layer => {
    map.on('click', layer, event => {
      const feature = event.features && event.features[0];
      if (!feature) return;
      fireClickPopup.setLngLat(event.lngLat).setHTML(emergencyCardHtml(feature.properties || {}, 'Fire / burn')).addTo(map);
    });
  });

  map.on('click', 'osint-incidents-core', event => {
    const feature = event.features && event.features[0];
    if (!feature) return;
    incidentClickPopup.setLngLat(event.lngLat).setHTML(emergencyCardHtml(feature.properties || {}, 'Emergency incident')).addTo(map);
  });

  map.on('click', 'osint-quakes-core', event => {
    const feature = event.features && event.features[0];
    if (!feature) return;
    const p = feature.properties || {};
    const depth = p.depth_km != null && p.depth_km !== '' ? `${p.depth_km} km` : 'n/a';
    quakeClickPopup.setLngLat(event.lngLat).setHTML(`
      <div class="osint-emergency-card">
        <strong>M${escapeHtml(String(p.mag ?? '?'))} earthquake</strong>
        <div class="osint-cam-meta">
          <span>Location: ${escapeHtml(p.place || 'Unknown')}</span>
          <span>Depth: ${escapeHtml(depth)}</span>
          <span>Source: USGS</span>
        </div>
        ${p.url ? `<a class="osint-card-link" href="${escapeHtml(p.url)}" target="_blank" rel="noopener noreferrer">USGS event page ↗</a>` : ''}
      </div>
    `).addTo(map);
  });

  ['osint-cameras-icon', 'osint-flights-icon', 'osint-fires-fill', 'osint-fires-point', 'osint-incidents-core', 'osint-quakes-core'].forEach(layer => {
    map.on('mouseenter', layer, () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', layer, () => { map.getCanvas().style.cursor = ''; });
  });
}

function stateHint(lat, lon) {
  // Lightweight AU state hint for the camera card (jurisdiction.js is pool-only).
  if (lat >= -29.0 && lon >= 138 && lon <= 154) return 'QLD';
  if (lat >= -37.6 && lat < -28.15 && lon >= 141) return 'NSW/ACT';
  if (lat < -34.0 && lon >= 141 && lon <= 150) return 'VIC';
  if (lon < 129) return 'WA';
  if (lat < -26 && lon >= 129 && lon <= 141) return 'SA';
  if (lat < -39) return 'TAS';
  if (lat >= -26 && lon >= 129 && lon <= 138) return 'NT';
  return 'AU';
}

// ---------------------------------------------------------------------------
// Feed polling.

// A hard per-request timeout so one slow/stuck feed can never block the others:
// these four run inside Promise.allSettled, which only resolves once EVERY
// input has settled — an unbounded fetch (e.g. cameras contending with the
// browser's per-origin connection cap against the AI-layer detector's own
// polling) previously stalled it forever, silently starving fires/incidents/
// quakes even though their own requests had already succeeded.
async function fetchJson(url, timeoutMs = 8000) {
  const signal = typeof AbortSignal !== 'undefined' && AbortSignal.timeout
    ? AbortSignal.timeout(timeoutMs)
    : (() => {
        const controller = new AbortController();
        setTimeout(() => controller.abort(), timeoutMs);
        return controller.signal;
      })();
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url} → ${response.status}`);
  return response.json();
}

async function refreshFlights() {
  if (!osintEnabled || !state.map) return;
  const b = boundsFromMap(state.map);
  // Pad the query so aircraft just outside the frame still stream in and animate.
  const url = `/api/osint/flights?south=${(b.getSouth() - 0.6).toFixed(4)}&west=${(b.getWest() - 0.6).toFixed(4)}` +
    `&north=${(b.getNorth() + 0.6).toFixed(4)}&east=${(b.getEast() + 0.6).toFixed(4)}&limit=1500`;
  try {
    const data = await fetchJson(url);
    const now = Date.now();
    (data.items || []).forEach(flight => {
      if (!flight.icao24) return;
      flight._seen = now;                       // stamp only aircraft in THIS response
      state.osint.flights.set(flight.icao24, flight);
      const trail = state.osint.flightTrails.get(flight.icao24) || [];
      trail.push({ lat: flight.lat, lon: flight.lon, t: now });
      while (trail.length > 12) trail.shift();
      state.osint.flightTrails.set(flight.icao24, trail);
    });
    // Drop aircraft not reported for 90s so departed planes age out instead of
    // accumulating as ghosts (only feed-present aircraft had _seen refreshed above).
    for (const [icao, flight] of state.osint.flights) {
      if (now - (flight._seen || 0) > 90000) {
        state.osint.flights.delete(icao);
        state.osint.flightTrails.delete(icao);
      }
    }
    setStatus(`${state.osint.flights.size} aircraft tracked · ${data.source || 'ADS-B'}`);
    renderFlights();
  } catch (error) {
    setStatus(`Flight feed unavailable: ${error.message}`);
  }
}

async function refreshSlowFeeds() {
  if (!osintEnabled || !state.map) return;
  const b = boundsFromMap(state.map);
  const camUrl = `/api/osint/cameras?south=${b.getSouth().toFixed(5)}&west=${b.getWest().toFixed(5)}` +
    `&north=${b.getNorth().toFixed(5)}&east=${b.getEast().toFixed(5)}&limit=400`;

  const [fires, incidents, quakes, cameras] = await Promise.allSettled([
    fetchJson('/api/osint/wildfires'),
    fetchJson('/api/osint/incidents'),
    fetchJson('/api/osint/earthquakes?region=au'),
    fetchJson(camUrl)
  ]);

  if (fires.status === 'fulfilled') {
    state.osint.wildfires = fires.value;
    setSourceData(state.map, 'osint-fires-source', fires.value.features ? fires.value : EMPTY_FC);
  }
  if (incidents.status === 'fulfilled') {
    state.osint.incidents = incidents.value.features || [];
    setSourceData(state.map, 'osint-incidents-source', incidents.value.features ? incidents.value : EMPTY_FC);
  }
  if (quakes.status === 'fulfilled') {
    state.osint.earthquakes = quakes.value.items || [];
    setSourceData(state.map, 'osint-quakes-source', featureCollection(
      (quakes.value.items || []).map(q => pointFeature(q, {
        mag: Number(q.mag) || 1, place: q.place, depth_km: q.depth_km, url: q.url,
        tooltip: `M${q.mag} · ${q.place}`
      }))
    ));
  }
  if (cameras.status === 'fulfilled') {
    state.osint.cameras = cameras.value.items || [];
    setSourceData(state.map, 'osint-cameras-source', featureCollection(
      (cameras.value.items || []).map(cam => pointFeature(cam, {
        id: cam.id, name: cam.name, camera_type: cam.camera_type, surveillance: cam.surveillance,
        operator: cam.operator, zone: cam.zone
      }))
    ));
  }
  refreshLinks();
}

function renderFlights() {
  if (!state.map || !state.map.getSource('osint-flights-source')) return;
  const flightFeatures = [];
  const trailFeatures = [];
  state.osint.flights.forEach((flight, icao) => {
    flightFeatures.push(pointFeature(flight, {
      icao24: flight.icao24,
      callsign: flight.callsign || '',
      track: Number(flight.track_deg) || 0,
      altitude_m: flight.altitude_m,
      velocity_ms: flight.velocity_ms,
      squawk: flight.squawk || '',
      aircraft_type: flight.aircraft_type || '',
      origin_country: flight.origin_country || ''
    }));
    const trail = state.osint.flightTrails.get(icao);
    if (trail && trail.length >= 2) trailFeatures.push(lineFeature(trail));
  });
  setSourceData(state.map, 'osint-flights-source', featureCollection(flightFeatures));
  setSourceData(state.map, 'osint-trails-source', featureCollection(trailFeatures));
  refreshLinks();
}

// Semantic network graph: dotted links from the selected property to nearby
// OSINT nodes (cameras and recent incidents within ~1.5 km).
function refreshLinks() {
  if (!state.map || !state.map.getSource('osint-links-source')) return;
  if (!selectedPropertyPoint) {
    setSourceData(state.map, 'osint-links-source', EMPTY_FC);
    return;
  }
  const origin = selectedPropertyPoint;
  const RADIUS_M = 1500;
  const links = [];
  const near = (item) => haversineMeters(origin, item) <= RADIUS_M;
  state.osint.cameras.filter(near).slice(0, 40).forEach(cam => {
    links.push(lineFeature([origin, cam], { kind: 'camera' }));
  });
  (state.osint.incidents || []).forEach(feature => {
    const g = feature.geometry;
    if (!g || g.type !== 'Point') return;
    const item = { lat: g.coordinates[1], lon: g.coordinates[0] };
    if (near(item)) links.push(lineFeature([origin, item], { kind: 'incident' }));
  });
  setSourceData(state.map, 'osint-links-source', featureCollection(links));
}

// Ripple animation for incident/quake rings — pulses radius + opacity.
function startPulse(map) {
  if (state.osint.pulseFrame) return;
  const tick = () => {
    if (!osintEnabled) { state.osint.pulseFrame = null; return; }
    state.osint.pulsePhase = (state.osint.pulsePhase + 0.05) % (Math.PI * 2);
    const wave = (Math.sin(state.osint.pulsePhase) + 1) / 2; // 0..1
    if (map.getLayer('osint-incidents-ripple')) {
      map.setPaintProperty('osint-incidents-ripple', 'circle-radius', 10 + wave * 16);
      map.setPaintProperty('osint-incidents-ripple', 'circle-opacity', 0.35 * (1 - wave));
    }
    if (map.getLayer('osint-quakes-ripple')) {
      map.setPaintProperty('osint-quakes-ripple', 'circle-opacity', 0.3 * (1 - wave));
    }
    state.osint.pulseFrame = requestAnimationFrame(tick);
  };
  state.osint.pulseFrame = requestAnimationFrame(tick);
}

function setStatus(message) {
  const el = document.getElementById('osint-status');
  if (el) el.textContent = message;
}

// ---------------------------------------------------------------------------
// Public API.

export function setSelectedPropertyPoint(point) {
  selectedPropertyPoint = point && Number.isFinite(point.lat) && Number.isFinite(point.lon) ? point : null;
  if (osintEnabled) refreshLinks();
}

export async function setOsintEnabled(enabled) {
  osintEnabled = Boolean(enabled);
  const panel = document.getElementById('osint-panel');
  if (panel) panel.classList.toggle('active', osintEnabled);

  if (!state.map) return;
  if (osintEnabled) {
    await ensureOsintStyle(state.map);
    setLayersVisibility(state.map, OSINT_LAYERS, true);
    startPulse(state.map);
    // The localhost-only rendered-layer harness supplies deterministic source
    // fixtures; do not let asynchronous live feeds overwrite them mid-assertion.
    const renderedLayerTest = new URLSearchParams(window.location.search).get('e2e_rendered_layers') === '1';
    if (renderedLayerTest && ['127.0.0.1', 'localhost', '::1'].includes(window.location.hostname)) {
      setStatus('Rendered-layer fixture mode.');
      return;
    }
    setStatus('Acquiring tactical feeds...');
    refreshFlights();
    refreshSlowFeeds();
    state.osint.flightTimer = window.setInterval(refreshFlights, 10000);
    state.osint.slowTimer = window.setInterval(refreshSlowFeeds, 60000);
  } else {
    window.clearInterval(state.osint.flightTimer);
    window.clearInterval(state.osint.slowTimer);
    state.osint.flightTimer = null;
    state.osint.slowTimer = null;
    if (state.map.getSource('osint-fires-source')) {
      setLayersVisibility(state.map, OSINT_LAYERS, false);
    }
    setStatus('Tactical intelligence layer idle.');
  }
}

export function initOsint() {
  const toggle = document.getElementById('osint-toggle');
  if (toggle) {
    toggle.addEventListener('change', () => setOsintEnabled(toggle.checked));
  }
  // Refresh flights on pan so new aircraft in the frame appear promptly.
  state.map?.on('moveend', () => {
    const renderedLayerTest = new URLSearchParams(window.location.search).get('e2e_rendered_layers') === '1';
    if (osintEnabled && !renderedLayerTest) refreshFlights();
  });
}
