// Interactive CAD-style polygon drawing on the MapLibre canvas: dashed live
// guideline from the last vertex to the cursor plus a floating readout with the
// current segment length and the accumulated area, all computed client-side
// with open Haversine/shoelace math.
import { state } from '../state.js';
import { featureIntersectsPolygon } from '../utils.js';
import { analyzePolygon, clearAnalysisScopedFeatures, retireActiveAreaScan } from '../services/api.js';
import {
  EMPTY_FC,
  featureCollection,
  haversineMeters,
  lineFeature,
  pointFeature,
  polygonFeature,
  ringAreaSqm,
  setSourceData
} from '../geo.js';

function setDrawingUI(active) {
  const mode = active ? state.drawingMode || 'analysis' : null;
  document.getElementById('map')?.classList.toggle('drawing-active', active);
  const button = document.getElementById('draw-area-btn');
  if (button) {
    const isActive = mode === 'analysis';
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
    button.textContent = isActive ? 'Drawing... double-click to finish' : 'Select analysis area';
  }
  const poolButton = document.getElementById('mark-pool-btn');
  if (poolButton) {
    const isActive = mode === 'pool';
    poolButton.classList.toggle('active', isActive);
    poolButton.setAttribute('aria-pressed', String(isActive));
    poolButton.textContent = isActive ? 'Tracing... double-click to save' : 'Add precise pool';
  }
  const hint = document.getElementById('draw-hint');
  if (hint) {
    const hintText = hint.querySelector('span');
    if (hintText) {
      hintText.textContent = mode === 'pool'
        ? 'Click pool corners · double-click or click the start point to save · Ctrl+Z to undo'
        : 'Click to place points · double-click or click the start point to finish · then Enter to analyse · Ctrl+Z to undo';
    }
    hint.style.opacity = active ? '1' : '0';
    hint.setAttribute('aria-hidden', String(!active));
  }
  if (!active) hideMeasureTooltip();
}

// ---------------------------------------------------------------------------
// Live measurement readout.

function measureTooltipEl() {
  return document.getElementById('measure-tooltip');
}

function hideMeasureTooltip() {
  const el = measureTooltipEl();
  if (el) el.style.display = 'none';
  if (state.mapReady) setSourceData(state.map, 'draw-guide-source', EMPTY_FC);
}

function formatDistance(meters) {
  return meters >= 1000 ? `${(meters / 1000).toFixed(2)} km` : `${meters.toFixed(1)} m`;
}

function formatArea(sqm) {
  return sqm >= 100000 ? `${(sqm / 10000).toFixed(2)} ha` : `${sqm.toFixed(0)} m²`;
}

function onMapMouseMove(event) {
  if (!state.drawingMode || state.vertices.length === 0) return;
  const cursor = { lat: event.lngLat.lat, lng: event.lngLat.lng };
  const last = state.vertices[state.vertices.length - 1];

  // Dashed guideline from the last placed vertex to the cursor.
  setSourceData(state.map, 'draw-guide-source', featureCollection([
    lineFeature([last, cursor])
  ]));

  const segment = haversineMeters(last, cursor);
  const liveRing = [...state.vertices, cursor];
  const area = liveRing.length >= 3 ? ringAreaSqm(liveRing) : 0;

  const el = measureTooltipEl();
  if (el) {
    el.style.display = 'block';
    el.style.left = `${event.point.x + 14}px`;
    el.style.top = `${event.point.y + 14}px`;
    el.innerHTML = `
      <span class="measure-seg">${formatDistance(segment)}</span>
      ${area > 0 ? `<span class="measure-area">${formatArea(area)}</span>` : ''}
      <span class="measure-hint">${state.vertices.length >= 3 ? 'double-click / click start to finish' : `${3 - state.vertices.length} more point${state.vertices.length === 2 ? '' : 's'}`}</span>
    `;
  }
}

// ---------------------------------------------------------------------------
// Draft geometry rendering (vertices, preview line, finished polygon).

function renderDraft() {
  if (!state.mapReady) return;
  const features = [];
  state.vertices.forEach(vertex => features.push(pointFeature(vertex, { role: 'vertex' })));
  if (state.vertices.length >= 2) {
    features.push(lineFeature(state.vertices, { role: 'preview' }));
  }
  if (state.polygon && state.polygon.vertices.length >= 3) {
    features.push(polygonFeature(state.polygon.vertices, { role: 'polygon' }));
  }
  setSourceData(state.map, 'draw-source', featureCollection(features));
}

export function stopDrawing() {
  state.map?.off('click', onMapClick);
  state.map?.off('dblclick', onMapDblClick);
  state.map?.off('mousemove', onMapMouseMove);
  state.drawingMode = null;
  setDrawingUI(false);
}

function updateVertexCounter() {
  const counter = document.getElementById('vertex-count');
  if (!counter) return;
  if (!state.vertices.length) {
    counter.style.display = 'none';
    counter.textContent = 'Vertices: 0';
    return;
  }
  counter.style.display = 'block';
  counter.textContent = `Vertices: ${state.vertices.length} ${state.vertices.length >= 3 ? '· double-click or click the start point to finish' : '· Need at least 3 points'}`;
}

function clearDraftVertices() {
  state.vertices = [];
  renderDraft();
  hideMeasureTooltip();
  updateVertexCounter();
}

function polygonCoordsFromSelection(polygon) {
  const vertices = polygon?.vertices || [];
  return vertices.map(point => [point.lng, point.lat]);
}

function updateManualPoolCounter() {
  const count = state.manualPools.length;
  const label = document.getElementById('manual-pool-count');
  const clearButton = document.getElementById('clear-pools-btn');
  if (label) {
    label.hidden = count === 0;
    label.textContent = `Precise pools: ${count}`;
  }
  if (clearButton) {
    clearButton.disabled = count === 0;
  }
}

export function updateExistingPoolFlag() {
  // Area selections are suitability-only; pool overlays must not change the
  // selected area's report or rating mode.
  state.hasExistingPool = false;
  return false;
}

// Screen-space distance (px) under which a click on the first vertex closes the
// ring — the standard "click the start point to finish" GIS gesture.
const CLOSE_RING_PX = 14;

export function onMapClick(e) {
  if (state.isPegmanActive || !state.drawingMode) {
    return;
  }
  // Clicking back on (or very near) the first vertex closes and finishes the
  // polygon — no need to hunt for the Enter key.
  if (state.vertices.length >= 3) {
    const first = state.vertices[0];
    const firstPt = state.map.project([first.lng, first.lat]);
    const clickPt = e.point;
    if (Math.hypot(firstPt.x - clickPt.x, firstPt.y - clickPt.y) <= CLOSE_RING_PX) {
      finishPolygon();
      return;
    }
  }
  state.vertices.push({ lat: e.lngLat.lat, lng: e.lngLat.lng });
  renderDraft();
  updateVertexCounter();
}

// Double-click finishes the polygon. MapLibre emits two `click`s before the
// `dblclick`, so the second click already added a duplicate vertex at the same
// spot — drop it before finishing.
export function onMapDblClick(e) {
  if (state.isPegmanActive || !state.drawingMode) return;
  if (e && e.preventDefault) e.preventDefault();
  if (state.vertices.length >= 2) {
    const last = state.vertices[state.vertices.length - 1];
    const prev = state.vertices[state.vertices.length - 2];
    const lastPt = state.map.project([last.lng, last.lat]);
    const prevPt = state.map.project([prev.lng, prev.lat]);
    if (Math.hypot(lastPt.x - prevPt.x, lastPt.y - prevPt.y) <= 6) {
      state.vertices.pop();
    }
  }
  finishPolygon();
}

export function undoSelection() {
  if (!state.polygon && state.vertices.length > 0) {
    state.vertices.pop();
    renderDraft();
    if (state.vertices.length === 0) hideMeasureTooltip();
    updateVertexCounter();
    return true;
  }

  const hasSelectedProperty = Boolean(state.currentData?.property_id);
  if (state.polygon || state.currentData || hasSelectedProperty) {
    import('./ui.js').then(module => module.clearAll());
    return true;
  }
  return false;
}

export function initDrawingShortcuts() {
  // Persistent keyboard handling so Enter works both mid-draw AND after a
  // polygon has been finalised (the draw-scoped listener is gone by then).
  document.addEventListener('keydown', event => {
    const target = event.target;
    if (target?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName)) return;
    if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'z') {
      if (undoSelection()) event.preventDefault();
      return;
    }
    if (event.key === 'Enter') {
      // Enter analyses the selection: a valid in-progress ring, or an
      // already-finalised polygon. Never fires an area/lead scan.
      if ((state.drawingMode && state.vertices.length >= 3) ||
          (state.polygon && state.polygon.vertices.length >= 3)) {
        event.preventDefault();
        analyzeSelection();
      }
      return;
    }
    if (event.key === 'Escape') {
      if (state.drawingMode || state.polygon || state.currentData) {
        event.preventDefault();
        import('./ui.js').then(m => m.clearAll());
      }
    }
  });
}

// Finalise the drawn shape without starting network work. Stopping drawing
// restores normal map interaction; the user can then press Enter for the
// suitability report or choose either explicit AI scan action.
export function finalizeSelection() {
  if (state.vertices.length < 3) return null;
  const mode = state.drawingMode || 'analysis';
  if (mode === 'pool') {
    finishManualPool();
    return null;
  }

  // Synchronously invalidates old polling before assigning the new polygon;
  // server cancellation continues in the background.
  retireActiveAreaScan('new-selection-finalized');

  const analysisVertices = state.vertices.slice();
  stopDrawing();

  state.vertices = [];
  state.polygon = { vertices: analysisVertices };
  renderDraft();

  const counter = document.getElementById('vertex-count');
  if (counter) counter.style.display = 'none';

  updateExistingPoolFlag();
  showSelectionActionHint();

  return analysisVertices.map(v => [v.lng, v.lat]);
}

// Finish gesture (double-click / click the start point): finalise only.
export function finishPolygon() {
  finalizeSelection();
}

// Enter: finalise and analyse an in-progress ring, or refresh the suitability
// report for an already-finalised selection. AI detection and deep scanning are
// only started by their own buttons.
export function analyzeSelection() {
  let coords = null;
  if (state.drawingMode === 'pool') {
    finishManualPool();
    return;
  }
  if (state.drawingMode && state.vertices.length >= 3) {
    coords = finalizeSelection();
  } else if (state.polygon && state.polygon.vertices.length >= 3) {
    coords = state.polygon.vertices.map(v => [v.lng, v.lat]);
  }
  if (coords && coords.length >= 3) {
    analyzePolygon(coords);
  }
}

// Brief on-map cue after a polygon is finalised, telling the user the two next
// steps now available to them.
function showSelectionActionHint() {
  const hint = document.getElementById('draw-hint');
  if (!hint) return;
  const hintText = hint.querySelector('span');
  if (hintText) hintText.textContent = 'Area selected · press Enter for Area analysis, or choose an AI scan separately · Esc to clear';
  hint.style.opacity = '1';
  hint.setAttribute('aria-hidden', 'false');
  clearTimeout(showSelectionActionHint._timer);
  showSelectionActionHint._timer = setTimeout(() => {
    hint.style.opacity = '0';
    hint.setAttribute('aria-hidden', 'true');
  }, 4200);
}

export function finishManualPool() {
  if (state.vertices.length < 3) return;
  const poolVertices = state.vertices.slice();
  stopDrawing();
  clearDraftVertices();

  const geometry = poolVertices.map(point => ({ lat: point.lat, lon: point.lng }));
  const poolId = `manual-pool-${++state.manualPoolSequence}`;
  const feature = {
    type: 'way',
    id: poolId,
    geometry,
    tags: {
      leisure: 'swimming_pool',
      source: 'manual_user_trace',
      pool_trace: 'precise'
    }
  };
  state.manualPools.push(feature);

  import('./map.js').then(m => m.renderManualPools());

  updateManualPoolCounter();
  updateExistingPoolFlag();

  if (state.polygon) {
    const coords = polygonCoordsFromSelection(state.polygon);
    if (coords.length >= 3) analyzePolygon(coords);
  }
}

export function clearAnalysisPolygon() {
  retireActiveAreaScan('selection-cleared');
  state.polygon = null;
  state.vertices = [];
  state.currentData = null;
  state.hasExistingPool = false;
  if (state.mapReady) {
    setSourceData(state.map, 'draw-source', EMPTY_FC);
    setSourceData(state.map, 'draw-guide-source', EMPTY_FC);
  }
  clearAnalysisScopedFeatures({ render: true });
  hideMeasureTooltip();
  updateVertexCounter();
}

export function startDrawing() {
  // Retire the old scope on the first user action, not after the new ring is
  // complete. This prevents a late poll from repainting the map while drawing.
  retireActiveAreaScan('new-selection-started');
  // A new drawing is a new analysis scope. Prevent the previous property's
  // report or existing-pool flag from leaking into it before Enter is pressed.
  state.currentData = null;
  state.hasExistingPool = false;
  import('./controls.js').then(m => m.toggleMapTilt(false));
  stopDrawing();
  state.drawingMode = 'analysis';
  clearDraftVertices();
  if (state.polygon) clearAnalysisPolygon();
  else clearAnalysisScopedFeatures({ render: true });

  setDrawingUI(true);
  state.map.on('click', onMapClick);
  state.map.on('dblclick', onMapDblClick);
  state.map.on('mousemove', onMapMouseMove);
}

export function startPoolMarking() {
  import('./controls.js').then(m => m.toggleMapTilt(false));
  stopDrawing();
  state.drawingMode = 'pool';
  clearDraftVertices();
  setDrawingUI(true);
  state.map.on('click', onMapClick);
  state.map.on('dblclick', onMapDblClick);
  state.map.on('mousemove', onMapMouseMove);
}

export function clearManualPools({ reanalyze = true } = {}) {
  state.manualPools = [];
  import('./map.js').then(m => m.renderManualPools());
  updateManualPoolCounter();
  updateExistingPoolFlag();
  if (reanalyze && state.polygon) {
    const coords = polygonCoordsFromSelection(state.polygon);
    if (coords.length >= 3) analyzePolygon(coords);
  }
}
