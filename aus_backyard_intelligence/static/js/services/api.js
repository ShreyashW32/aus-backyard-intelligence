import { state } from '../state.js';
import { renderAILayers } from '../components/map.js';
import { renderAnalysis, showToast, updateOsirisIntelPanel } from '../components/ui.js';
import { escapeHtml, featureIntersectsPolygon, geometryIntersectsPolygon, isPointInPolygon } from '../utils.js';
import {
  isPoolTagged,
  isValidBuildingGeometry,
  poolConfidence
} from './feature_algorithms.js';
import {
  describeRequestError,
  polygonScopeKey,
  reconcileOwnedSnapshot,
  scanIsTerminal,
  scanMatchesPolygon,
  scanPollIsCurrent
} from './scan_flow.js';
import {
  EMPTY_FC,
  GeoBounds,
  boundsFromMap,
  featureCollection,
  fitGeoBounds,
  geojsonBounds,
  pointFeature,
  polygonFeature,
  setLayersVisibility,
  setSourceData,
  setView,
  uiZoom
} from '../geo.js';

// Legacy key retained only so old cross-viewport detections can be deleted.
const SATELLITE_POOL_CACHE_KEY = 'abi_satellite_pool_cache_v16';
// Display floor for one-shot selected-area satellite detections.
// Detections below this are never shown or cached; mapped OSM pools (tag confidence
// >= 0.7) and scan-voted pools are exempt. Lowered 0.65→0.55 (build r22b) to surface
// more real backyard pools — the server-side strict-water / roof-veto /
// area gates already reject non-pool blobs, so this band is mostly shaded or partial
// pools. The latest larger r23 eval is recall 0.882 / precision~0.917 / F1~0.899.
// Override at runtime with `window.ABI_POOL_CONF_FLOOR = 0.6` before the map loads.
const SATELLITE_POOL_MIN_CONFIDENCE = (() => {
  const override = Number(typeof window !== 'undefined' && window.ABI_POOL_CONF_FLOOR);
  return Number.isFinite(override) && override > 0 && override < 1 ? override : 0.55;
})();
const MAX_PERSISTED_POOLS = 350;
const LEGACY_POOL_CACHE_KEYS = [
  'qld_backyard_satellite_pool_cache_v6',
  'qld_backyard_satellite_pool_cache_v7',
  'qld_backyard_satellite_pool_cache_v8',
  'qld_backyard_satellite_pool_cache_v9',
  'qld_backyard_satellite_pool_cache_v10',
  'qld_backyard_satellite_pool_cache_v12',
  'qld_backyard_satellite_pool_cache_v13',
  'abi_satellite_pool_cache_v14',
  'abi_satellite_pool_cache_v15'
];

function setFeatureLayerStatus(message, tone = 'idle', retry = false) {
  const status = document.getElementById('feature-layer-status');
  const text = document.getElementById('feature-layer-status-text');
  const retryButton = document.getElementById('feature-layer-retry');
  if (!status || !text || !retryButton) return;
  status.dataset.tone = tone;
  text.textContent = message;
  retryButton.hidden = !retry;
}

function setAgentScanStatus(message, tone = 'idle') {
  const status = document.getElementById('agent-scan-status');
  if (!status) return;
  status.textContent = message;
  status.dataset.tone = tone;
}

function setAgentScanControlsRunning(running, label = '') {
  const scanButton = document.getElementById('run-agent-scan-btn');
  const detectionButton = document.getElementById('run-ai-detection-btn');
  const passInput = document.getElementById('agent-scan-passes');
  if (scanButton) {
    if (!scanButton.dataset.idleLabel) scanButton.dataset.idleLabel = scanButton.textContent || 'Deep pool scan';
    scanButton.disabled = Boolean(running);
    scanButton.textContent = running ? (label || 'Scan running…') : scanButton.dataset.idleLabel;
  }
  if (detectionButton) detectionButton.disabled = Boolean(running);
  if (passInput) passInput.disabled = Boolean(running);
}

function setAIDetectionControlsRunning(running, label = '') {
  const detectionButton = document.getElementById('run-ai-detection-btn');
  const deepScanButton = document.getElementById('run-agent-scan-btn');
  const passInput = document.getElementById('agent-scan-passes');
  if (detectionButton) {
    if (!detectionButton.dataset.idleLabel) detectionButton.dataset.idleLabel = detectionButton.textContent || 'AI detection scan';
    detectionButton.disabled = Boolean(running);
    detectionButton.textContent = running ? (label || 'Detecting…') : detectionButton.dataset.idleLabel;
  }
  if (deepScanButton) deepScanButton.disabled = Boolean(running);
  if (passInput) passInput.disabled = Boolean(running);
}

function requestOrigin() {
  return typeof window !== 'undefined' && window.location?.origin ? window.location.origin : '';
}

const SCAN_CANCEL_TIMEOUT_MS = 12000;
const SCAN_POLL_TIMEOUT_MS = 15000;
const SCAN_START_TIMEOUT_MS = 30000;
const PLACE_SCAN_START_TIMEOUT_MS = 45000;

async function fetchWithDeadline(url, options = {}, timeoutMs = SCAN_POLL_TIMEOUT_MS) {
  const controller = new AbortController();
  const deadline = Math.max(1000, Number(timeoutMs) || SCAN_POLL_TIMEOUT_MS);
  let deadlineReached = false;
  const timer = window.setTimeout(() => {
    deadlineReached = true;
    controller.abort();
  }, deadline);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (deadlineReached) {
      const timeoutError = new Error(`Request exceeded ${Math.round(deadline / 1000)} seconds.`);
      timeoutError.name = 'TimeoutError';
      throw timeoutError;
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

function currentSelectionCoords() {
  return polygonCoordsFromSelection(state.polygon);
}

function currentSelectionScopeKey() {
  return polygonScopeKey(currentSelectionCoords());
}

// Immediately detach the browser from an old-scope background scan, then ask
// the server to stop it. The synchronous state reset is intentional: a poll or
// start response already in flight must become stale before the user's first
// vertex/new polygon can establish a different analysis scope.
export async function retireActiveAreaScan(reason = 'selection-changed') {
  const scanId = state.agentScanActiveId;
  const hadPendingStart = state.agentScanStartInFlight;
  if (!scanId && state.agentScanCancellationInFlight) {
    return { retired: true, cancelled: false, scan_id: null, cancellation_pending: true };
  }
  const retirementGeneration = ++state.agentScanPollGeneration;
  clearTimeout(state.agentScanPollTimer);
  state.agentScanPollTimer = null;
  state.agentScanActiveId = null;
  state.agentScanScopeKey = null;

  if (!scanId) {
    setAgentScanControlsRunning(false);
    if (hadPendingStart) {
      setAgentScanStatus('Selection changed; the pending old-area scan will be discarded.', 'idle');
    }
    return { retired: hadPendingStart, cancelled: false, scan_id: null };
  }

  state.agentScanCancellationInFlight = true;
  setAgentScanControlsRunning(true, 'Stopping old scan…');
  setAgentScanStatus('Stopping the scan for the previous selection…', 'running');
  try {
    const response = await fetchWithDeadline(`/api/agent/scans/${encodeURIComponent(scanId)}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: String(reason || 'selection-changed').slice(0, 80) })
    }, SCAN_CANCEL_TIMEOUT_MS);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Scan cancellation failed (${response.status})`);
    if (state.agentScanPollGeneration === retirementGeneration && !state.agentScanActiveId) {
      setAgentScanStatus('Previous-area scan cancelled. New selections will receive only their own results.', 'idle');
    }
    return { retired: true, cancelled: payload.status === 'cancelled', scan_id: scanId, job: payload };
  } catch (error) {
    // Local retirement remains authoritative even if the server disappeared:
    // generation checks below guarantee its late results cannot be rendered.
    const explained = describeRequestError(error, 'Scan cancellation', requestOrigin());
    console.warn('Could not confirm server-side scan cancellation:', explained);
    if (state.agentScanPollGeneration === retirementGeneration && !state.agentScanActiveId) {
      setAgentScanStatus(`Previous scan detached locally. ${explained}`, 'error');
    }
    return { retired: true, cancelled: false, scan_id: scanId, error };
  } finally {
    state.agentScanCancellationInFlight = false;
    if (state.agentScanPollGeneration === retirementGeneration && !state.agentScanActiveId) {
      setAgentScanControlsRunning(false);
    }
  }
}

function selectedScanPasses() {
  const input = document.getElementById('agent-scan-passes');
  const value = Number(input?.value || 12);
  if (!Number.isFinite(value)) return 12;
  return Math.max(3, Math.min(40, Math.round(value)));
}

function isSatelliteDetectedPool(feature) {
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

// A pool produced by a deliberate area scan (cross-pass voting adds scan:* tags).
function isScanSourcedPool(feature) {
  const tags = feature?.tags || {};
  return Boolean(tags['scan:verdict'] || tags['scan:passes_seen'] || tags['scan:coverage']);
}

function poolCacheUid(feature) {
  const tags = feature?.tags || {};
  const source = tags['detector:source'] || 'pool';
  if (feature?.id !== undefined) return `${feature.type}-${feature.id}-${source}`;
  if (feature?.lat !== undefined && feature?.lon !== undefined) {
    return `${feature.type}-${Number(feature.lat).toFixed(6)}-${Number(feature.lon).toFixed(6)}-${source}`;
  }
  const geometry = Array.isArray(feature?.geometry) ? feature.geometry : [];
  const first = geometry[0] || {};
  return `${feature?.type || 'pool'}-${Number(first.lat || 0).toFixed(6)}-${Number(first.lon || 0).toFixed(6)}-${source}`;
}

function serializePoolFeature(feature) {
  return {
    type: feature.type,
    id: feature.id,
    lat: feature.lat,
    lon: feature.lon,
    geometry: Array.isArray(feature.geometry) ? feature.geometry.map(point => ({ lat: point.lat, lon: point.lon })) : undefined,
    tags: {
      ...(feature.tags || {}),
      'cache:client': 'local-storage'
    },
    cached_at: new Date().toISOString()
  };
}

function readPersistedPoolCache() {
  if (typeof localStorage === 'undefined') return [];
  try {
    const raw = localStorage.getItem(SATELLITE_POOL_CACHE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function loadPersistentPoolCache() {
  if (typeof localStorage !== 'undefined') {
    [...LEGACY_POOL_CACHE_KEYS, SATELLITE_POOL_CACHE_KEY].forEach(key => localStorage.removeItem(key));
  }
  // Detection overlays are scoped to the current explicit selection. Persisted
  // detections cannot prove that scope after a reload, so never restore them.
  return 0;
}

function persistSatellitePoolCache() {
  if (typeof localStorage !== 'undefined') localStorage.removeItem(SATELLITE_POOL_CACHE_KEY);
}

function polygonCoordsFromSelection(polygon) {
  const vertices = polygon?.vertices || [];
  return vertices.map(point => [point.lng, point.lat]);
}

function polygonFromBounds(bounds) {
  return [
    [bounds.getWest(), bounds.getSouth()],
    [bounds.getEast(), bounds.getSouth()],
    [bounds.getEast(), bounds.getNorth()],
    [bounds.getWest(), bounds.getNorth()]
  ];
}

function scanStatusText(job) {
  const counts = job?.counts || {};
  const leads = Array.isArray(job?.leads) ? job.leads.length : 0;
  const base = job?.summary || `Scan ${job?.status || 'running'}.`;
  if (!job) return 'Background scan idle.';
  const detector = job.detector?.engine
    ? ` Detector: ${String(job.detector.engine).replaceAll('_', ' ')}.`
    : '';
  return `${base}${detector} Leads: ${leads}. Pools: ${counts.pools || 0}. Solar candidates: ${counts.solar || 0}. Structures: ${counts.buildings || 0}.`;
}

async function refreshAreaScan(scanId, pollGeneration) {
  let response;
  try {
    response = await fetchWithDeadline(
      `/api/agent/scans/${encodeURIComponent(scanId)}`,
      {},
      SCAN_POLL_TIMEOUT_MS
    );
  } catch (error) {
    throw new Error(describeRequestError(error, 'Scan status', requestOrigin()));
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Scan status failed (${response.status})`);
  if (!scanPollIsCurrent(
    state.agentScanActiveId,
    state.agentScanPollGeneration,
    scanId,
    pollGeneration
  )) {
    return null;
  }
  const activeCoords = currentSelectionCoords();
  const matchesSelection = scanMatchesPolygon(payload, activeCoords);
  if (matchesSelection) ingestScanFeatures(payload);
  const done = scanIsTerminal(payload.status);
  const statusTone = payload.status === 'failed' || payload.status === 'cancelled' ? 'error' : 'success';
  if (!matchesSelection) {
    setAgentScanStatus(
      done
        ? 'A scan for a previous area finished; its detections were not added to the current selection.'
        : 'A scan for a previous area is still running; its detections are hidden from this selection.',
      done ? 'idle' : 'running'
    );
  } else {
    setAgentScanStatus(scanStatusText(payload), done ? statusTone : 'running');
  }
  return payload;
}

// Pan/zoom the map to frame all satellite-detected pools currently cached, so a finished
// scan can never leave the user staring at an empty part of the selection while the
// detections sit off-screen. Returns the number of pools framed (0 if none).
function focusDetectedPools() {
  if (!state.map) return 0;
  let south = 90, west = 180, north = -90, east = -180, count = 0;
  for (const feature of state.globalPools.values()) {
    if (!isSatelliteDetectedPool(feature)) continue;
    const points = feature.type === 'node'
      ? [{ lat: feature.lat, lon: feature.lon }]
      : (Array.isArray(feature.geometry) ? feature.geometry : []);
    for (const point of points) {
      const lat = Number(point.lat);
      const lon = Number(point.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      south = Math.min(south, lat); north = Math.max(north, lat);
      west = Math.min(west, lon); east = Math.max(east, lon);
      count += 1;
    }
  }
  if (count === 0 || south > north || west > east) return 0;
  fitGeoBounds(state.map, new GeoBounds(south, west, north, east), { maxZoom: 18, padding: 40 });
  return state.globalPools.size;
}

function startAreaScanPolling(scanId) {
  if (!scanId) return;
  clearTimeout(state.agentScanPollTimer);
  state.agentScanPollTimer = null;
  const pollGeneration = ++state.agentScanPollGeneration;
  let missedPolls = 0;
  const poll = async () => {
    // A newer scan superseded this timer.
    if (!scanPollIsCurrent(
      state.agentScanActiveId,
      state.agentScanPollGeneration,
      scanId,
      pollGeneration
    )) return;
    try {
      const job = await refreshAreaScan(scanId, pollGeneration);
      if (!job) return;
      missedPolls = 0;
      if (scanIsTerminal(job.status)) {
        const matchesCurrentSelection = scanMatchesPolygon(job, currentSelectionCoords());
        clearTimeout(state.agentScanPollTimer);
        state.agentScanPollTimer = null;
        state.agentScanActiveId = null;
        state.agentScanScopeKey = null;
        setAgentScanControlsRunning(false);
        if (matchesCurrentSelection && !['failed', 'cancelled'].includes(job.status)) {
          // Guarantee the detections are actually on screen once the scan settles.
          renderAILayers();
          // Only re-frame the camera for scans where the user isn't already
          // positioned (place-name searches). For a drawn-area scan they are
          // looking right at it, so hijacking the view feels like the map
          // "won't stay still" while they pan.
          if (state.scanCameraFollow) {
            const framed = focusDetectedPools();
            if (framed > 0) {
              showToast(`Scan complete — framed ${framed} detected pool${framed === 1 ? '' : 's'} on the map.`, 'success');
            }
          }
        }
        return;
      }
    } catch (error) {
      // Retirement may happen while the fetch is in flight. A rejection from
      // that old request must not overwrite the new selection's status or arm
      // another old-generation timer.
      if (!scanPollIsCurrent(
        state.agentScanActiveId,
        state.agentScanPollGeneration,
        scanId,
        pollGeneration
      )) return;
      missedPolls += 1;
      const exhausted = missedPolls >= 8;
      setAgentScanStatus(
        exhausted
          ? `Scan status could not be recovered: ${error.message}`
          : `Scan may still be running; reconnecting to status (${missedPolls}/8)…`,
        exhausted ? 'error' : 'running'
      );
      if (exhausted) {
        state.agentScanPollTimer = null;
        state.agentScanActiveId = null;
        state.agentScanScopeKey = null;
        setAgentScanControlsRunning(false);
        return;
      }
    }
    if (!scanPollIsCurrent(
      state.agentScanActiveId,
      state.agentScanPollGeneration,
      scanId,
      pollGeneration
    )) return;
    const delay = missedPolls > 0 ? Math.min(15000, 2500 * (2 ** (missedPolls - 1))) : 3500;
    state.agentScanPollTimer = window.setTimeout(poll, delay);
  };
  // Fetch once immediately so queued/running state and early detections appear
  // without an unnecessary 3.5 second dead period.
  state.agentScanPollTimer = window.setTimeout(poll, 0);
}

function featureCountText(counts, activeLayers, prefix = 'Loaded') {
  const parts = [];
  if (activeLayers.includes('pools')) parts.push(`${counts.pools || 0} pool${counts.pools === 1 ? '' : 's'}`);
  // Solar candidates are emitted by the same selected-area satellite model as
  // pools, so report their result even though "solar" is not an API layer name.
  if (activeLayers.includes('pools')) parts.push(`${counts.solar || 0} solar candidate${counts.solar === 1 ? '' : 's'}`);
  if (activeLayers.includes('buildings')) parts.push(`${counts.buildings || 0} structure${counts.buildings === 1 ? '' : 's'}`);
  if (activeLayers.includes('trees')) parts.push(`${counts.trees || 0} vegetation feature${counts.trees === 1 ? '' : 's'}`);
  const total = activeLayers.reduce((sum, layer) => sum + (counts[layer] || 0), 0) + (counts.solar || 0);
  return {
    message: `${prefix}: ${parts.join(' · ')}`,
    tone: total > 0 ? 'success' : 'empty'
  };
}

function materializeGlobalFeatureCache() {
  state.cachedPoolsAndTrees = {
    pools: Array.from(state.globalPools.values()),
    trees: Array.from(state.globalTrees.values()),
    powerlines: Array.from(state.globalPowerlines.values())
  };
  state.cachedBuildings = {
    buildings: Array.from(state.globalBuildings.values())
  };
  state.cacheDirty = false;
  state.aiFeatureRevision += 1;
}

function clearSatellitePoolDetections() {
  let removed = 0;
  Array.from(state.globalPools.entries()).forEach(([key, feature]) => {
    if (isSatelliteDetectedPool(feature)) {
      state.globalPools.delete(key);
      removed += 1;
    }
  });
  if (removed > 0) {
    materializeGlobalFeatureCache();
    persistSatellitePoolCache();
    renderAILayers();
  }
  return removed;
}

export function clearAnalysisScopedFeatures({ render = true } = {}) {
  state.aiFetchController?.abort();
  state.aiFetchController = null;
  state.aiFetchSequence += 1;
  state.isFetchingAIData = false;
  state.fetchedBBoxes = [];
  state.globalPools.clear();
  state.globalTrees.clear();
  state.globalPowerlines.clear();
  state.globalBuildings.clear();
  state.globalSolarPanels.clear();
  state.scanOwnedFeatureSnapshots.clear();
  materializeGlobalFeatureCache();
  persistSatellitePoolCache();
  updateOsirisIntelPanel('Selected-area feature cache cleared.');
  if (render) renderAILayers();
}

function featureIntersectsBounds(feature, bounds) {
  if (!feature || !bounds) return false;
  if (feature.type === 'node') {
    return bounds.contains({ lat: Number(feature.lat), lon: Number(feature.lon) });
  }
  const geometry = Array.isArray(feature.geometry) ? feature.geometry : [];
  return geometry.some(point => bounds.contains({ lat: Number(point?.lat), lon: Number(point?.lon) }));
}

function clearSatellitePoolDetectionsInBounds(bounds, { includeScanSourced = false } = {}) {
  let removed = 0;
  Array.from(state.globalPools.entries()).forEach(([key, feature]) => {
    // Scan-sourced pools are preserved unless a new scan explicitly replaces them.
    if (!includeScanSourced && isScanSourcedPool(feature)) return;
    if (isSatelliteDetectedPool(feature) && featureIntersectsBounds(feature, bounds)) {
      state.globalPools.delete(key);
      removed += 1;
    }
  });
  if (removed > 0) {
    materializeGlobalFeatureCache();
    persistSatellitePoolCache();
  }
  return removed;
}

const FEATURE_CACHE_NAMES = ['pools', 'solar', 'buildings', 'trees', 'powerlines'];

function featureCacheMap(name) {
  return {
    pools: state.globalPools,
    solar: state.globalSolarPanels,
    buildings: state.globalBuildings,
    trees: state.globalTrees,
    powerlines: state.globalPowerlines
  }[name];
}

function beginScanSnapshot(scanId) {
  if (!scanId) return null;
  let ownership = state.scanOwnedFeatureSnapshots.get(scanId);
  if (!ownership) {
    ownership = Object.fromEntries(FEATURE_CACHE_NAMES.map(name => [name, new Map()]));
    state.scanOwnedFeatureSnapshots.set(scanId, ownership);
  }
  return {
    scanId,
    ownership,
    desired: Object.fromEntries(FEATURE_CACHE_NAMES.map(name => [name, new Set()])),
    changed: false
  };
}

function storeFeatureInCache(cacheName, key, feature, scanContext = null) {
  const cache = featureCacheMap(cacheName);
  if (!cache) return;
  if (!scanContext) {
    cache.set(key, feature);
    return;
  }

  const { scanId, ownership, desired } = scanContext;
  desired[cacheName].add(key);
  if (!ownership[cacheName].has(key)) {
    ownership[cacheName].set(key, {
      hadPrevious: cache.has(key),
      previous: cache.get(key)
    });
  }
  const ownedFeature = { ...feature };
  Object.defineProperty(ownedFeature, '_clientScanId', {
    value: scanId,
    enumerable: false,
    configurable: true
  });
  cache.set(key, ownedFeature);
  scanContext.changed = true;
}

function finishScanSnapshot(scanContext) {
  if (!scanContext) return false;
  const { scanId, ownership, desired } = scanContext;
  for (const cacheName of FEATURE_CACHE_NAMES) {
    const cache = featureCacheMap(cacheName);
    if (reconcileOwnedSnapshot(cache, ownership[cacheName], desired[cacheName], scanId)) {
      scanContext.changed = true;
    }
  }
  return scanContext.changed;
}

function restoreAndClearPriorScanSnapshots() {
  let changed = false;
  // Unwind newest-to-oldest so stacked scan overlays resolve all the way back
  // to the original mapped/one-shot baseline, never to a stale prior scan.
  const snapshots = [...state.scanOwnedFeatureSnapshots.entries()].reverse();
  for (const [scanId, ownership] of snapshots) {
    for (const cacheName of FEATURE_CACHE_NAMES) {
      if (reconcileOwnedSnapshot(featureCacheMap(cacheName), ownership[cacheName], new Set(), scanId)) {
        changed = true;
      }
    }
  }
  state.scanOwnedFeatureSnapshots.clear();
  if (changed) {
    materializeGlobalFeatureCache();
    persistSatellitePoolCache();
    renderAILayers();
  }
  return changed;
}

function ingestFeatureElements(elements, { scanId = null } = {}) {
  const responseCounts = { pools: 0, solar: 0, buildings: 0, trees: 0, powerlines: 0 };
  const scanContext = beginScanSnapshot(scanId);
  if (!Array.isArray(elements)) elements = [];

  elements.forEach(e => {
    if (!e || !e.tags) return;
    const uid = `${e.type}-${e.id}`;
    const isVegetation = ['tree', 'wood', 'scrub', 'heath', 'grassland', 'shrubbery', 'tree_row', 'wetland'].includes(e.tags.natural) ||
      ['forest', 'orchard', 'vineyard', 'meadow', 'grass', 'recreation_ground', 'greenfield'].includes(e.tags.landuse) ||
      ['garden', 'park', 'nature_reserve'].includes(e.tags.leisure);

    if (e.tags['detector:source'] === 'satellite-yolo-solar') {
      // Classify model solar output before generic mapped tags so a future
      // enrichment tag (building/landuse) cannot swallow the detection.
      if (e.geometry) {
        storeFeatureInCache('solar', uid, e, scanContext);
        responseCounts.solar += 1;
      }
    } else if (isPoolTagged(e)) {
      // Accuracy floor for one-shot selected-area detections only. Scan-sourced pools have
      // already survived cross-pass voting + area/building gates server-side — hiding
      // a voted pool because its raw frame confidence is 0.6 would recreate misses.
      if (
        isSatelliteDetectedPool(e) && !isScanSourcedPool(e) &&
        poolConfidence(e) < SATELLITE_POOL_MIN_CONFIDENCE
      ) return;
      if (e.type === 'node') {
        storeFeatureInCache('pools', poolCacheUid(e), e, scanContext);
        responseCounts.pools += 1;
      } else if (e.type === 'way' && e.geometry) {
        storeFeatureInCache('pools', poolCacheUid(e), e, scanContext);
        responseCounts.pools += 1;
      } else if (e.type === 'relation' && e.members) {
        e.members.forEach((member, index) => {
          if (member.type === 'way' && member.role !== 'inner' && member.geometry) {
            const feature = { type: 'way', id: e.id, geometry: member.geometry, tags: e.tags };
            storeFeatureInCache('pools', `${uid}-m${index}`, feature, scanContext);
            responseCounts.pools += 1;
          }
        });
      }
    } else if (isVegetation) {
      if (e.type === 'relation' && Array.isArray(e.members)) {
        e.members.forEach((member, index) => {
          if (member.type === 'way' && member.role !== 'inner' && member.geometry) {
            storeFeatureInCache(
              'trees',
              `${uid}-m${index}`,
              { type: 'way', id: e.id, geometry: member.geometry, tags: e.tags },
              scanContext
            );
            responseCounts.trees += 1;
          }
        });
      } else {
        storeFeatureInCache('trees', uid, e, scanContext);
        responseCounts.trees += 1;
      }
    } else if (e.tags.power === 'line') {
      storeFeatureInCache('powerlines', uid, e, scanContext);
      responseCounts.powerlines += 1;
    } else if (e.tags.building && !['no', 'demolished'].includes(e.tags.building)) {
      if (e.type === 'way' && isValidBuildingGeometry(e.geometry)) {
        storeFeatureInCache('buildings', uid, e, scanContext);
        responseCounts.buildings += 1;
      } else if (e.type === 'relation' && e.members) {
        e.members.forEach((member, index) => {
          if (member.type === 'way' && member.role !== 'inner' && isValidBuildingGeometry(member.geometry)) {
            storeFeatureInCache(
              'buildings',
              `${uid}-m${index}`,
              { type: 'way', id: e.id, geometry: member.geometry, tags: e.tags },
              scanContext
            );
            responseCounts.buildings += 1;
          }
        });
      }
    }
  });

  Object.defineProperty(responseCounts, '_changed', {
    value: finishScanSnapshot(scanContext),
    enumerable: false
  });
  return responseCounts;
}

function ingestScanFeatures(job) {
  if (!job?.scan_id || !state.polygon || !scanMatchesPolygon(job, currentSelectionCoords())) {
    return { pools: 0, solar: 0, buildings: 0, trees: 0, powerlines: 0 };
  }
  const elements = (Array.isArray(job?.feature_elements) ? job.feature_elements : []).filter(feature => {
    if (feature?.type === 'relation') {
      return (feature.members || []).some(member => geometryIntersectsPolygon(member?.geometry, state.polygon));
    }
    return featureIntersectsPolygon(feature, state.polygon);
  });
  const before = {
    pools: state.globalPools.size,
    buildings: state.globalBuildings.size,
    trees: state.globalTrees.size,
    powerlines: state.globalPowerlines.size,
    solar: state.globalSolarPanels.size
  };
  // feature_elements is an authoritative snapshot for this scan, not a delta.
  // Reconciliation removes candidates withdrawn by later voting while restoring
  // any mapped/one-shot feature that the scan temporarily overlaid at the same key.
  const ingested = ingestFeatureElements(elements, { scanId: job.scan_id });
  const changed = ingested._changed ||
    state.globalPools.size !== before.pools ||
    state.globalBuildings.size !== before.buildings ||
    state.globalTrees.size !== before.trees ||
    state.globalPowerlines.size !== before.powerlines ||
    state.globalSolarPanels.size !== before.solar;

  if (changed) {
    materializeGlobalFeatureCache();
    persistSatellitePoolCache();
    const cachedStatus = featureCountText({
      pools: state.globalPools.size,
      solar: state.globalSolarPanels.size,
      buildings: state.globalBuildings.size,
      trees: state.globalTrees.size
    }, ['pools', 'buildings', 'trees'], 'Cached scan overlay');
    setFeatureLayerStatus(cachedStatus.message, cachedStatus.tone);
    updateOsirisIntelPanel(job.summary || cachedStatus.message);
    renderAILayers();
    // The first time a scan surfaces pools, snap the view to them so the detections are
    // guaranteed on screen instead of sitting off in an unscanned corner of a big area.
    // Suppressed for user-drawn scans (scanCameraFollow=false) — they are already there.
    if (state.scanCameraFollow && !state.scanAutoFocused && ingested.pools > 0) {
      if (focusDetectedPools() > 0) {
        state.scanAutoFocused = true;
        showToast(`Showing ${state.globalPools.size} detected pool${state.globalPools.size === 1 ? '' : 's'} — zoom in for outlines.`, 'success');
      }
    }
  }

  return ingested;
}

function layerSet(layers) {
  return new Set(String(layers || '').split(',').map(layer => layer.trim()).filter(Boolean));
}

function cachedLayersCover(cachedLayers, requestedLayers) {
  const cached = layerSet(cachedLayers);
  const requested = layerSet(requestedLayers);
  for (const layer of requested) {
    if (!cached.has(layer)) return false;
  }
  return requested.size > 0;
}

function snapOutward(value, step, direction) {
  return (direction === 'down' ? Math.floor(value / step) : Math.ceil(value / step)) * step;
}

function snappedBufferedBounds(bounds, zoom) {
  const latDiff = bounds.getNorth() - bounds.getSouth();
  const lngDiff = bounds.getEast() - bounds.getWest();
  const bufferRatio = zoom >= 18 ? 0.34 : zoom >= 16 ? 0.24 : 0.18;
  const padLat = latDiff * bufferRatio;
  const padLng = lngDiff * bufferRatio;
  const step = zoom >= 19 ? 0.00075 : zoom >= 18 ? 0.00125 : zoom >= 16 ? 0.0025 : 0.004;

  const south = snapOutward(bounds.getSouth() - padLat, step, 'down');
  const west = snapOutward(bounds.getWest() - padLng, step, 'down');
  const north = snapOutward(bounds.getNorth() + padLat, step, 'up');
  const east = snapOutward(bounds.getEast() + padLng, step, 'up');
  return new GeoBounds(south, west, north, east);
}

export async function loadRankedProperties() {
  const urlParams = new URLSearchParams(window.location.search);
  const moduleParam = urlParams.get('module') || 'pool_3d';
  const lgaParam = urlParams.get('lga') || '';
  if (!lgaParam) return;
  try {
    const res = await fetch(`/api/viewer/map?module=${moduleParam}&lga=${encodeURIComponent(lgaParam)}&top_k=15`);
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.error || `Ranked properties failed (${res.status})`);
    }
    const geojson = await res.json();
    renderRankedProperties(geojson);
  } catch (err) {
    console.error("Failed to load ranked properties:", err);
    showToast(`Could not load rankings: ${err.message}`, 'error');
  }
}

export function renderRankedProperties(geojson) {
  if (!state.mapReady) {
    state.map.once('load', () => renderRankedProperties(geojson));
    return;
  }
  setSourceData(state.map, 'selected-property-source', EMPTY_FC);

  const markers = [];
  const parcelFeatures = [];
  let south = 90, west = 180, north = -90, east = -180;

  if (geojson.parcels && geojson.parcels.features) {
    geojson.parcels.features.forEach(f => {
      const props = f.properties || {};
      const tooltip = `#${props.rank}: ${props.address} · Score: ${props.score}`;
      parcelFeatures.push({
        ...f,
        properties: { ...props, tooltip: `#${props.rank}: ${props.address}` }
      });
      const centroid = props.centroid;
      if (!centroid) return;
      markers.push(pointFeature({ lat: centroid[1], lon: centroid[0] }, {
        property_id: props.property_id,
        tooltip
      }));
      south = Math.min(south, centroid[1]); north = Math.max(north, centroid[1]);
      west = Math.min(west, centroid[0]); east = Math.max(east, centroid[0]);
    });
  }

  setSourceData(state.map, 'ranked-parcels-source', featureCollection(parcelFeatures));
  setSourceData(state.map, 'ranked-markers-source', featureCollection(markers));

  if (markers.length > 0 && south <= north && west <= east) {
    fitGeoBounds(state.map, new GeoBounds(south, west, north, east), { padding: 40 });
  }

  updateVisibilityByZoom();
}

export function updateVisibilityByZoom() {
  if (!state.mapReady) return;
  const showParcels = uiZoom(state.map) >= 16;
  setLayersVisibility(state.map, ['ranked-parcels-fill', 'ranked-parcels-line'], showParcels);
  setLayersVisibility(state.map, ['ranked-markers-circle'], !showParcels);
}

export async function selectPropertyById(propId) {
  state.hasExistingPool = false;
  import('../components/ui.js').then(m => m.openPanel());
  document.getElementById('sp-body').style.opacity = '.4';
  document.getElementById('map-loader').classList.add('active');
  
  try {
    const res = await fetch(`/api/viewer/property/${propId}`);
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.error || `Property request failed (${res.status})`);
    }
    const d = await res.json();

    const geoms = d.geometries || {};
    // Role-tagged features drive the per-role paint expressions on the
    // selected-property layers (parcel / building / backyard styling).
    const toRoleFeatures = (geom, role) => {
      if (!geom) return [];
      const features = geom.type === 'FeatureCollection'
        ? (geom.features || [])
        : geom.type === 'Feature' ? [geom] : [{ type: 'Feature', properties: {}, geometry: geom }];
      return features.map(f => ({ ...f, properties: { ...(f.properties || {}), role } }));
    };
    const selectedFeatures = [
      ...toRoleFeatures(geoms.parcel_geom, 'parcel'),
      ...toRoleFeatures(geoms.building_geom, 'building'),
      ...toRoleFeatures(geoms.backyard_geom, 'backyard')
    ];
    if (state.mapReady) {
      setSourceData(state.map, 'selected-property-source', featureCollection(selectedFeatures));
    }

    const parcelBounds = geojsonBounds(geoms.parcel_geom);
    if (parcelBounds) {
      fitGeoBounds(state.map, parcelBounds, { padding: 50, maxZoom: 19 });
    }

    state.currentData = d;
    renderAnalysis(d);
  } catch (err) {
    console.error("Failed to select property:", err);
    showToast(`Could not load property: ${err.message}`, 'error');
  } finally {
    document.getElementById('sp-body').style.opacity = '1';
    document.getElementById('map-loader').classList.remove('active');
  }
}

export async function analyzePolygon(coords, options = {}) {
  const { silent = false } = options;
  if (!silent) {
    import('../components/ui.js').then(m => m.openPanel());
    document.getElementById('sp-body').style.opacity = '.4';
    document.getElementById('map-loader').classList.add('active');
  }

  try {
    const containedBuildings = [];
    const containedTrees = [];

    if (state.polygon) {
      if (state.cachedBuildings && state.cachedBuildings.buildings) {
        state.cachedBuildings.buildings.forEach(b => {
          if (geometryIntersectsPolygon(b.geometry, state.polygon)) {
            containedBuildings.push(b);
          }
        });
      }

      if (state.cachedPoolsAndTrees && state.cachedPoolsAndTrees.trees) {
        state.cachedPoolsAndTrees.trees.forEach(t => {
          if (t.type === 'node') {
            if (isPointInPolygon({ lat: t.lat, lng: t.lon }, state.polygon)) {
              containedTrees.push(t);
            }
          } else if (geometryIntersectsPolygon(t.geometry, state.polygon)) {
            containedTrees.push(t);
          }
        });
      }
    }

    // Pass only out-of-area pool detections as neighbourhood context. This can
    // modestly support feasibility scoring, while the selected area's own pool
    // list remains empty so no detection can turn the area into an "existing
    // pool" result.
    const selectionLats = coords.map(point => Number(point[1]));
    const selectionLons = coords.map(point => Number(point[0]));
    const selectionBounds = new GeoBounds(
      Math.min(...selectionLats),
      Math.min(...selectionLons),
      Math.max(...selectionLats),
      Math.max(...selectionLons)
    );
    const nearbyPoolContext = Array.from(state.globalPools.values())
      .filter(feature => featureIntersectsBounds(feature, selectionBounds.pad(0.5)))
      .filter(feature => !featureIntersectsPolygon(feature, state.polygon))
      .slice(0, 20);

    let hasPowerline = false;
    if (state.cachedPoolsAndTrees && state.cachedPoolsAndTrees.powerlines) {
      state.cachedPoolsAndTrees.powerlines.forEach(line => {
        if (line.geometry) {
          line.geometry.forEach(pt => {
            if (isPointInPolygon({ lat: pt.lat, lng: pt.lon }, state.polygon)) {
              hasPowerline = true;
            }
          });
        }
      });
    }

    let res;
    try {
      res = await fetch('/api/analyze/area', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          polygon: coords,
          features: { overhead_powerline: hasPowerline },
          local_features: {
            buildings: containedBuildings,
            pools: [],
            trees: containedTrees
          },
          context_features: {
            nearby_pools: nearbyPoolContext
          }
        })
      });
    } catch (error) {
      throw new Error(describeRequestError(error, 'Area suitability analysis', requestOrigin()));
    }
    const d = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(d.error || `Area analysis failed (${res.status})`);
    state.currentData = d;
    state.hasExistingPool = Boolean(d.has_existing_pool || (d.features && d.features.has_existing_pool));
    renderAnalysis(d);
    return { ok: true, data: d };
  } catch (e) {
    console.error(e);
    if (!silent) showToast(`Could not analyze this area: ${e.message}`, 'error');
    return { ok: false, error: e };
  } finally {
    if (!silent) {
      document.getElementById('sp-body').style.opacity = '1';
      document.getElementById('map-loader').classList.remove('active');
    }
  }
}

// Selected-area AI is deliberately independent from the suitability report and
// the multi-pass deep pool scan. It loads pool, solar, mapped structure and
// vegetation overlays without opening or recomputing the Area Analysis panel.
export async function scanSelectedAreaForAIDetections(options = {}) {
  const coords = currentSelectionCoords();
  if (coords.length < 3) throw new Error('Select an analysis area first.');
  if (state.agentScanActiveId || state.agentScanStartInFlight || state.agentScanCancellationInFlight) {
    throw new Error('A deep pool scan is starting, running, or stopping. Wait for it before running AI detection.');
  }
  setAIDetectionControlsRunning(true, 'Detecting…');
  clearAnalysisScopedFeatures({ render: true });
  try {
    const detection = await fetchAIDataInView({
      forceRefresh: options.forceRefresh !== false,
      source: 'selected-area-ai-scan',
      polygonCoords: coords
    });
    if (!detection?.ok) {
      if (detection?.aborted) return detection;
      throw detection?.error || new Error('Selected-area AI detection did not complete.');
    }
    showToast('AI detections loaded. Run Area analysis separately if you want an updated suitability report.', 'success');
    return detection;
  } catch (error) {
    showToast(`AI detection scan failed: ${error.message}`, 'error');
    throw error;
  } finally {
    setAIDetectionControlsRunning(false);
  }
}

// Dev-mode ✓/✗ feedback on a detection. Stores a labelled example server-side (feeds
// both the evaluation set and the future fine-tuning dataset). "wrong" also removes
// the detection from the map immediately — the user just told us it isn't a pool.
export async function sendPoolFeedback(verdict, feature) {
  const tags = feature?.tags || {};
  let lat, lon;
  if (feature.type === 'node') {
    lat = Number(feature.lat); lon = Number(feature.lon);
  } else {
    const points = Array.isArray(feature.geometry) ? feature.geometry : [];
    const lats = points.map(p => Number(p.lat)).filter(Number.isFinite);
    const lons = points.map(p => Number(p.lon)).filter(Number.isFinite);
    lat = lats.reduce((a, b) => a + b, 0) / (lats.length || 1);
    lon = lons.reduce((a, b) => a + b, 0) / (lons.length || 1);
  }
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) throw new Error('Feature has no usable position.');

  const response = await fetch('/api/feedback/pool', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ verdict, lat, lon, tags })
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Feedback failed (${response.status})`);

  if (verdict === 'wrong') {
    for (const [key, cached] of state.globalPools.entries()) {
      if (cached === feature) {
        state.globalPools.delete(key);
        break;
      }
    }
    materializeGlobalFeatureCache();
    persistSatellitePoolCache();
    renderAILayers();
  }
  const total = payload.stats?.total ?? '?';
  showToast(
    verdict === 'wrong'
      ? `Marked as not a pool — removed. (${total} feedback labels saved)`
      : `Feedback saved — thanks. (${total} labels)`,
    'success'
  );
  return payload;
}

export async function startAreaScan(area, options = {}) {
  const payload = Array.isArray(area) ? { polygon: area } : { ...(area || {}) };
  payload.passes = Math.max(3, Math.min(40, Number(options.passes || payload.passes || selectedScanPasses())));
  payload.source = options.source || payload.source || 'manual';
  payload.label = options.label || payload.label || 'Selected area';
  // Explicit user action owns the detector scope. This also replaces an old
  // server-side job after a page reload, when the browser no longer knows its id.
  payload.replace_active = options.replaceActive !== false;

  if (!Array.isArray(payload.polygon) || payload.polygon.length < 3) {
    throw new Error('Area scans require an explicit selected polygon.');
  }
  const requestedScopeKey = polygonScopeKey(payload.polygon);
  if (!requestedScopeKey || requestedScopeKey !== currentSelectionScopeKey()) {
    throw new Error('The scan polygon must match the current selected analysis area.');
  }
  if (state.agentScanStartInFlight || state.agentScanCancellationInFlight) {
    throw new Error('A selected-area scan is already starting or stopping. Wait for it to settle.');
  }
  if (state.agentScanActiveId) {
    if (state.agentScanScopeKey === requestedScopeKey) {
      setAgentScanStatus('This selected area is already being scanned.', 'running');
      setAgentScanControlsRunning(true);
      if (!state.agentScanPollTimer) startAreaScanPolling(state.agentScanActiveId);
      return { scan_id: state.agentScanActiveId, status: 'running', client_reused: true };
    }
    throw new Error('Another selected-area scan is still running. Wait for it to finish before scanning a new area.');
  }

  // Drawn-area scans keep the user's current camera; only place-name
  // searches (startPlaceScan) frame the results. Callers can force either via
  // options.autoFocus.
  state.scanCameraFollow = options.autoFocus === true;
  state.scanAutoFocused = false;
  // Preserve any selected-area overlays while the deeper pool scan runs. Scan
  // results are merged/deduplicated as they arrive.
  renderAILayers();
  const startGeneration = state.agentScanPollGeneration;
  state.agentScanStartInFlight = true;
  setAgentScanControlsRunning(true, 'Starting scan…');
  setAgentScanStatus(`Starting ${payload.passes}-pass deep pool scan…`, 'running');
  try {
    let response;
    try {
      response = await fetchWithDeadline('/api/agent/scan-area', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }, SCAN_START_TIMEOUT_MS);
    } catch (error) {
      throw new Error(describeRequestError(error, 'Area scan', requestOrigin()));
    }
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Area scan failed (${response.status})`);
    if (!result.scan_id) throw new Error('Area scan started without a scan identifier.');
    const selectionChangedDuringStart =
      state.agentScanPollGeneration !== startGeneration ||
      currentSelectionScopeKey() !== requestedScopeKey;
    const returnedDifferentScope = !scanMatchesPolygon(result, payload.polygon);
    if (selectionChangedDuringStart || returnedDifferentScope) {
      // The server may have admitted the job just before the selection changed,
      // or may have returned a pre-reload old-scope job. Attach only long enough
      // for the common retirement path to cancel it; never poll or ingest it.
      state.agentScanActiveId = result.scan_id;
      state.agentScanScopeKey = polygonScopeKey(result.polygon);
      await retireActiveAreaScan(
        selectionChangedDuringStart ? 'selection-changed-during-start' : 'replaced-different-scope'
      );
      throw new Error(
        selectionChangedDuringStart
          ? 'The old-area scan was cancelled because the selection changed while it was starting.'
          : 'The detector returned a different-area scan; it was cancelled and no results were loaded.'
      );
    }

    restoreAndClearPriorScanSnapshots();
    state.lastAgentScanId = result.scan_id;
    state.agentScanActiveId = result.scan_id;
    state.agentScanScopeKey = requestedScopeKey;
    setAgentScanControlsRunning(true);
    setAgentScanStatus(scanStatusText(result), 'running');
    startAreaScanPolling(result.scan_id);
    showToast('Deep pool scan is running for the selected area.', 'success');
    return result;
  } catch (error) {
    setAgentScanStatus(error.message, 'error');
    setAgentScanControlsRunning(false);
    throw error;
  } finally {
    state.agentScanStartInFlight = false;
  }
}

// Suburb/area search scan. Identical pipeline to startAreaScan (clear → start → poll →
// ingest → render detected pools as map polygons → update both status panels); the only
// difference is the input: a place name geocoded to a bbox server-side instead of a drawn
// polygon. This is what makes the OSIRIS area-search scan and the polygon "Run area scan"
// do the same thing.
export async function startPlaceScan(area, options = {}) {
  const place = String(area || '').trim();
  if (place.length < 2) throw new Error('Enter a suburb or area to scan.');
  const passes = Math.max(3, Math.min(40, Number(options.passes || selectedScanPasses())));
  const radiusKm = Number(options.radius_km) || 3;
  if (state.agentScanStartInFlight || state.agentScanCancellationInFlight) {
    throw new Error('Another selected-area scan is already starting or stopping. Wait before scanning a place.');
  }
  if (state.agentScanActiveId) {
    // An explicit place search selects a new rectangle. Retire the known local
    // scope first; replace_active handles a server job unknown after reload or
    // a cancellation request that could not be confirmed.
    await retireActiveAreaScan('place-scan-started');
  }

  // A place search means the user wants to be taken to the results.
  state.scanCameraFollow = true;
  state.scanAutoFocused = false;
  const startGeneration = state.agentScanPollGeneration;
  state.agentScanStartInFlight = true;
  setAgentScanControlsRunning(true, 'Starting scan…');
  setAgentScanStatus(`Geocoding "${place}" and starting ${passes}-pass scan...`, 'running');
  try {
    let response;
    try {
      response = await fetchWithDeadline('/api/agent/scan-place', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ area: place, passes, radius_km: radiusKm, replace_active: true })
      }, PLACE_SCAN_START_TIMEOUT_MS);
    } catch (error) {
      throw new Error(describeRequestError(error, 'Place scan', requestOrigin()));
    }
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Place scan failed (${response.status})`);
    if (!result.scan_id || !Array.isArray(result.polygon) || result.polygon.length < 3) {
      throw new Error('Place scan did not return a valid selected area.');
    }
    const expectedLabel = String(result.place?.display_name || place).trim().toLowerCase().slice(0, 120);
    const actualLabel = String(result.label || '').trim().toLowerCase();
    const selectionChangedDuringStart = state.agentScanPollGeneration !== startGeneration;
    const returnedDifferentScope = result.source !== 'dashboard-place' || (actualLabel && actualLabel !== expectedLabel);
    if (selectionChangedDuringStart || returnedDifferentScope) {
      state.agentScanActiveId = result.scan_id;
      state.agentScanScopeKey = polygonScopeKey(result.polygon);
      await retireActiveAreaScan(
        selectionChangedDuringStart ? 'selection-changed-during-place-start' : 'replaced-different-place'
      );
      throw new Error(
        selectionChangedDuringStart
          ? 'The place scan was cancelled because a new selection was started.'
          : 'The detector returned a different-area scan; it was cancelled and no results were loaded.'
      );
    }

    // A place search resolves to a server-selected rectangle. Make that scope
    // explicit on the map before accepting any streamed detections.
    clearAnalysisScopedFeatures({ render: false });
    state.polygon = {
      vertices: result.polygon.map(point => ({ lng: Number(point[0]), lat: Number(point[1]) }))
    };
    if (state.mapReady) {
      const geometry = state.polygon.vertices.map(point => ({ lat: point.lat, lon: point.lng }));
      setSourceData(state.map, 'draw-source', featureCollection([polygonFeature(geometry, { role: 'selection' })]));
      setSourceData(state.map, 'draw-guide-source', EMPTY_FC);
    }
    renderAILayers();

    // Move to the resolved place so its pools/leads stream into view as the scan refines.
    if (result.place && state.map) {
      const lat = Number(result.place.lat);
      const lon = Number(result.place.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) setView(state.map, lat, lon, 16);
    }
    state.lastAgentScanId = result.scan_id;
    state.agentScanActiveId = result.scan_id;
    state.agentScanScopeKey = polygonScopeKey(result.polygon);
    setAgentScanControlsRunning(true);
    setAgentScanStatus(scanStatusText(result), 'running');
    startAreaScanPolling(result.scan_id);
    showToast(`Area lead scan started for ${result.place?.display_name || place}.`, 'success');
    return result;
  } catch (error) {
    setAgentScanStatus(error.message, 'error');
    setAgentScanControlsRunning(false);
    throw error;
  } finally {
    state.agentScanStartInFlight = false;
  }
}

export async function startAreaScanFromCurrentView(options = {}) {
  if (state.polygon) {
    const coords = polygonCoordsFromSelection(state.polygon);
    if (coords.length >= 3) {
      return startAreaScan(coords, { source: 'manual-selection', label: 'Selected area', ...options });
    }
  }
  if (!state.map) {
    throw new Error('Map is not ready.');
  }
  throw new Error('Select an analysis area first. Viewport-wide AI scans are disabled.');
}

export async function fetchAIDataInView(options = {}) {
  const selectionCoords = Array.isArray(options.polygonCoords)
    ? options.polygonCoords
    : polygonCoordsFromSelection(state.polygon);
  if (options.source !== 'selected-area-ai-scan' || selectionCoords.length < 3) {
    setFeatureLayerStatus('Select an area, then choose AI detection scan.', 'empty');
    return { ok: false, error: new Error('An explicit selected analysis polygon is required.') };
  }
  if (state.agentScanActiveId || state.agentScanStartInFlight || state.agentScanCancellationInFlight) {
    const error = new Error('A deep pool scan is starting, running, or stopping. Wait before retrying selected-area inference.');
    setFeatureLayerStatus(error.message, 'error', false);
    return { ok: false, error };
  }

  // Purge cache maps if Leaflet reset state
  if (!state.cachedPoolsAndTrees) {
    state.globalPools.clear();
    state.globalTrees.clear();
    state.globalPowerlines.clear();
    state.globalBuildings.clear();
    state.globalSolarPanels.clear();
    state.fetchedBBoxes = [];
  }

  // An explicit analysis is a one-shot load of every core selected-area
  // category. Checkboxes only control rendering; otherwise enabling structures
  // or canopy after the request would misleadingly show an empty layer.
  const activeLayers = ['pools', 'buildings', 'trees'];
  
  // Power lines are only needed by the solar-conflict workflow. Bundling them
  // into every pool request makes public Overpass mirrors dramatically slower.
  if (document.getElementById('overlay-solar')?.checked) activeLayers.push('powerlines');
  const layersParam = activeLayers.join(',');

  const selectionLats = selectionCoords.map(point => Number(point[1]));
  const selectionLons = selectionCoords.map(point => Number(point[0]));
  if (![...selectionLats, ...selectionLons].every(Number.isFinite)) {
    throw new Error('The selected analysis polygon contains invalid coordinates.');
  }
  const bounds = new GeoBounds(
    Math.min(...selectionLats),
    Math.min(...selectionLons),
    Math.max(...selectionLats),
    Math.max(...selectionLons)
  );
  const selectionScopeKey = polygonScopeKey(selectionCoords);

  // Reuse only an exact in-session selection whose server detector signature matches.
  let isCached = false;
  if (!options.forceRefresh) {
    for (const cachedEntry of state.fetchedBBoxes) {
      if (cachedEntry.bounds && 
          cachedEntry.bounds.contains(bounds) && 
          cachedLayersCover(cachedEntry.layers, layersParam) &&
          cachedEntry.segmentationMode === state.segmentationMode &&
          cachedEntry.detectorPrecision === 'precise' &&
          cachedEntry.selectionScopeKey === selectionScopeKey &&
          cachedEntry.detectorSignature === state.detectorSignature) {
        isCached = true;
        break;
      }
    }
  }

  if (isCached) {
    state.aiFetchController?.abort();
    state.aiFetchController = null;
    state.aiFetchSequence += 1;
    state.isFetchingAIData = false;
    // Local cache hit — only re-materialize if data changed.
    if (state.cacheDirty) {
      state.cachedPoolsAndTrees = {
        pools: Array.from(state.globalPools.values()),
        trees: Array.from(state.globalTrees.values()),
        powerlines: Array.from(state.globalPowerlines.values())
      };
      state.cachedBuildings = {
        buildings: Array.from(state.globalBuildings.values())
      };
      state.cacheDirty = false;
    }
    const cachedStatus = featureCountText({
      pools: Array.from(state.globalPools.values()).filter(feature => featureIntersectsBounds(feature, bounds)).length,
      solar: Array.from(state.globalSolarPanels.values()).filter(feature => featureIntersectsBounds(feature, bounds)).length,
      buildings: Array.from(state.globalBuildings.values()).filter(feature => featureIntersectsBounds(feature, bounds)).length,
      trees: Array.from(state.globalTrees.values()).filter(feature => featureIntersectsBounds(feature, bounds)).length
    }, activeLayers, 'Cached');
    setFeatureLayerStatus(cachedStatus.message, cachedStatus.tone);
    updateOsirisIntelPanel(cachedStatus.message);
    renderAILayers();
    return { ok: true, cached: true };
  }

  state.aiFetchController?.abort();
  const controller = new AbortController();
  const requestSequence = ++state.aiFetchSequence;
  state.aiFetchController = controller;
  state.isFetchingAIData = true;
  setFeatureLayerStatus(
    'Analyzing the selected area with mapped data and scoped satellite inference...',
    'loading'
  );

  const currentZoom = Math.max(18, uiZoom(state.map));
  // The pool detector is calibrated for native Google z20 tiles. The basemap now allows
  // display zoom up to 22 for crisper imagery, but running detection past z20 (on Google's
  // upscaled, blurrier tiles) shifts the colour/shape statistics the strict-water gate
  // relies on and reintroduces false positives — so cap the detector's zoom at 20.
  const detectZoom = Math.min(currentZoom, 20);
  const bufferedBox = bounds;
  const bSouth = bufferedBox.getSouth();
  const bWest = bufferedBox.getWest();
  const bNorth = bufferedBox.getNorth();
  const bEast = bufferedBox.getEast();
  let response;
  try {
    const params = new URLSearchParams({
      south: String(bSouth),
      west: String(bWest),
      north: String(bNorth),
      east: String(bEast),
      zoom: String(detectZoom),
      layers: layersParam,
      segmentation_mode: state.segmentationMode,
      detector_precision: 'precise',
      analysis_polygon: JSON.stringify(selectionCoords)
    });
    if (options.forceRefresh) {
      params.set('force_refresh', '1');
    }
    // Request local/YOLO feature detection from the backend
    let res;
    res = await fetch(`/api/local/features?${params}`, { signal: controller.signal });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.error || `Feature detection request failed (${res.status})`);
    if (requestSequence !== state.aiFetchSequence) return { ok: false, aborted: true };
    response = payload;
    state.detectorSignature = payload.pipeline?.detector_signature || null;
    state.lastSuccessfulEndpoint = payload.endpoint || 'local';
  } catch (error) {
    if (error?.name !== 'AbortError') {
      console.error("Feature detection request failed:", error);
    }
    if (requestSequence === state.aiFetchSequence) {
      state.isFetchingAIData = false;
      state.aiFetchController = null;
      const explained = describeRequestError(error, 'Selected-area detection', requestOrigin());
      setFeatureLayerStatus(explained, 'error', true);
    }
    const explainedError = error?.name === 'AbortError'
      ? error
      : new Error(describeRequestError(error, 'Selected-area detection', requestOrigin()));
    return {
      ok: false,
      aborted: error?.name === 'AbortError',
      error: explainedError
    };
  }

  // Accumulate and deduplicate fetched features in global cache maps
  try {
  const responseCounts = { pools: 0, solar: 0, buildings: 0, trees: 0 };
  const clearedScopedPools = clearSatellitePoolDetectionsInBounds(bufferedBox);
  if (Array.isArray(response.elements)) {
    response.elements.forEach(e => {
      const intersectsSelection = e?.type === 'relation'
        ? (e.members || []).some(member => geometryIntersectsPolygon(member?.geometry, state.polygon))
        : featureIntersectsPolygon(e, state.polygon);
      if (state.polygon && !intersectsSelection) return;
      const uid = `${e.type}-${e.id}`;
      if (e.tags) {
        const isVegetation = ['tree', 'wood', 'scrub', 'heath', 'grassland', 'shrubbery', 'tree_row', 'wetland'].includes(e.tags.natural) ||
          ['forest', 'orchard', 'vineyard', 'meadow', 'grass', 'recreation_ground', 'greenfield'].includes(e.tags.landuse) ||
          ['garden', 'park', 'nature_reserve'].includes(e.tags.leisure);

        if (e.tags['detector:source'] === 'satellite-yolo-solar') {
          if (e.geometry) {
            state.globalSolarPanels.set(uid, e);
            responseCounts.solar += 1;
          }
        }
        else if (isPoolTagged(e)) {
          // One-shot selected-area detections: apply the confidence floor (scan-sourced
          // pools never come through this path, but keep the guard consistent).
          if (
            isSatelliteDetectedPool(e) && !isScanSourcedPool(e) &&
            poolConfidence(e) < SATELLITE_POOL_MIN_CONFIDENCE
          ) return;
          if (e.type === 'node') {
            state.globalPools.set(poolCacheUid(e), e);
            responseCounts.pools += 1;
          } else if (e.type === 'way' && e.geometry) {
            state.globalPools.set(poolCacheUid(e), e);
            responseCounts.pools += 1;
          } else if (e.type === 'relation' && e.members) {
            e.members.forEach((member, index) => {
              if (member.type === 'way' && member.role !== 'inner' && member.geometry) {
                const feature = { type: 'way', id: e.id, geometry: member.geometry, tags: e.tags };
                state.globalPools.set(`${poolCacheUid(feature)}-m${index}`, feature);
                responseCounts.pools += 1;
              }
            });
          }
        }
        else if (isVegetation) {
          if (e.type === 'relation' && Array.isArray(e.members)) {
            e.members.forEach((member, index) => {
              if (member.type === 'way' && member.role !== 'inner' && member.geometry) {
                state.globalTrees.set(`${uid}-m${index}`, { type: 'way', id: e.id, geometry: member.geometry, tags: e.tags });
                responseCounts.trees += 1;
              }
            });
          } else {
            state.globalTrees.set(uid, e);
            responseCounts.trees += 1;
          }
        }
        else if (e.tags.power === 'line') {
          state.globalPowerlines.set(uid, e);
        }
        else if (e.tags.building && !['no', 'demolished'].includes(e.tags.building)) {
          if (e.type === 'way' && isValidBuildingGeometry(e.geometry)) {
            state.globalBuildings.set(uid, e);
            responseCounts.buildings += 1;
          } else if (e.type === 'relation' && e.members) {
            e.members.forEach((member, index) => {
              if (member.type === 'way' && member.role !== 'inner' && isValidBuildingGeometry(member.geometry)) {
                state.globalBuildings.set(`${uid}-m${index}`, { type: 'way', id: e.id, geometry: member.geometry, tags: e.tags });
                responseCounts.buildings += 1;
              }
            });
          }
        }
      }
    });
  }

  const satelliteStep = response.pipeline?.steps?.find(step => step.name === 'satellite_pool_detector');
  const satelliteScanUsable = satelliteStep?.status === 'ok';
  if (satelliteScanUsable) {
    // Save only usable scans to the viewport cache. Failed imagery loads should
    // retry on the next pass instead of freezing an incorrect zero-pool result.
    state.fetchedBBoxes.push({
      bounds: bufferedBox,
      layers: layersParam,
      segmentationMode: state.segmentationMode,
      detectorPrecision: 'precise',
      selectionScopeKey,
      detectorSignature: state.detectorSignature
    });
    if (state.fetchedBBoxes.length > 24) state.fetchedBBoxes.shift();
  }

  // Materialize cached collections for rendering (mark dirty = false since we just built them)
  state.cachedPoolsAndTrees = {
    pools: Array.from(state.globalPools.values()),
    trees: Array.from(state.globalTrees.values()),
    powerlines: Array.from(state.globalPowerlines.values())
  };
  state.cachedBuildings = {
    buildings: Array.from(state.globalBuildings.values())
  };
  state.cacheDirty = false;
  state.aiFeatureRevision += 1;
  persistSatellitePoolCache();

  const scopedYolo = Boolean(satelliteStep?.diagnostics?.pool_yolo_active);
  const loadedStatus = featureCountText(
    responseCounts,
    activeLayers,
    scopedYolo ? 'Selected-area YOLO' : 'Selected-area data'
  );
  if (clearedScopedPools > 0 && responseCounts.pools === 0 && scopedYolo) {
    loadedStatus.message += ` · cleared ${clearedScopedPools} stale pool candidate${clearedScopedPools === 1 ? '' : 's'}`;
  }
  if (!satelliteScanUsable) {
    const status = satelliteStep?.status || 'unavailable';
    loadedStatus.message = `${loadedStatus.message}. Satellite pool/solar inference is ${status}; mapped features are still shown.`;
    loadedStatus.tone = 'error';
  }
  setFeatureLayerStatus(loadedStatus.message, loadedStatus.tone, !satelliteScanUsable);
  updateOsirisIntelPanel(loadedStatus.message);
  renderAILayers();
  return {
    ok: true,
    counts: responseCounts,
    satelliteInferenceUsable: satelliteScanUsable,
    detectorSignature: state.detectorSignature
  };
  } catch (error) {
    console.error("Failed to process detected features:", error);
    setFeatureLayerStatus('Detection data could not be rendered.', 'error', true);
    return { ok: false, error };
  } finally {
    if (requestSequence === state.aiFetchSequence) {
      state.isFetchingAIData = false;
      state.aiFetchController = null;
    }
  }
}

export async function fetchTerrainGridInView() {
  if (!document.getElementById('overlay-drainage')?.checked || uiZoom(state.map) < 14) return;
  const bounds = boundsFromMap(state.map);
  const coverage = state.terrainGrid?.coverage;
  if (coverage && bounds.getSouth() >= coverage.south && bounds.getWest() >= coverage.west &&
      bounds.getNorth() <= coverage.north && bounds.getEast() <= coverage.east) return;

  const latPad = (bounds.getNorth() - bounds.getSouth()) * 0.25;
  const lonPad = (bounds.getEast() - bounds.getWest()) * 0.25;
  const requestedCoverage = {
    south: bounds.getSouth() - latPad,
    west: bounds.getWest() - lonPad,
    north: bounds.getNorth() + latPad,
    east: bounds.getEast() + lonPad
  };
  const key = Object.values(requestedCoverage).map(value => value.toFixed(4)).join(':');
  if (key === state.terrainGridKey) return;

  state.terrainFetchController?.abort();
  const controller = new AbortController();
  state.terrainFetchController = controller;
  state.terrainGridKey = key;
  const params = new URLSearchParams({
    south: String(requestedCoverage.south), west: String(requestedCoverage.west),
    north: String(requestedCoverage.north), east: String(requestedCoverage.east), size: '6'
  });
  try {
    const response = await fetch(`/api/terrain/elevation-grid?${params}`, { signal: controller.signal });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Terrain request failed (${response.status})`);
    if (state.terrainFetchController !== controller) return;
    state.terrainGrid = { ...payload, coverage: requestedCoverage };
    state.aiFeatureRevision += 1;
    renderAILayers();
  } catch (error) {
    if (error?.name !== 'AbortError') console.error('Terrain grid request failed:', error);
    if (state.terrainFetchController === controller) state.terrainGridKey = null;
  } finally {
    if (state.terrainFetchController === controller) state.terrainFetchController = null;
  }
}
