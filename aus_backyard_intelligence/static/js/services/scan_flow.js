// Small, DOM-free helpers shared by the selected-area analysis and background
// scan client. Keeping these deterministic makes scope matching and request
// error reporting independently testable.

const TERMINAL_SCAN_STATUSES = new Set([
  'completed',
  'completed_with_warnings',
  'failed',
  'cancelled'
]);

function normalizedRing(coords) {
  if (!Array.isArray(coords)) return [];
  const ring = [];
  for (const point of coords) {
    if (!Array.isArray(point) || point.length < 2) return [];
    const lon = Number(point[0]);
    const lat = Number(point[1]);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) return [];
    ring.push([Number(lon.toFixed(7)), Number(lat.toFixed(7))]);
  }

  if (ring.length > 1) {
    const first = ring[0];
    const last = ring[ring.length - 1];
    if (first[0] === last[0] && first[1] === last[1]) ring.pop();
  }
  return ring;
}

function rotations(points) {
  const variants = [];
  for (let index = 0; index < points.length; index += 1) {
    variants.push(points.slice(index).concat(points.slice(0, index)));
  }
  return variants;
}

export function polygonScopeKey(coords) {
  const ring = normalizedRing(coords);
  if (ring.length < 3) return '';
  const forward = rotations(ring);
  const reverse = rotations([...ring].reverse());
  return [...forward, ...reverse]
    .map(candidate => JSON.stringify(candidate))
    .sort()[0];
}

export function scanIsTerminal(status) {
  return TERMINAL_SCAN_STATUSES.has(String(status || '').toLowerCase());
}

export function scanMatchesPolygon(job, coords) {
  const expected = polygonScopeKey(coords);
  const actual = polygonScopeKey(job?.polygon);
  return Boolean(expected && actual && expected === actual);
}

export function scanPollIsCurrent(activeScanId, activeGeneration, scanId, pollGeneration) {
  return Boolean(
    scanId &&
    activeScanId === scanId &&
    Number(activeGeneration) === Number(pollGeneration)
  );
}

// Reconcile one cache collection against an authoritative scan snapshot. Only
// entries currently owned by this scan may be removed; any value that existed
// before the scan overlaid the key is restored.
export function reconcileOwnedSnapshot(cache, ownership, desiredKeys, scanId) {
  let changed = false;
  for (const [key, prior] of ownership.entries()) {
    if (desiredKeys.has(key)) continue;
    const current = cache.get(key);
    if (current?._clientScanId === scanId) {
      if (prior.hadPrevious) cache.set(key, prior.previous);
      else cache.delete(key);
      changed = true;
    }
    ownership.delete(key);
  }
  return changed;
}

export function describeRequestError(error, action = 'Request', origin = '') {
  const raw = String(error?.message || error || 'request failed').trim();
  if (error?.name === 'TimeoutError') {
    return `${action} timed out. The server may still be finishing the request; retry safely after checking its status.`;
  }
  const networkFailure = error?.name === 'TypeError' ||
    /failed to fetch|networkerror|network request failed|load failed/i.test(raw);
  if (networkFailure) {
    const target = origin ? ` at ${origin}` : '';
    return `${action} could not reach the project server${target}. The server may have stopped; reload after it is running again.`;
  }
  return `${action} failed: ${raw}`;
}
