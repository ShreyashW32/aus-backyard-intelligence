const EARTH_RADIUS_M = 6378137;

function finiteCoordinate(value) {
  if (Array.isArray(value) && value.length >= 2) {
    if (value[0] === null || value[0] === '' || value[1] === null || value[1] === '') return null;
    const lon = Number(value[0]);
    const lat = Number(value[1]);
    return Number.isFinite(lon) && Number.isFinite(lat) ? [lon, lat] : null;
  }
  if (value && typeof value === 'object') {
    const rawLon = value.lon ?? value.lng ?? value.longitude;
    const rawLat = value.lat ?? value.latitude;
    if (rawLon === null || rawLon === '' || rawLat === null || rawLat === '') return null;
    const lon = Number(rawLon);
    const lat = Number(rawLat);
    return Number.isFinite(lon) && Number.isFinite(lat) ? [lon, lat] : null;
  }
  return null;
}

export function normalizeRing(values) {
  if (!Array.isArray(values)) return [];
  const ring = values.map(finiteCoordinate).filter(Boolean);
  if (ring.length > 1) {
    const first = ring[0];
    const last = ring[ring.length - 1];
    if (first[0] === last[0] && first[1] === last[1]) ring.pop();
  }
  return ring.length >= 3 ? ring : [];
}

export function geometryRings(value) {
  if (!value) return [];
  if (value.type === 'FeatureCollection') {
    return (value.features || []).flatMap(feature => geometryRings(feature));
  }
  if (value.type === 'Feature') return geometryRings(value.geometry);
  if (value.type === 'Polygon') {
    const ring = normalizeRing(value.coordinates?.[0]);
    return ring.length ? [ring] : [];
  }
  if (value.type === 'MultiPolygon') {
    return (value.coordinates || [])
      .map(polygon => normalizeRing(polygon?.[0]))
      .filter(ring => ring.length);
  }
  if (Array.isArray(value)) {
    const direct = normalizeRing(value);
    if (direct.length) return [direct];
    if (value.length && Array.isArray(value[0])) {
      return value.map(normalizeRing).filter(ring => ring.length);
    }
  }
  if (value.geometry) return geometryRings(value.geometry);
  return [];
}

export function boundsForRings(rings) {
  const points = (rings || []).flat();
  if (!points.length) return null;
  const lons = points.map(point => point[0]);
  const lats = points.map(point => point[1]);
  return {
    west: Math.min(...lons),
    east: Math.max(...lons),
    south: Math.min(...lats),
    north: Math.max(...lats),
    centerLon: (Math.min(...lons) + Math.max(...lons)) / 2,
    centerLat: (Math.min(...lats) + Math.max(...lats)) / 2
  };
}

export function geoToLocal(lon, lat, origin) {
  const centerLon = Number(origin?.centerLon) || 0;
  const centerLat = Number(origin?.centerLat) || 0;
  const x = (Number(lon) - centerLon) * Math.PI / 180 * EARTH_RADIUS_M *
    Math.cos(centerLat * Math.PI / 180);
  const north = (Number(lat) - centerLat) * Math.PI / 180 * EARTH_RADIUS_M;
  return { x, north };
}

export function ringAreaSquareMetres(ring) {
  const bounds = boundsForRings([ring]);
  if (!bounds) return 0;
  const points = ring.map(([lon, lat]) => geoToLocal(lon, lat, bounds));
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current.x * next.north - next.x * current.north;
  }
  return Math.abs(twiceArea) / 2;
}

function stableCoordinates(data) {
  const geometries = data?.geometries || {};
  const parcel = geometryRings(geometries.parcel_geom)[0];
  const polygon = normalizeRing(data?.polygon);
  return parcel?.length ? parcel : polygon;
}

export function modelSelectionFingerprint(data) {
  if (data?.property_id) return `property:${String(data.property_id)}`;
  const ring = stableCoordinates(data);
  if (!ring.length) return null;
  const value = ring.map(([lon, lat]) => `${lon.toFixed(7)},${lat.toFixed(7)}`).join('|');
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `area:${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

export function packModelSession(data) {
  return JSON.stringify({
    version: 1,
    createdAt: Date.now(),
    selection: modelSelectionFingerprint(data),
    propertyId: data?.property_id ? String(data.property_id) : null,
    payload: data
  });
}

export function unpackModelSession(raw, { propertyId = null, selection = null } = {}) {
  if (!raw) return null;
  let envelope;
  try {
    envelope = JSON.parse(raw);
  } catch {
    return null;
  }
  if (envelope?.version !== 1 || !envelope.payload) return null;
  const createdAt = Number(envelope.createdAt);
  if (!Number.isFinite(createdAt) || Date.now() - createdAt > 4 * 60 * 60 * 1000) return null;
  const expectedPropertyId = propertyId ? String(propertyId) : null;
  if (expectedPropertyId && envelope.propertyId !== expectedPropertyId) return null;
  if (selection && envelope.selection !== selection) return null;
  if (!expectedPropertyId && !selection) return null;
  if (modelSelectionFingerprint(envelope.payload) !== envelope.selection) return null;
  return envelope.payload;
}

function clampHeight(value, fallback = 6.4) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(2.4, Math.min(numeric, 40)) : fallback;
}

function buildingHeight(item, fallback) {
  const tags = item?.tags || item?.properties || {};
  const explicit = Number(tags.height ?? tags['building:height']);
  if (Number.isFinite(explicit)) return clampHeight(explicit, fallback);
  const levels = Number(tags.levels ?? tags['building:levels']);
  return Number.isFinite(levels) ? clampHeight(levels * 3.1, fallback) : fallback;
}

function ringItems(values) {
  if (!Array.isArray(values)) return [];
  return values.flatMap(item => {
    const rings = geometryRings(item?.geometry ?? item);
    return rings.map(ring => ({ ring, source: item }));
  });
}

function finiteMetric(...values) {
  for (const value of values) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  return null;
}

export function buildPropertyModelSpec(data) {
  const payload = data && typeof data === 'object' ? data : {};
  const geometries = payload.geometries || {};
  const parcelRings = geometryRings(geometries.parcel_geom);
  const fallbackHeight = clampHeight(payload.elevations?.building_height_m, 6.4);

  let buildingItems = geometryRings(geometries.building_geom)
    .map(ring => ({ ring, heightM: fallbackHeight }));
  if (!buildingItems.length) {
    buildingItems = ringItems(geometries.buildings).map(({ ring, source }) => ({
      ring,
      heightM: buildingHeight(source, fallbackHeight)
    }));
  }

  const backyardRings = geometryRings(geometries.backyard_geom);
  const poolRings = ringItems(geometries.pools).map(item => item.ring);
  const treePoints = (Array.isArray(geometries.trees) ? geometries.trees : [])
    .map(finiteCoordinate)
    .filter(Boolean);
  const allRings = [...parcelRings, ...buildingItems.map(item => item.ring), ...backyardRings, ...poolRings];
  const bounds = boundsForRings(parcelRings.length ? parcelRings : allRings);
  const parcelArea = finiteMetric(
    payload.features?.total_area_sqm,
    payload.features?.land_area_sqm,
    payload.area_sqm,
    parcelRings[0] ? ringAreaSquareMetres(parcelRings[0]) : null
  );
  const buildingArea = buildingItems.reduce((sum, item) => sum + ringAreaSquareMetres(item.ring), 0);
  const backyardArea = finiteMetric(
    payload.features?.backyard_area_sqm,
    backyardRings[0] ? ringAreaSquareMetres(backyardRings[0]) : null
  );

  const provenancePayload = payload.geometry_provenance;
  const provenance = provenancePayload && typeof provenancePayload === 'object'
    ? {
        geometrySource: String(provenancePayload.geometry_source || 'unknown'),
        scope: String(provenancePayload.scope || 'selected-area-only'),
        heightSource: String(provenancePayload.height_source || 'unknown'),
        roofSource: String(provenancePayload.roof_source || 'procedural massing'),
        note: String(provenancePayload.note || '')
      }
    : {
        geometrySource: 'analysis-geometry',
        scope: payload.property_id ? 'selected-property-only' : 'selected-area-only',
        heightSource: payload.elevations?.building_height_source || 'analysis estimate',
        roofSource: 'procedural massing',
        note: String(provenancePayload || 'Geometry supplied by the current analysis payload.')
      };

  return {
    propertyId: payload.property_id ? String(payload.property_id) : null,
    address: String(payload.address || 'Selected area'),
    locationLabel: [payload.suburb, payload.lga].filter(Boolean).join(', '),
    explanation: String(payload.explanation || ''),
    parcelRings,
    buildings: buildingItems,
    backyardRings,
    poolRings,
    treePoints,
    bounds,
    metrics: {
      parcelAreaSqm: parcelArea,
      buildingAreaSqm: buildingArea || null,
      backyardAreaSqm: backyardArea,
      buildingHeightM: buildingItems.length ? Math.max(...buildingItems.map(item => item.heightM)) : null,
      slopePct: finiteMetric(payload.features?.slope_pct, payload.slope_pct),
      reliefM: finiteMetric(payload.features?.relief_m, payload.relief_m),
      baseElevationM: finiteMetric(payload.elevations?.base_elev_m),
      vegetationPct: finiteMetric(payload.features?.vegetation_pct, payload.vegetation_pct)
    },
    provenance
  };
}

export function propertyModelEligibility(data) {
  const spec = buildPropertyModelSpec(data);
  if (!spec.propertyId) return { eligible: false, reason: 'area_not_property', spec };
  if (!spec.parcelRings.length || !spec.bounds) return { eligible: false, reason: 'missing_parcel', spec };
  if (!spec.buildings.length) return { eligible: false, reason: 'missing_building', spec };
  return { eligible: true, reason: null, spec };
}
