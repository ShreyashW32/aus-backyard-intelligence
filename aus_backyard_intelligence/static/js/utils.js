import { state } from './state.js';

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  })[character]);
}

export function ratingLabel(v) {
  if (v >= 80) return { text: 'Excellent', cls: 'excellent' };
  if (v >= 65) return { text: 'Good', cls: 'good' };
  if (v >= 45) return { text: 'Moderate', cls: 'moderate' };
  return { text: 'Poor', cls: 'poor' };
}

export function ratingColor(v) {
  return v >= 80 ? '#10b981' : v >= 65 ? '#3b82f6' : v >= 45 ? '#f59e0b' : '#ef4444';
}

export function getActiveCoords() {
  if (state.currentData) {
    const centroid = state.currentData.centroid || (state.currentData.polygon ? state.currentData.polygon[0] : null);
    if (centroid) {
      return { lat: centroid[1], lng: centroid[0] };
    }
  }
  const center = state.map.getCenter();
  return { lat: center.lat, lng: center.lng };
}

// Ring extraction: accepts the plain selection object ({ vertices: [...] }) or a
// bare vertex array. Every point is normalized to { lat, lng }.
export function polygonRing(polygon) {
  const raw = Array.isArray(polygon) ? polygon : polygon?.vertices;
  if (!Array.isArray(raw)) return [];
  return raw
    .map(point => ({ lat: Number(point.lat), lng: Number(point.lng ?? point.lon) }))
    .filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lng));
}

export function isPointInPolygon(point, polygon) {
  const ring = polygonRing(polygon);
  if (ring.length < 3) return false;
  const x = Number(point.lat), y = Number(point.lng ?? point.lon);
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].lat, yi = ring[i].lng;
    const xj = ring[j].lat, yj = ring[j].lng;
    const intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function orientation(a, b, c) {
  const value = (b.lng - a.lng) * (c.lat - b.lat) - (b.lat - a.lat) * (c.lng - b.lng);
  if (Math.abs(value) < 1e-12) return 0;
  return value > 0 ? 1 : 2;
}

function onSegment(a, b, c) {
  return b.lat <= Math.max(a.lat, c.lat) + 1e-12 &&
    b.lat + 1e-12 >= Math.min(a.lat, c.lat) &&
    b.lng <= Math.max(a.lng, c.lng) + 1e-12 &&
    b.lng + 1e-12 >= Math.min(a.lng, c.lng);
}

function segmentsIntersect(a, b, c, d) {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  if (o1 !== o2 && o3 !== o4) return true;
  return (o1 === 0 && onSegment(a, c, b)) ||
    (o2 === 0 && onSegment(a, d, b)) ||
    (o3 === 0 && onSegment(c, a, d)) ||
    (o4 === 0 && onSegment(c, b, d));
}

export function geometryIntersectsPolygon(geometry, polygon) {
  if (!Array.isArray(geometry) || geometry.length === 0) return false;
  const ring = polygonRing(polygon);
  if (ring.length < 3) return false;

  const points = geometry
    .map(point => ({ lat: Number(point.lat), lng: Number(point.lon ?? point.lng) }))
    .filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lng));
  if (points.length === 0) return false;
  if (points.some(point => isPointInPolygon(point, ring))) return true;

  if (points.length >= 3 && ring.some(point => isPointInPolygon(point, points))) return true;

  for (let i = 0; i < points.length - 1; i++) {
    for (let j = 0; j < ring.length; j++) {
      if (segmentsIntersect(points[i], points[i + 1], ring[j], ring[(j + 1) % ring.length])) {
        return true;
      }
    }
  }
  return false;
}

export function featureIntersectsPolygon(feature, polygon) {
  if (!feature || !polygon) return false;
  if (feature.type === 'node') {
    const lat = Number(feature.lat);
    const lon = Number(feature.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
    return isPointInPolygon({ lat, lng: lon }, polygon);
  }
  return geometryIntersectsPolygon(feature.geometry, polygon);
}
