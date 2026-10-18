// Leaflet-free geometry primitives for the MapLibre GL viewer.
//
// MapLibre works in [longitude, latitude] order (GeoJSON), while all of the
// app's cached feature data and the backend API speak {lat, lon}. Everything
// that crosses that boundary goes through this module so the inversion happens
// in exactly one place.

const METERS_PER_DEGREE_LAT = 111320;

// Leaflet-style latitude-first bounds with the exact method surface the app
// logic already depends on (getSouth/getWest/getNorth/getEast, contains for
// both points and other bounds). MapLibre's LngLatBounds only contains points.
export class GeoBounds {
  constructor(south, west, north, east) {
    this.south = Math.min(south, north);
    this.north = Math.max(south, north);
    this.west = Math.min(west, east);
    this.east = Math.max(west, east);
  }

  getSouth() { return this.south; }
  getWest() { return this.west; }
  getNorth() { return this.north; }
  getEast() { return this.east; }

  // Accepts {lat, lon}, {lat, lng}, or another GeoBounds (Leaflet semantics).
  contains(target) {
    if (!target) return false;
    if (target instanceof GeoBounds || (target.south !== undefined && target.north !== undefined)) {
      return target.south >= this.south && target.north <= this.north &&
        target.west >= this.west && target.east <= this.east;
    }
    const lat = Number(target.lat);
    const lon = Number(target.lon ?? target.lng);
    return Number.isFinite(lat) && Number.isFinite(lon) &&
      lat >= this.south && lat <= this.north && lon >= this.west && lon <= this.east;
  }

  pad(ratio) {
    const latPad = (this.north - this.south) * ratio;
    const lonPad = (this.east - this.west) * ratio;
    return new GeoBounds(this.south - latPad, this.west - lonPad, this.north + latPad, this.east + lonPad);
  }

  center() {
    return { lat: (this.south + this.north) / 2, lon: (this.west + this.east) / 2 };
  }

  // MapLibre fitBounds argument: [[west, south], [east, north]].
  toLngLat() {
    return [[this.west, this.south], [this.east, this.north]];
  }
}

export function boundsFromMap(map) {
  const b = map.getBounds();
  return new GeoBounds(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
}

// ---------------------------------------------------------------------------
// Zoom convention bridge.
//
// The whole app (thresholds, localStorage, /3d links, the backend detector's
// slippy-tile zoom cap) is calibrated in Leaflet/256px-tile zoom levels.
// MapLibre's style zoom for the same view of a 256px raster source is one
// level lower, so translate at the boundary instead of re-tuning every
// threshold in the codebase.
const ZOOM_OFFSET = 1;

export function uiZoom(map) {
  return map.getZoom() + ZOOM_OFFSET;
}

export function toStyleZoom(z) {
  return z - ZOOM_OFFSET;
}

export function normalizeStartupZoom(value, fallback = 17) {
  return Number.isFinite(value) ? Math.max(3, Math.min(22, Math.round(value))) : fallback;
}

// Leaflet setView replacement: latitude-first, ui-zoom in, no animation.
export function setView(map, lat, lon, zoom) {
  map.jumpTo({ center: [lon, lat], zoom: toStyleZoom(zoom) });
}

export function flyToView(map, lat, lon, zoom, options = {}) {
  map.flyTo({ center: [lon, lat], zoom: toStyleZoom(zoom), ...options });
}

export function fitGeoBounds(map, bounds, { padding = 40, maxZoom = null } = {}) {
  const options = { padding, duration: 700 };
  if (maxZoom !== null) options.maxZoom = toStyleZoom(maxZoom);
  map.fitBounds(bounds.toLngLat(), options);
}

// ---------------------------------------------------------------------------
// Distance / area math (open equations only — Haversine + local equirectangular).

export function haversineMeters(a, b) {
  const lat1 = Number(a.lat), lon1 = Number(a.lon ?? a.lng);
  const lat2 = Number(b.lat), lon2 = Number(b.lon ?? b.lng);
  const rad = Math.PI / 180;
  const dLat = (lat2 - lat1) * rad;
  const dLon = (lon2 - lon1) * rad;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(s)));
}

// Shoelace area over a local metric projection; points are {lat, lon|lng}.
export function ringAreaSqm(points) {
  if (!Array.isArray(points) || points.length < 3) return 0;
  const meanLat = points.reduce((sum, p) => sum + Number(p.lat), 0) / points.length;
  const mPerLon = METERS_PER_DEGREE_LAT * Math.cos(meanLat * Math.PI / 180);
  let twiceArea = 0;
  for (let i = 0; i < points.length; i += 1) {
    const c = points[i];
    const n = points[(i + 1) % points.length];
    const cx = Number(c.lon ?? c.lng) * mPerLon, cy = Number(c.lat) * METERS_PER_DEGREE_LAT;
    const nx = Number(n.lon ?? n.lng) * mPerLon, ny = Number(n.lat) * METERS_PER_DEGREE_LAT;
    twiceArea += cx * ny - nx * cy;
  }
  return Math.abs(twiceArea) / 2;
}

// ---------------------------------------------------------------------------
// GeoJSON builders (single inversion point: {lat, lon} → [lon, lat]).

export function lonLat(point) {
  return [Number(point.lon ?? point.lng), Number(point.lat)];
}

export function polygonFeature(points, properties = {}) {
  const ring = points.map(lonLat);
  if (ring.length >= 3) {
    const [x0, y0] = ring[0];
    const [xn, yn] = ring[ring.length - 1];
    if (x0 !== xn || y0 !== yn) ring.push([x0, y0]);
  }
  return { type: 'Feature', properties, geometry: { type: 'Polygon', coordinates: [ring] } };
}

export function lineFeature(points, properties = {}) {
  return { type: 'Feature', properties, geometry: { type: 'LineString', coordinates: points.map(lonLat) } };
}

export function pointFeature(point, properties = {}) {
  return { type: 'Feature', properties, geometry: { type: 'Point', coordinates: lonLat(point) } };
}

// Approximate circle polygon (used where a real ground radius in meters is
// required — canopies — which a screen-space circle layer cannot represent).
export function circlePolygonFeature(center, radiusMeters, properties = {}, segments = 18) {
  const lat = Number(center.lat);
  const lon = Number(center.lon ?? center.lng);
  const dLat = radiusMeters / METERS_PER_DEGREE_LAT;
  const dLon = radiusMeters / (METERS_PER_DEGREE_LAT * Math.max(0.2, Math.cos(lat * Math.PI / 180)));
  const ring = [];
  for (let i = 0; i <= segments; i += 1) {
    const angle = (i / segments) * Math.PI * 2;
    ring.push([lon + Math.cos(angle) * dLon, lat + Math.sin(angle) * dLat]);
  }
  return { type: 'Feature', properties, geometry: { type: 'Polygon', coordinates: [ring] } };
}

export function featureCollection(features) {
  return { type: 'FeatureCollection', features: features.filter(Boolean) };
}

export const EMPTY_FC = Object.freeze({ type: 'FeatureCollection', features: [] });

// Recursive bbox of any GeoJSON geometry/feature (replaces L.geoJSON().getBounds()).
export function geojsonBounds(geojson) {
  let south = 90, west = 180, north = -90, east = -180, found = false;
  const visit = (coords) => {
    if (!Array.isArray(coords)) return;
    if (coords.length >= 2 && typeof coords[0] === 'number' && typeof coords[1] === 'number') {
      const [lon, lat] = coords;
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        south = Math.min(south, lat); north = Math.max(north, lat);
        west = Math.min(west, lon); east = Math.max(east, lon);
        found = true;
      }
      return;
    }
    coords.forEach(visit);
  };
  const visitGeometry = (geometry) => {
    if (!geometry) return;
    if (geometry.type === 'GeometryCollection') {
      (geometry.geometries || []).forEach(visitGeometry);
    } else {
      visit(geometry.coordinates);
    }
  };
  if (geojson) {
    if (geojson.type === 'FeatureCollection') (geojson.features || []).forEach(f => visitGeometry(f.geometry));
    else if (geojson.type === 'Feature') visitGeometry(geojson.geometry);
    else visitGeometry(geojson);
  }
  return found ? new GeoBounds(south, west, north, east) : null;
}

// ---------------------------------------------------------------------------
// MapLibre source/image helpers.

export function setSourceData(map, sourceId, data) {
  const source = map.getSource(sourceId);
  if (source) source.setData(data || EMPTY_FC);
}

export function setLayersVisibility(map, layerIds, visible) {
  layerIds.forEach(id => {
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
    }
  });
}

// Rasterize an inline SVG into a map sprite image (keyless, no CDN sprites).
export function addSvgImage(map, id, svg, size = 32) {
  return new Promise(resolve => {
    if (map.hasImage(id)) { resolve(true); return; }
    const image = new Image(size, size);
    image.onload = () => {
      try {
        if (!map.hasImage(id)) map.addImage(id, image, { pixelRatio: 2 });
        resolve(true);
      } catch (e) { resolve(false); }
    };
    image.onerror = () => resolve(false);
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  });
}
