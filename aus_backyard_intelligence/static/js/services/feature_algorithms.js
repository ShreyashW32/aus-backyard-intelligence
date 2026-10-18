const METERS_PER_DEGREE_LAT = 111320;

function finiteCoordinate(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function normalizeGeometry(geometry) {
  if (!Array.isArray(geometry)) return [];
  if (geometry._normalized) return geometry._normalized;

  const normalized = [];
  for (const point of geometry) {
    const lat = finiteCoordinate(point?.lat);
    const lon = finiteCoordinate(point?.lon ?? point?.lng);
    if (lat === null || lon === null || Math.abs(lat) > 90 || Math.abs(lon) > 180) continue;
    const previous = normalized[normalized.length - 1];
    if (previous && Math.abs(previous.lat - lat) < 1e-10 && Math.abs(previous.lon - lon) < 1e-10) continue;
    normalized.push({ lat, lon });
  }
  if (normalized.length > 2) {
    const first = normalized[0];
    const last = normalized[normalized.length - 1];
    if (Math.abs(first.lat - last.lat) < 1e-10 && Math.abs(first.lon - last.lon) < 1e-10) {
      normalized.pop();
    }
  }

  Object.defineProperty(geometry, '_normalized', {
    value: normalized,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return normalized;
}

export function geometryCentroid(geometry) {
  if (!geometry) return null;
  if (geometry._centroid) return geometry._centroid;

  const points = normalizeGeometry(geometry);
  let centroid = null;
  if (!points.length) {
    centroid = null;
  } else if (points.length < 3) {
    const sum = points.reduce((result, point) => ({ lat: result.lat + point.lat, lon: result.lon + point.lon }), { lat: 0, lon: 0 });
    centroid = { lat: sum.lat / points.length, lon: sum.lon / points.length };
  } else {
    let twiceArea = 0;
    let weightedLon = 0;
    let weightedLat = 0;
    for (let index = 0; index < points.length; index += 1) {
      const current = points[index];
      const next = points[(index + 1) % points.length];
      const cross = current.lon * next.lat - next.lon * current.lat;
      twiceArea += cross;
      weightedLon += (current.lon + next.lon) * cross;
      weightedLat += (current.lat + next.lat) * cross;
    }
    if (Math.abs(twiceArea) < 1e-14) {
      const sum = points.reduce((result, point) => ({ lat: result.lat + point.lat, lon: result.lon + point.lon }), { lat: 0, lon: 0 });
      centroid = { lat: sum.lat / points.length, lon: sum.lon / points.length };
    } else {
      centroid = {
        lat: weightedLat / (3 * twiceArea),
        lon: weightedLon / (3 * twiceArea)
      };
    }
  }

  Object.defineProperty(geometry, '_centroid', {
    value: centroid,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return centroid;
}

export function geometryAreaSqm(geometry) {
  if (!geometry) return 0;
  if (typeof geometry._area === 'number') return geometry._area;

  const points = normalizeGeometry(geometry);
  let area = 0;
  if (points.length >= 3) {
    const meanLat = points.reduce((sum, point) => sum + point.lat, 0) / points.length;
    const metersPerDegreeLon = METERS_PER_DEGREE_LAT * Math.cos(meanLat * Math.PI / 180);
    let twiceArea = 0;
    for (let index = 0; index < points.length; index += 1) {
      const current = points[index];
      const next = points[(index + 1) % points.length];
      const currentX = current.lon * metersPerDegreeLon;
      const currentY = current.lat * METERS_PER_DEGREE_LAT;
      const nextX = next.lon * metersPerDegreeLon;
      const nextY = next.lat * METERS_PER_DEGREE_LAT;
      twiceArea += currentX * nextY - nextX * currentY;
    }
    area = Math.abs(twiceArea) / 2;
  }

  Object.defineProperty(geometry, '_area', {
    value: area,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return area;
}

export function geometryExtent(geometry) {
  if (!geometry) return null;
  if (geometry._extent) return geometry._extent;

  const points = normalizeGeometry(geometry);
  let extent = null;
  if (points.length) {
    extent = points.reduce((ext, point) => ({
      south: Math.min(ext.south, point.lat),
      west: Math.min(ext.west, point.lon),
      north: Math.max(ext.north, point.lat),
      east: Math.max(ext.east, point.lon)
    }), { south: points[0].lat, west: points[0].lon, north: points[0].lat, east: points[0].lon });
  }

  Object.defineProperty(geometry, '_extent', {
    value: extent,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return extent;
}

export function parseLengthMeters(value) {
  if (value === null || value === undefined || value === '') return null;
  const text = String(value).trim().toLowerCase().replace(',', '.');
  const number = Number.parseFloat(text);
  if (!Number.isFinite(number) || number < 0) return null;
  if (/\b(ft|feet|foot)\b/.test(text) || text.includes("'")) return number * 0.3048;
  if (/\b(cm|centimet)/.test(text)) return number / 100;
  return number;
}

export function inferBuildingHeight(tags = {}) {
  if (!tags) return 5;
  if (typeof tags._buildingHeight === 'number') return tags._buildingHeight;

  const explicitHeight = parseLengthMeters(tags.height);
  let height;
  if (explicitHeight !== null) {
    height = clamp(explicitHeight, 2.4, 250);
  } else {
    const levels = Number.parseFloat(tags['building:levels']);
    const roofLevels = Number.parseFloat(tags['roof:levels']);
    if (Number.isFinite(levels) && levels > 0) {
      height = clamp(levels * 3 + (Number.isFinite(roofLevels) ? roofLevels * 1.8 : 0), 2.4, 250);
    } else {
      const kind = String(tags.building || '').toLowerCase();
      if (['garage', 'garages', 'shed', 'carport', 'roof'].includes(kind)) height = 3;
      else if (['apartments', 'commercial', 'office', 'retail', 'industrial'].includes(kind)) height = 9;
      else if (['house', 'detached', 'semidetached_house', 'residential'].includes(kind)) height = 6;
      else height = 5;
    }
  }

  Object.defineProperty(tags, '_buildingHeight', {
    value: height,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return height;
}

export function inferCanopyRadius(tags = {}) {
  if (!tags) return 4.5;
  if (typeof tags._canopyRadius === 'number') return tags._canopyRadius;

  const crownDiameter = parseLengthMeters(tags.diameter_crown ?? tags['crown:diameter']);
  let radius;
  if (crownDiameter !== null) {
    radius = Math.round(clamp(crownDiameter / 2, 1.5, 20) * 10) / 10;
  } else {
    const height = parseLengthMeters(tags.height);
    if (height !== null) {
      radius = Math.round(clamp(height * 0.28, 2, 14) * 10) / 10;
    } else {
      radius = 4.5;
    }
  }

  Object.defineProperty(tags, '_canopyRadius', {
    value: radius,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return radius;
}

export function poolConfidence(feature) {
  if (!feature) return 0;
  if (typeof feature._poolConfidence === 'number') return feature._poolConfidence;

  const tags = feature?.tags || {};
  const detectorSource = String(tags['detector:source'] || '');
  const detectorMethod = String(tags['detector:method'] || '');
  let confidence;
  if (
    tags['detector:yolo'] === 'yes' ||
    tags['detector:yolo'] === 'validator' ||
    detectorSource === 'satellite-yolo' ||
    detectorMethod === 'pool-trained-yolo'
  ) {
    confidence = parseFloat(tags['confidence:yolo']) || 0.92;
  } else if (tags['confidence:cv']) {
    confidence = parseFloat(tags['confidence:cv']) || 0.72;
  } else if (tags['confidence:tag']) {
    confidence = parseFloat(tags['confidence:tag']) || 0.84;
  } else if (tags.disused === 'yes' || tags.abandoned === 'yes' || tags.swimming_pool === 'no') {
    confidence = 0;
  } else {
    const leisure = String(tags.leisure || '').toLowerCase();
    const sport = String(tags.sport || '').toLowerCase();
    const amenity = String(tags.amenity || '').toLowerCase();
    const tourism = String(tags.tourism || '').toLowerCase();
    const water = String(tags.water || '').toLowerCase();
    const hasSwimmingPoolTag = Boolean(tags.swimming_pool && tags.swimming_pool !== 'no');
    let conf = 0;
    if (leisure === 'swimming_pool') conf = 0.98;
    else if (hasSwimmingPoolTag) conf = 0.9;
    else if (['pool', 'swimming_pool', 'wading_pool'].includes(water)) conf = 0.84;
    else if (leisure === 'water_park' || leisure === 'swimming_area') conf = 0.78;
    else if (tourism === 'water_park') conf = 0.76;
    else if (sport === 'swimming') conf = 0.72;
    else if (amenity === 'public_bath' || amenity === 'swimming_pool') conf = 0.7;
    else if (water === 'lagoon' && hasAquaticName(tags)) conf = 0.64;
    else if (hasAquaticName(tags) && canUseAquaticNameHint(tags)) conf = 0.56;
    const area = geometryAreaSqm(feature?.geometry);
    if (area > 0 && area < 4) conf *= 0.35;
    if (area > maximumPoolAreaSqm(tags)) conf *= 0.5;
    confidence = clamp(conf, 0, 1);
  }

  Object.defineProperty(feature, '_poolConfidence', {
    value: confidence,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return confidence;
}

export function isPoolTagged(feature) {
  if (!feature) return false;
  return poolConfidence(feature) > 0;
}

function maximumPoolAreaSqm(tags = {}) {
  const leisure = String(tags.leisure || '').toLowerCase();
  const sport = String(tags.sport || '').toLowerCase();
  const amenity = String(tags.amenity || '').toLowerCase();
  const tourism = String(tags.tourism || '').toLowerCase();
  const water = String(tags.water || '').toLowerCase();
  const isPublicAquaticVenue = leisure === 'water_park' ||
    leisure === 'swimming_area' ||
    tourism === 'water_park' ||
    sport === 'swimming' ||
    amenity === 'public_bath' ||
    amenity === 'swimming_pool' ||
    water === 'lagoon';
  return isPublicAquaticVenue ? 25000 : 5000;
}

function hasAquaticName(tags = {}) {
  const name = String(tags.name || tags['name:en'] || '').toLowerCase();
  return /\b(pool|swim|swimming|aquatic|lagoon|water\s*park|waterpark)\b/.test(name) ||
    name.includes('streets beach') ||
    name.includes('boat pool');
}

function canUseAquaticNameHint(tags = {}) {
  const blockedAmenities = new Set([
    'bar', 'bbq', 'biergarten', 'cafe', 'fast_food', 'food_court', 'ice_cream',
    'parking', 'pub', 'restaurant', 'school'
  ]);
  if (blockedAmenities.has(String(tags.amenity || '').toLowerCase())) return false;
  if (tags.highway || tags.railway || tags.route || tags.public_transport || tags.boundary) return false;
  if (tags.shop || tags.office || tags.craft) return false;
  if (tags.building && !tags.leisure && !tags.sport && !tags.amenity && !tags.tourism && !tags.water) return false;
  return true;
}

export function isValidPoolGeometry(geometry, tags = {}) {
  if (!geometry) return false;
  const maximumArea = maximumPoolAreaSqm(tags);
  if (typeof geometry._isValidPool === 'boolean' && geometry._isValidPoolMaximumArea === maximumArea) return geometry._isValidPool;

  const area = geometryAreaSqm(geometry);
  const isValid = normalizeGeometry(geometry).length >= 3 && area >= 4 && area <= maximumArea;

  Object.defineProperty(geometry, '_isValidPool', {
    value: isValid,
    writable: true,
    enumerable: false,
    configurable: true
  });
  Object.defineProperty(geometry, '_isValidPoolMaximumArea', {
    value: maximumArea,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return isValid;
}

export function isValidBuildingGeometry(geometry) {
  if (!geometry) return false;
  if (typeof geometry._isValidBuilding === 'boolean') return geometry._isValidBuilding;

  const area = geometryAreaSqm(geometry);
  const isValid = normalizeGeometry(geometry).length >= 3 && area >= 4 && area <= 250000;

  Object.defineProperty(geometry, '_isValidBuilding', {
    value: isValid,
    writable: true,
    enumerable: false,
    configurable: true
  });

  return isValid;
}

export function estimateSolarScore(feature, nearbyTreeCount = 0) {
  const tags = feature?.tags || {};
  const roofShape = String(tags['roof:shape'] || '').toLowerCase();
  let score = roofShape === 'flat' ? 82 : roofShape ? 74 : 68;
  score -= Math.min(nearbyTreeCount, 8) * 4;
  if (tags.solar === 'yes' || tags['generator:source'] === 'solar') score += 10;
  return Math.round(clamp(score, 20, 95));
}

function interpolateContourPoint(first, second, level) {
  const difference = second.elevation - first.elevation;
  const fraction = Math.abs(difference) < 1e-9 ? 0.5 : clamp((level - first.elevation) / difference, 0, 1);
  return {
    lat: first.lat + (second.lat - first.lat) * fraction,
    lon: first.lon + (second.lon - first.lon) * fraction
  };
}

export function buildElevationContours(grid, requestedLevels = 6) {
  const latitudes = grid?.latitudes;
  const longitudes = grid?.longitudes;
  const elevations = grid?.elevations;
  if (!Array.isArray(latitudes) || !Array.isArray(longitudes) || !Array.isArray(elevations) ||
      latitudes.length < 2 || longitudes.length < 2 || elevations.length !== latitudes.length) return [];
  if (elevations.some(row => !Array.isArray(row) || row.length !== longitudes.length || row.some(value => !Number.isFinite(Number(value))))) return [];

  const values = elevations.flat().map(Number);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum - minimum < 0.5) return [];
  const levelCount = Math.max(2, Math.min(Number.parseInt(requestedLevels) || 6, 12));
  const levels = Array.from({ length: levelCount }, (_, index) => minimum + (maximum - minimum) * (index + 1) / (levelCount + 1));
  const contours = [];

  for (const level of levels) {
    for (let row = 0; row < latitudes.length - 1; row += 1) {
      for (let column = 0; column < longitudes.length - 1; column += 1) {
        const southWest = { lat: latitudes[row], lon: longitudes[column], elevation: Number(elevations[row][column]) };
        const southEast = { lat: latitudes[row], lon: longitudes[column + 1], elevation: Number(elevations[row][column + 1]) };
        const northEast = { lat: latitudes[row + 1], lon: longitudes[column + 1], elevation: Number(elevations[row + 1][column + 1]) };
        const northWest = { lat: latitudes[row + 1], lon: longitudes[column], elevation: Number(elevations[row + 1][column]) };
        const edges = [[southWest, southEast], [southEast, northEast], [northEast, northWest], [northWest, southWest]];
        const intersections = [];
        for (const [first, second] of edges) {
          const crosses = (first.elevation < level && second.elevation >= level) ||
            (second.elevation < level && first.elevation >= level);
          if (crosses) intersections.push(interpolateContourPoint(first, second, level));
        }
        if (intersections.length === 2) {
          contours.push({ level, points: intersections });
        } else if (intersections.length === 4) {
          contours.push({ level, points: intersections.slice(0, 2) });
          contours.push({ level, points: intersections.slice(2, 4) });
        }
      }
    }
  }
  return contours;
}
