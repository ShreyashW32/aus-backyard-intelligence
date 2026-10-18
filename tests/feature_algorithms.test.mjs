import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildElevationContours,
  estimateSolarScore,
  geometryAreaSqm,
  inferBuildingHeight,
  inferCanopyRadius,
  isValidBuildingGeometry,
  isValidPoolGeometry,
  isPoolTagged,
  normalizeGeometry,
  poolConfidence
} from '../aus_backyard_intelligence/static/js/services/feature_algorithms.js';
import { normalizeStartupZoom } from '../aus_backyard_intelligence/static/js/geo.js';

const square = [
  { lat: -27.4700, lon: 153.0300 },
  { lat: -27.4700, lon: 153.0301 },
  { lat: -27.4699, lon: 153.0301 },
  { lat: -27.4699, lon: 153.0300 },
  { lat: -27.4700, lon: 153.0300 }
];

test('honours low URL zoom levels within the map range', () => {
  assert.equal(normalizeStartupZoom(10), 10);
  assert.equal(normalizeStartupZoom(3), 3);
  assert.equal(normalizeStartupZoom(1), 3);
  assert.equal(normalizeStartupZoom(30), 22);
  assert.equal(normalizeStartupZoom(Number.NaN), 17);
});

test('normalizes closed geometry without duplicate vertices', () => {
  assert.equal(normalizeGeometry(square).length, 4);
  assert.ok(geometryAreaSqm(square) > 90);
});

test('validates realistic pools and rejects tiny or giant shapes', () => {
  assert.equal(isValidPoolGeometry(square), true);
  assert.equal(isValidPoolGeometry(square.map(point => ({ lat: point.lat, lon: 153 + (point.lon - 153) / 100 }))), false);
  assert.equal(poolConfidence({ tags: { leisure: 'swimming_pool' }, geometry: square }), 0.98);
  assert.equal(poolConfidence({ tags: { leisure: 'water_park', sport: 'swimming' }, geometry: square }), 0.78);
  assert.equal(poolConfidence({ tags: { water: 'pool' }, geometry: square }), 0.84);
  assert.equal(poolConfidence({ tags: { name: 'Boat Pool' }, geometry: square }), 0.56);
  assert.equal(isPoolTagged({ tags: { amenity: 'public_bath' }, geometry: square }), true);
  assert.equal(isPoolTagged({ tags: { name: 'Pool Bar', amenity: 'bar' }, geometry: square }), false);
  assert.equal(poolConfidence({ tags: { leisure: 'swimming_pool', abandoned: 'yes' }, geometry: square }), 0);
});

test('allows larger public aquatic venues without allowing generic giant private pools', () => {
  const largeVenue = [
    { lat: -27.4700, lon: 153.0300 },
    { lat: -27.4700, lon: 153.0310 },
    { lat: -27.4690, lon: 153.0310 },
    { lat: -27.4690, lon: 153.0300 },
    { lat: -27.4700, lon: 153.0300 }
  ];
  assert.equal(isValidPoolGeometry(largeVenue), false);
  assert.equal(isValidPoolGeometry(largeVenue, { leisure: 'water_park' }), true);
});

test('validates building footprints and infers deterministic tagged heights', () => {
  assert.equal(isValidBuildingGeometry(square), true);
  assert.equal(inferBuildingHeight({ height: '30 ft' }).toFixed(2), '9.14');
  assert.equal(inferBuildingHeight({ 'building:levels': '3', 'roof:levels': '1' }), 10.8);
  assert.equal(inferBuildingHeight({ building: 'garage' }), 3);
});

test('uses crown diameter or height for canopy radius', () => {
  assert.equal(inferCanopyRadius({ diameter_crown: '12 m' }), 6);
  assert.equal(inferCanopyRadius({ height: '20' }), 5.6);
});

test('solar screening is deterministic and reduced by nearby canopy', () => {
  const feature = { tags: { 'roof:shape': 'flat' } };
  assert.equal(estimateSolarScore(feature, 0), 82);
  assert.equal(estimateSolarScore(feature, 5), 62);
  assert.equal(estimateSolarScore(feature, 5), estimateSolarScore(feature, 5));
});

test('builds contour segments from an elevation grid', () => {
  const contours = buildElevationContours({
    latitudes: [-27.48, -27.47, -27.46],
    longitudes: [153.02, 153.03, 153.04],
    elevations: [[5, 10, 15], [10, 15, 20], [15, 20, 25]]
  }, 4);
  assert.ok(contours.length > 0);
  assert.ok(contours.every(contour => contour.points.length === 2));
});
