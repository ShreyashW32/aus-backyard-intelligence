import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildPropertyModelSpec,
  geometryRings,
  modelSelectionFingerprint,
  packModelSession,
  propertyModelEligibility,
  unpackModelSession
} from '../aus_backyard_intelligence/static/js/model3d_geometry.js';

const irregularParcel = {
  type: 'Polygon',
  coordinates: [[
    [153.1, -27.2],
    [153.1004, -27.2001],
    [153.1005, -27.2004],
    [153.1002, -27.2006],
    [153.0999, -27.2003],
    [153.1, -27.2]
  ]]
};

function propertyPayload(id = 'PROP-A') {
  return {
    property_id: id,
    address: '10 Test Street',
    suburb: 'Redcliffe',
    lga: 'Moreton Bay',
    elevations: { building_height_m: 8.9, building_height_source: 'derived estimate' },
    features: { land_area_sqm: 720, building_area_sqm: 180, backyard_area_sqm: 310 },
    geometry_provenance: {
      geometry_source: 'geocoded-anchor',
      scope: 'selected-property-only',
      height_source: 'derived estimate',
      roof_source: 'procedural massing',
      note: 'Illustrative geometry.'
    },
    geometries: {
      parcel_geom: irregularParcel,
      building_geom: {
        type: 'Polygon',
        coordinates: [[[153.10005, -27.20018], [153.10031, -27.2002], [153.1003, -27.20039], [153.10004, -27.20037], [153.10005, -27.20018]]]
      },
      backyard_geom: {
        type: 'Polygon',
        coordinates: [[[153.10001, -27.2004], [153.10035, -27.20042], [153.10025, -27.20056], [153.09998, -27.20032], [153.10001, -27.2004]]]
      }
    }
  };
}

test('preserves the exact irregular selected parcel instead of its bounding box', () => {
  const spec = buildPropertyModelSpec(propertyPayload());
  assert.deepEqual(spec.parcelRings[0], irregularParcel.coordinates[0].slice(0, -1));
  assert.equal(spec.parcelRings[0].length, 5);
});

test('uses the selected property building height and provenance', () => {
  const spec = buildPropertyModelSpec(propertyPayload());
  assert.equal(spec.propertyId, 'PROP-A');
  assert.equal(spec.buildings.length, 1);
  assert.equal(spec.buildings[0].heightM, 8.9);
  assert.equal(spec.provenance.geometrySource, 'geocoded-anchor');
  assert.equal(spec.provenance.scope, 'selected-property-only');
});

test('rejects stale model session data from another selected property', () => {
  const packed = packModelSession(propertyPayload('PROP-A'));
  assert.equal(unpackModelSession(packed, { propertyId: 'PROP-B' }), null);
  assert.equal(unpackModelSession(packed, { propertyId: 'PROP-A' }).property_id, 'PROP-A');
});

test('requires an exact matching fingerprint for selected-area session data', () => {
  const payload = { polygon: irregularParcel.coordinates[0], geometries: { parcel_geom: irregularParcel } };
  const selection = modelSelectionFingerprint(payload);
  const packed = packModelSession(payload);
  assert.ok(unpackModelSession(packed, { selection }));
  assert.equal(unpackModelSession(packed, { selection: 'area:stale000' }), null);
});

test('normalizes Polygon, Feature, FeatureCollection and MultiPolygon inputs', () => {
  assert.equal(geometryRings(irregularParcel).length, 1);
  assert.equal(geometryRings({ type: 'Feature', properties: {}, geometry: irregularParcel }).length, 1);
  assert.equal(geometryRings({
    type: 'FeatureCollection',
    features: [{ type: 'Feature', properties: {}, geometry: irregularParcel }]
  }).length, 1);
  assert.equal(geometryRings({
    type: 'MultiPolygon',
    coordinates: [irregularParcel.coordinates, irregularParcel.coordinates]
  }).length, 2);
});

test('drops invalid pool and tree coordinates without generating placeholder geometry', () => {
  const payload = propertyPayload();
  payload.geometries.pools = [{ geometry: [{ lon: 'bad', lat: -27.2 }] }];
  payload.geometries.trees = [{ lon: 153.1, lat: -27.2 }, { lon: null, lat: 'bad' }];
  const spec = buildPropertyModelSpec(payload);
  assert.equal(spec.poolRings.length, 0);
  assert.deepEqual(spec.treePoints, [[153.1, -27.2]]);
  assert.equal(spec.parcelRings.length, 1);
});

test('only a selected property with parcel and building geometry is 3D eligible', () => {
  assert.equal(propertyModelEligibility(propertyPayload()).eligible, true);
  const selectedArea = propertyPayload();
  delete selectedArea.property_id;
  assert.equal(propertyModelEligibility(selectedArea).reason, 'area_not_property');
  const noBuilding = propertyPayload();
  noBuilding.geometries.building_geom = null;
  assert.equal(propertyModelEligibility(noBuilding).reason, 'missing_building');
});
