// Localhost-only rendered-layer regression harness. Loaded dynamically when the
// viewer URL contains ?e2e_rendered_layers=1; never part of normal application flow.
import { state } from './state.js';
import { setOsintEnabled, setSelectedPropertyPoint } from './components/osint.js';

const LAYER_IDS = [
  'contours-line', 'canopy-fill', 'trees-fill', 'solar-suit-fill',
  'solar-panels-fill', 'solar-panels-line', 'buildings-fill', 'buildings-outline',
  'buildings-3d-extrusion', 'ranked-parcels-fill', 'ranked-parcels-line',
  'ranked-markers-circle', 'selected-property-fill', 'selected-property-line',
  'manual-pools-fill', 'manual-pools-line', 'pools-fill', 'pools-outline-glow',
  'pools-outline', 'pools-points', 'draw-polygon-fill', 'draw-polygon-line',
  'draw-preview-line', 'draw-guide-line', 'draw-vertices', 'locate-circle',
  'osint-fires-fill', 'osint-fires-outline', 'osint-fires-line', 'osint-fires-point',
  'osint-links-line', 'osint-incidents-ripple', 'osint-incidents-core',
  'osint-quakes-ripple', 'osint-quakes-core', 'osint-cameras-glow',
  'osint-cameras-icon', 'osint-trails-line', 'osint-flights-icon'
];

const fc = features => ({ type: 'FeatureCollection', features });
const polygon = (properties = {}) => ({
  type: 'Feature', properties, geometry: {
    type: 'Polygon',
    coordinates: [[[153.0245, -27.2484], [153.0255, -27.2484], [153.0255, -27.2476], [153.0245, -27.2476], [153.0245, -27.2484]]]
  }
});
const line = (properties = {}) => ({
  type: 'Feature', properties,
  geometry: { type: 'LineString', coordinates: [[153.0244, -27.2483], [153.0256, -27.2477]] }
});
const point = (properties = {}, coordinates = [153.025, -27.248]) => ({
  type: 'Feature', properties, geometry: { type: 'Point', coordinates }
});

function waitForMapReady() {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 10000;
    const poll = () => {
      if (state.mapReady && state.map?.getSource('pools-source')) return resolve();
      if (Date.now() >= deadline) return reject(new Error('Map overlay sources did not become ready'));
      setTimeout(poll, 50);
    };
    poll();
  });
}

export async function runRenderedLayerSelfTest() {
  await waitForMapReady();
  const map = state.map;
  await setOsintEnabled(true);

  const sourceFixtures = {
    'contours-source': fc([line()]),
    'canopy-source': fc([polygon()]),
    'trees-source': fc([polygon()]),
    'solar-suit-source': fc([polygon({ color: '#facc15' })]),
    'solar-panels-source': fc([polygon()]),
    'buildings-source': fc([polygon({ height: 8, levels: 2 })]),
    'ranked-parcels-source': fc([polygon()]),
    'ranked-markers-source': fc([point()]),
    'selected-property-source': fc([polygon({ role: 'backyard' })]),
    'manual-pools-source': fc([polygon()]),
    'pools-source': fc([polygon()]),
    'pools-points-source': fc([point({ isDetected: true })]),
    'draw-source': fc([polygon({ role: 'polygon' }), line({ role: 'preview' }), point({ role: 'vertex' })]),
    'draw-guide-source': fc([line()]),
    'locate-source': fc([point()]),
    'osint-fires-source': fc([polygon(), point({}, [153.0252, -27.248])]),
    'osint-incidents-source': fc([point({}, [153.0251, -27.248])]),
    'osint-quakes-source': fc([point({ mag: 2.5 }, [153.0249, -27.248])]),
    'osint-cameras-source': fc([point({}, [153.0251, -27.2479])]),
    'osint-flights-source': fc([point({ track: 45, callsign: 'TEST1' }, [153.0248, -27.2478])]),
    'osint-trails-source': fc([line()]),
    'osint-links-source': fc([line()])
  };
  setSelectedPropertyPoint({ lat: -27.2481, lon: 153.025 });
  for (const [sourceId, data] of Object.entries(sourceFixtures)) map.getSource(sourceId).setData(data);
  for (const layerId of LAYER_IDS) map.setLayoutProperty(layerId, 'visibility', 'visible');
  map.triggerRepaint();

  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  // Symbol placement (especially freshly rasterized SVG icons) can lag GeoJSON
  // source updates by more than two animation frames on a cold/software-rendered
  // browser. Poll the exact rendered-feature assertion instead of racing it.
  const deadline = Date.now() + 5000;
  let counts;
  do {
    await new Promise(resolve => setTimeout(resolve, 100));
    map.triggerRepaint();
    counts = Object.fromEntries(
      LAYER_IDS.map(layerId => [layerId, map.queryRenderedFeatures({ layers: [layerId] }).length])
    );
  } while (Object.values(counts).some(count => count <= 0) && Date.now() < deadline);
  const invisible = Object.entries(counts).filter(([, count]) => count <= 0).map(([layerId]) => layerId);
  const result = {
    passed: invisible.length === 0,
    layer_count: LAYER_IDS.length,
    invisible,
    counts,
    images: {
      camera: map.hasImage('osint-camera'),
      flight: map.hasImage('osint-flight')
    },
    source_loaded: {
      camera: map.isSourceLoaded('osint-cameras-source'),
      flight: map.isSourceLoaded('osint-flights-source')
    }
  };
  window.__renderedLayerTestResults = result;
  document.documentElement.dataset.renderedLayerTest = result.passed ? 'passed' : 'failed';
  document.documentElement.dataset.renderedLayerTestCount = String(result.layer_count);
  document.documentElement.dataset.renderedLayerTestInvisible = invisible.join(',');
  document.documentElement.dataset.renderedLayerTestImages = [
    `camera:${map.hasImage('osint-camera')}`,
    `flight:${map.hasImage('osint-flight')}`
  ].join(',');
  return result;
}
