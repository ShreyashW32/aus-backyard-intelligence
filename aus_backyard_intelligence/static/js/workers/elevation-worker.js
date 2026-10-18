// Background contour builder: receives a raw elevation grid, runs the
// marching-squares pass off the UI thread, and returns a ready-to-load
// GeoJSON FeatureCollection for MapLibre's contours-source.
import { buildElevationContours } from '../services/feature_algorithms.js';

self.onmessage = (event) => {
  const { grid, levels = 7, key } = event.data || {};
  let features = [];
  try {
    const contours = buildElevationContours(grid, levels);
    features = contours.map(contour => ({
      type: 'Feature',
      properties: { level: Math.round(contour.level * 10) / 10 },
      geometry: {
        type: 'LineString',
        coordinates: contour.points.map(point => [point.lon, point.lat])
      }
    }));
  } catch (error) {
    // A malformed grid yields an empty overlay rather than a dead worker.
    features = [];
  }
  self.postMessage({ key, featureCollection: { type: 'FeatureCollection', features } });
};
