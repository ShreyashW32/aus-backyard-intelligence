import { initMap } from './components/map.js';
import { initControls } from './components/controls.js';
import { initDrawingShortcuts } from './components/drawing.js';
import { initUIElements } from './components/ui.js';
import { initOsint } from './components/osint.js';
import { loadRankedProperties, selectPropertyById } from './services/api.js';

// Boot application
document.addEventListener('DOMContentLoaded', () => {
  console.log('[BOOT] viewer.html modular loading (MapLibre GL build r30)...');

  // Each boot step is isolated so a failure in one (e.g. a blocked browser API)
  // cannot leave the map or the rest of the UI unrendered.
  const step = (name, fn) => {
    try {
      fn();
      console.log(`[BOOT] ${name}() completed successfully`);
    } catch (err) {
      console.error(`[BOOT] ${name}() failed:`, err);
    }
  };

  step('initMap', initMap);                            // 1. MapLibre GL map + base imagery
  step('initUIElements', initUIElements);              // 2. UI interactions & resizers
  step('initControls', initControls);                  // 3. Custom controls & camera
  step('initDrawingShortcuts', initDrawingShortcuts);  // 4. Drawing shortcuts
  step('initOsint', initOsint);                        // 5. Tactical OSINT fusion layer
  step('loadRankedProperties', loadRankedProperties);  // 6. Ranks if param exists

  const testRequested = new URLSearchParams(window.location.search).get('e2e_rendered_layers') === '1';
  const isLocalhost = ['127.0.0.1', 'localhost', '::1'].includes(window.location.hostname);
  if (testRequested && isLocalhost) {
    import('./rendered_layer_test.js')
      .then(module => module.runRenderedLayerSelfTest())
      .then(result => console.log('[E2E] rendered-layer result', result))
      .catch(error => {
        window.__renderedLayerTestResults = { passed: false, error: String(error?.message || error) };
        document.documentElement.dataset.renderedLayerTest = 'failed';
        document.documentElement.dataset.renderedLayerTestError = String(error?.message || error);
        console.error('[E2E] rendered-layer test failed', error);
      });
  }

  console.log('[BOOT] Backyard lead intelligence map loaded');
});

window.addEventListener('message', async (event) => {
  if (event.origin !== window.location.origin) return;
  const data = event.data;
  if (data && data.action === 'selectProperty') {
    const propId = data.id;
    selectPropertyById(propId);
  }
});
