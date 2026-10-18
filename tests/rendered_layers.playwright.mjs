import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

import { chromium } from 'playwright';

const require = createRequire(import.meta.url);
const maplibreJs = require.resolve('maplibre-gl');
const maplibreCss = join(dirname(maplibreJs), 'maplibre-gl.css');
const port = Number(process.env.ABI_E2E_PORT || 8099);
const baseUrl = process.env.ABI_E2E_BASE_URL || `http://127.0.0.1:${port}`;
const python = process.env.ABI_E2E_PYTHON || (process.platform === 'win32' ? '.venv\\Scripts\\python.exe' : 'python');
let server = null;

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`${baseUrl}/api/health`);
      if (response.ok) return;
    } catch {}
    await delay(125);
  }
  throw new Error(`Demo server did not become ready at ${baseUrl}`);
}

function osintFixture(pathname) {
  const point = coordinates => ({ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates } });
  const polygon = {
    type: 'Feature', properties: {}, geometry: {
      type: 'Polygon', coordinates: [[[153.0245, -27.2484], [153.0255, -27.2484], [153.0255, -27.2476], [153.0245, -27.2476], [153.0245, -27.2484]]]
    }
  };
  if (pathname.endsWith('/wildfires')) {
    return { type: 'FeatureCollection', features: [polygon, point([153.025, -27.248])] };
  }
  if (pathname.endsWith('/incidents')) {
    return { type: 'FeatureCollection', features: [point([153.0251, -27.248])] };
  }
  if (pathname.includes('/earthquakes')) {
    return { items: [{ lat: -27.248, lon: 153.0249, mag: 2.5, place: 'fixture' }] };
  }
  if (pathname.includes('/cameras')) {
    return { items: [{ lat: -27.2479, lon: 153.0251, id: 1, name: 'fixture camera' }] };
  }
  if (pathname.includes('/flights')) {
    return { source: 'fixture', items: [{ lat: -27.2478, lon: 153.0248, icao24: 'abc123', callsign: 'TEST1', track_deg: 45 }] };
  }
  return {};
}

if (!process.env.ABI_E2E_BASE_URL) {
  server = spawn(python, ['-m', 'aus_backyard_intelligence', '--host', '127.0.0.1', '--port', String(port)], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, ABI_DEV_API: '0', QLDR_SKIP_SATELLITE_DETECTOR_FOR_TESTS: '1' }
  });
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => {
    // Raster basemap requests are intentionally aborted so this regression test
    // is deterministic and offline. MapLibre logs those aborts as console errors;
    // only fail on errors emitted by our application/test harness.
    if (message.type() === 'error' && /\[(?:BOOT|E2E)\]/.test(message.text())) {
      pageErrors.push(message.text());
    }
  });

  await page.route('https://**/*', async route => {
    const url = route.request().url();
    if (url.endsWith('/maplibre-gl.js')) return route.fulfill({ path: maplibreJs, contentType: 'application/javascript' });
    if (url.endsWith('/maplibre-gl.css')) return route.fulfill({ path: maplibreCss, contentType: 'text/css' });
    return route.abort();
  });
  await page.route('**/api/osint/**', route => {
    const body = JSON.stringify(osintFixture(new URL(route.request().url()).pathname));
    return route.fulfill({ status: 200, body, contentType: 'application/json' });
  });

  await page.goto(
    `${baseUrl}/?lat=-27.248&lon=153.025&zoom=18&e2e_rendered_layers=1`,
    { waitUntil: 'domcontentloaded' }
  );
  await page.waitForFunction(() => window.__renderedLayerTestResults, null, { timeout: 15000 });
  const results = await page.evaluate(() => window.__renderedLayerTestResults);

  assert.equal(
    results.passed,
    true,
    `Seeded layers not rendered: ${JSON.stringify(results)}; browser errors: ${pageErrors.join('; ')}`
  );
  assert.deepEqual(pageErrors, [], `Browser errors: ${pageErrors.join('; ')}`);
  process.stdout.write(`Rendered-layer regression passed for ${results.layer_count} layers.\n`);
} finally {
  await browser?.close();
  if (server) {
    server.kill();
    await Promise.race([new Promise(resolve => server.once('exit', resolve)), delay(1500)]);
  }
}
