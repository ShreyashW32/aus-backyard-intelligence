import test from 'node:test';
import assert from 'node:assert/strict';

import {
  describeRequestError,
  polygonScopeKey,
  reconcileOwnedSnapshot,
  scanIsTerminal,
  scanMatchesPolygon,
  scanPollIsCurrent
} from '../aus_backyard_intelligence/static/js/services/scan_flow.js';

const polygon = [
  [153.01, -33.88],
  [153.02, -33.88],
  [153.02, -33.87],
  [153.01, -33.87]
];

test('polygon scope is stable across rotation, winding and a closing point', () => {
  const rotated = [polygon[2], polygon[3], polygon[0], polygon[1]];
  const reversedClosed = [...polygon].reverse().concat([polygon[3]]);
  assert.equal(polygonScopeKey(rotated), polygonScopeKey(polygon));
  assert.equal(polygonScopeKey(reversedClosed), polygonScopeKey(polygon));
  assert.equal(polygonScopeKey([[1, 2], [3, 4]]), '');
  assert.equal(polygonScopeKey([...polygon, ['bad', -33.87]]), '');
});

test('scan scope must match the active selected polygon', () => {
  assert.equal(scanMatchesPolygon({ polygon }, [...polygon].reverse()), true);
  assert.equal(scanMatchesPolygon({ polygon }, polygon.map(([x, y]) => [x + 0.01, y])), false);
  assert.equal(scanMatchesPolygon({ polygon: null }, polygon), false);
});

test('recognises every terminal scan status', () => {
  assert.equal(scanIsTerminal('completed'), true);
  assert.equal(scanIsTerminal('completed_with_warnings'), true);
  assert.equal(scanIsTerminal('failed'), true);
  assert.equal(scanIsTerminal('cancelled'), true);
  assert.equal(scanIsTerminal('running'), false);
});

test('rejects late polling from a retired scan or generation', () => {
  assert.equal(scanPollIsCurrent('scan-new', 8, 'scan-new', 8), true);
  assert.equal(scanPollIsCurrent(null, 9, 'scan-old', 8), false);
  assert.equal(scanPollIsCurrent('scan-new', 9, 'scan-old', 9), false);
  assert.equal(scanPollIsCurrent('scan-new', 9, 'scan-new', 8), false);
});

test('snapshot reconciliation removes only this scan and restores prior data', () => {
  const mapped = { id: 'mapped-pool' };
  const otherScan = { id: 'other-scan-pool', _clientScanId: 'scan-other' };
  const cache = new Map([
    ['overlaid', { id: 'scan-version', _clientScanId: 'scan-current' }],
    ['withdrawn', { id: 'withdrawn', _clientScanId: 'scan-current' }],
    ['other', otherScan],
    ['kept', { id: 'kept', _clientScanId: 'scan-current' }]
  ]);
  const ownership = new Map([
    ['overlaid', { hadPrevious: true, previous: mapped }],
    ['withdrawn', { hadPrevious: false, previous: undefined }],
    ['other', { hadPrevious: false, previous: undefined }],
    ['kept', { hadPrevious: false, previous: undefined }]
  ]);
  const changed = reconcileOwnedSnapshot(cache, ownership, new Set(['kept']), 'scan-current');
  assert.equal(changed, true);
  assert.equal(cache.get('overlaid'), mapped);
  assert.equal(cache.has('withdrawn'), false);
  assert.equal(cache.get('other'), otherScan);
  assert.equal(cache.get('kept').id, 'kept');
  assert.deepEqual([...ownership.keys()], ['kept']);
});

test('sequential same-area scans unwind to non-scan baseline', () => {
  const mapped = { id: 'mapped-baseline' };
  const scanA = { id: 'candidate-a', _clientScanId: 'scan-a' };
  const cache = new Map([['pool-key', scanA]]);
  const ownershipA = new Map([
    ['pool-key', { hadPrevious: true, previous: mapped }]
  ]);

  // Admission of scan B retires all ownership from scan A first.
  reconcileOwnedSnapshot(cache, ownershipA, new Set(), 'scan-a');
  assert.equal(cache.get('pool-key'), mapped);

  const ownershipB = new Map([
    ['pool-key', { hadPrevious: true, previous: cache.get('pool-key') }]
  ]);
  cache.set('pool-key', { id: 'candidate-b', _clientScanId: 'scan-b' });
  reconcileOwnedSnapshot(cache, ownershipB, new Set(), 'scan-b');
  assert.equal(cache.get('pool-key'), mapped);
  assert.notEqual(cache.get('pool-key'), scanA);
});

test('turns opaque browser fetch failures into actionable server guidance', () => {
  const message = describeRequestError(new TypeError('Failed to fetch'), 'Area scan', 'http://127.0.0.1:8017');
  assert.match(message, /could not reach the project server/i);
  assert.match(message, /127\.0\.0\.1:8017/);
  assert.match(message, /reload/i);
  assert.equal(
    describeRequestError(new Error('bad polygon'), 'Area scan'),
    'Area scan failed: bad polygon'
  );
  const timeout = new Error('Request exceeded 30 seconds.');
  timeout.name = 'TimeoutError';
  assert.match(describeRequestError(timeout, 'Area scan'), /timed out.*retry safely/i);
});
