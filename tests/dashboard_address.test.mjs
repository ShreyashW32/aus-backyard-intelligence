import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const dashboardPath = new URL('../aus_backyard_intelligence/static/index.html', import.meta.url);
const dashboardHtml = readFileSync(dashboardPath, 'utf8');
const scriptMatch = dashboardHtml.match(/<script>([\s\S]*?)<\/script>/);

function dashboardHarness() {
  assert.ok(scriptMatch, 'dashboard inline script should exist');
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) {
      elements.set(id, {
        id,
        value: id === 'area-query' ? 'Sydney NSW' : id === 'module' ? 'pool_3d' : '',
        innerHTML: '',
        textContent: '',
        hidden: false,
        disabled: false,
        href: '',
        addEventListener() {},
      });
    }
    return elements.get(id);
  };
  const window = {
    location: { search: '', href: '' },
    addEventListener() {},
    setInterval() {},
  };
  const context = vm.createContext({
    Array,
    Boolean,
    Error,
    JSON,
    Math,
    Number,
    Object,
    String,
    URLSearchParams,
    document: { getElementById: element, title: '' },
    encodeURIComponent,
    fetch: async () => ({ ok: true, status: 200, json: async () => ({ items: [] }) }),
    window,
  });
  vm.runInContext(scriptMatch[1], context, { filename: 'dashboard-inline.js' });
  return { context, elements };
}

function baseLead(overrides = {}) {
  return {
    title: 'Potential property near -33.87688, 151.27982',
    score: 82,
    lat: -33.8768814,
    lon: 151.2798189,
    pitch: 'Candidate property',
    evidence: {},
    contact: {},
    links: {
      google_maps: 'https://www.google.com/maps/search/?api=1&query=-33.8768814,151.2798189',
      street_view: 'https://www.google.com/maps/@?api=1&map_action=pano',
      realestate: 'https://www.realestate.com.au/property/unit-5-12-vine-street',
      domain: 'https://www.domain.com.au/property-profile/unit-5-12-vine-street',
      planning: 'https://example.gov.au/planning/unit-5-12-vine-street',
    },
    ...overrides,
  };
}

test('dashboard renders a verified unit address with provenance and coordinates', () => {
  const { context, elements } = dashboardHarness();
  context.renderAgentLeads([baseLead({
    address_details: {
      formatted: 'Unit 5, 12 Vine Street, Sydney NSW 2000',
      unit: '5',
      source: 'google',
      precision: 'rooftop',
      status: 'verified',
      verified: true,
    },
  })]);

  const rendered = elements.get('agent-leads').innerHTML;
  assert.match(rendered, /Unit 5, 12 Vine Street, Sydney NSW 2000/);
  assert.match(rendered, /google-attribution[^>]*translate="no">Google Maps<\/span>/);
  assert.match(rendered, /Rooftop · Verified/);
  assert.match(rendered, /Explicit unit 5/);
  assert.match(rendered, /-33\.876881, 151\.279819/);
  assert.match(rendered, />realestate\.com\.au</);
  assert.match(rendered, />Domain</);
  assert.match(rendered, />Planning</);
  assert.doesNotMatch(rendered, /Potential property near/);
});

test('dashboard marks approximate addresses and withholds address-dependent links', () => {
  const { context, elements } = dashboardHarness();
  context.renderAgentLeads([baseLead({
    address_details: {
      formatted: '12 Vine Street, Sydney NSW 2000',
      source: 'nominatim',
      precision: 'approximate',
      status: 'approximate',
      verified: false,
    },
  })]);

  const rendered = elements.get('agent-leads').innerHTML;
  assert.match(rendered, /12 Vine Street, Sydney NSW 2000/);
  assert.match(rendered, /OpenStreetMap Nominatim · Approximate · Not verified/);
  assert.match(rendered, />Maps</);
  assert.match(rendered, />Street View</);
  assert.doesNotMatch(rendered, />realestate\.com\.au</);
  assert.doesNotMatch(rendered, />Domain</);
  assert.doesNotMatch(rendered, />Planning</);
});

test('dashboard does not present a coordinate placeholder as an exact address', () => {
  const { context, elements } = dashboardHarness();
  context.renderAgentLeads([
    baseLead({ address_details: { source: 'pending', precision: 'pending', status: 'pending', verified: false } }),
    baseLead({
      title: '-33.87688, 151.27982',
      address_details: { source: 'google', precision: 'rooftop', status: 'verified', verified: true },
    }),
  ]);

  const rendered = elements.get('agent-leads').innerHTML;
  assert.match(rendered, /Address lookup pending/);
  assert.match(rendered, /Address not verified/);
  assert.doesNotMatch(rendered, /<h3>Potential property near/);
  assert.doesNotMatch(rendered, /<h3>-33\.87688, 151\.27982<\/h3>/);
  assert.doesNotMatch(rendered, />realestate\.com\.au</);
});

test('dashboard escapes address provenance and rejects non-http lead links', () => {
  const { context, elements } = dashboardHarness();
  context.renderAgentLeads([baseLead({
    address_details: {
      formatted: '<img src=x onerror=alert(1)> Unit 5',
      source: '<script>alert(1)</script>',
      precision: 'rooftop',
      verified: true,
    },
    links: { google_maps: 'javascript:alert(1)' },
  })]);

  const rendered = elements.get('agent-leads').innerHTML;
  assert.match(rendered, /&lt;img src=x onerror=alert\(1\)&gt; Unit 5/);
  assert.match(rendered, /&lt;Script&gt;Alert\(1\)&lt;\/Script&gt;/);
  assert.doesNotMatch(rendered, /<img|<script|javascript:/i);
});

test('dashboard explains whether exact Google address verification is configured', () => {
  const { context, elements } = dashboardHarness();

  context.renderAddressLookupStatus({ google_configured: false });
  assert.match(elements.get('address-config-status').textContent, /not configured/i);
  assert.match(elements.get('address-config-status').textContent, /units are never guessed/i);

  context.renderAddressLookupStatus({ google_configured: true });
  assert.match(elements.get('address-config-status').textContent, /verification enabled/i);
  assert.match(elements.get('address-config-status').className, /configured/);
});

test('verified premise address says when no selected unit was identified', () => {
  const { context, elements } = dashboardHarness();
  context.renderAgentLeads([baseLead({
    address_details: {
      formatted: '8 Example Street, Sydney NSW 2000',
      unit: '',
      source: 'google',
      precision: 'rooftop',
      status: 'verified',
      verified: true,
    },
  })]);

  const rendered = elements.get('agent-leads').innerHTML;
  assert.match(rendered, /8 Example Street, Sydney NSW 2000/);
  assert.match(rendered, /Unit not identified/);
  assert.doesNotMatch(rendered, /Explicit unit/);
});
