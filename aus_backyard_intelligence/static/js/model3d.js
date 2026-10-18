import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  geoToLocal,
  propertyModelEligibility,
  ringAreaSquareMetres,
  unpackModelSession
} from './model3d_geometry.js';

const params = new URLSearchParams(window.location.search);
const propertyId = params.get('propertyId');
const selection = params.get('selection');
const canvas = document.getElementById('scene-canvas');
const loader = document.getElementById('loader');
const errorState = document.getElementById('error-state');
const modelViewport = document.getElementById('model-viewport');

let scene;
let camera;
let renderer;
let controls;
let modelRoot;
let modelCenter = new THREE.Vector3();
let modelSpan = 40;
const structureLines = [];

if (params.get('embed') === 'true') {
  const topBar = document.querySelector('.top-bar');
  if (topBar) topBar.style.display = 'none';
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function formatMetric(value, unit, digits = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits)} ${unit}` : '—';
}

function sourceLabel(source) {
  return ({
    real: 'Source footprint',
    'geocoded-anchor': 'Illustrative geocoded geometry',
    'suburb-anchor': 'Illustrative suburb geometry',
    'analysis-geometry': 'Current analysis geometry'
  })[source] || 'Geometry source unverified';
}

function showError(title, message) {
  setText('page-title', 'Select a property to build 3D');
  setText('scope-badge', 'Property required');
  setText('error-title', title);
  setText('error-message', message);
  errorState.classList.add('visible');
  loader.classList.add('hidden');
  document.documentElement.dataset.modelStatus = 'error';
  window.__property3DManifest = { ready: false, error: message, usedPlaceholder: false };
}

function readScopedSession() {
  try {
    return unpackModelSession(sessionStorage.getItem('current3DData'), { propertyId, selection }) ||
      unpackModelSession(localStorage.getItem('abi_current_3d_selection'), { propertyId, selection });
  } catch {
    return null;
  }
}

async function loadSelectedPayload() {
  const cached = readScopedSession();
  if (propertyId) {
    try {
      const response = await fetch(`/api/viewer/property/${encodeURIComponent(propertyId)}`);
      if (!response.ok) throw new Error(`Property request failed (${response.status})`);
      const payload = await response.json();
      if (String(payload.property_id || '') !== String(propertyId)) {
        throw new Error('The property response did not match the selected property.');
      }
      return payload;
    } catch (error) {
      if (cached) return cached;
      throw error;
    }
  }
  if (cached) return cached;
  throw new Error('The exact selected-area geometry is no longer available. Return to the map and open 3D from the current selection.');
}

function initializeRenderer() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#8db0be');
  scene.fog = new THREE.FogExp2('#8db0be', 0.0042);

  camera = new THREE.PerspectiveCamera(47, canvas.clientWidth / Math.max(canvas.clientHeight, 1), 0.1, 2500);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.065;
  controls.maxPolarAngle = Math.PI / 2 - 0.025;
  controls.minDistance = 5;
  controls.maxDistance = 400;

  scene.add(new THREE.HemisphereLight(0xd8f3ff, 0x203d2f, 1.25));
  const sun = new THREE.DirectionalLight(0xfff2d5, 2.2);
  sun.position.set(70, 110, 55);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 350;
  sun.shadow.camera.left = -100;
  sun.shadow.camera.right = 100;
  sun.shadow.camera.top = 100;
  sun.shadow.camera.bottom = -100;
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x9bc9ff, 0.55);
  fill.position.set(-55, 35, -60);
  scene.add(fill);

  modelRoot = new THREE.Group();
  modelRoot.name = 'selected-property-model';
  scene.add(modelRoot);

  window.addEventListener('resize', resizeRenderer);
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    showError('The 3D renderer stopped', 'The browser lost its graphics context. Reload this page to rebuild the selected-property model.');
  });

  (function renderLoop() {
    window.requestAnimationFrame(renderLoop);
    controls.update();
    renderer.render(scene, camera);
  })();
}

function resizeRenderer() {
  const width = modelViewport.clientWidth;
  const height = modelViewport.clientHeight;
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function shapeForRing(ring, origin) {
  const shape = new THREE.Shape();
  ring.forEach(([lon, lat], index) => {
    const point = geoToLocal(lon, lat, origin);
    if (index === 0) shape.moveTo(point.x, point.north);
    else shape.lineTo(point.x, point.north);
  });
  shape.closePath();
  return shape;
}

function worldPointsForRing(ring, origin, y) {
  return ring.map(([lon, lat]) => {
    const point = geoToLocal(lon, lat, origin);
    return new THREE.Vector3(point.x, y, -point.north);
  });
}

function addRingLine(ring, origin, y, color, opacity = 1) {
  const points = worldPointsForRing(ring, origin, y);
  if (!points.length) return null;
  points.push(points[0].clone());
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const line = new THREE.Line(geometry, material);
  modelRoot.add(line);
  return line;
}

function addHorizontalShape(ring, origin, y, material) {
  const geometry = new THREE.ShapeGeometry(shapeForRing(ring, origin));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y;
  mesh.receiveShadow = true;
  modelRoot.add(mesh);
  return mesh;
}

function addParcel(spec) {
  spec.parcelRings.forEach(ring => {
    const geometry = new THREE.ExtrudeGeometry(shapeForRing(ring, spec.bounds), {
      depth: 0.32,
      bevelEnabled: true,
      bevelThickness: 0.05,
      bevelSize: 0.05,
      bevelSegments: 2
    });
    const material = new THREE.MeshStandardMaterial({ color: 0x315a47, roughness: 0.96, metalness: 0 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = -0.28;
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    modelRoot.add(mesh);
    addRingLine(ring, spec.bounds, 0.08, 0x2dd4bf, 1);
  });
}

function addBackyard(spec) {
  const material = new THREE.MeshStandardMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.38,
    roughness: 0.8,
    depthWrite: false,
    side: THREE.DoubleSide
  });
  spec.backyardRings.forEach(ring => {
    addHorizontalShape(ring, spec.bounds, 0.095, material);
    addRingLine(ring, spec.bounds, 0.11, 0xfbbf24, 0.9);
  });
}

function roofGeometry(ring, origin, eaveHeight, roofHeight) {
  const perimeter = worldPointsForRing(ring, origin, eaveHeight);
  const center = perimeter.reduce((sum, point) => sum.add(point), new THREE.Vector3())
    .multiplyScalar(1 / perimeter.length);
  center.y = eaveHeight + roofHeight;
  const positions = [];
  for (let index = 0; index < perimeter.length; index += 1) {
    const a = perimeter[index];
    const b = perimeter[(index + 1) % perimeter.length];
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, center.x, center.y, center.z);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function addBuildings(spec) {
  spec.buildings.forEach((building, index) => {
    const height = building.heightM;
    const geometry = new THREE.ExtrudeGeometry(shapeForRing(building.ring, spec.bounds), {
      depth: height,
      bevelEnabled: false
    });
    const material = new THREE.MeshStandardMaterial({ color: 0x8da3bd, roughness: 0.74, metalness: 0.02 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `selected-building-${index + 1}`;
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = 0.1;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    modelRoot.add(mesh);

    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0xdbeafe, transparent: true, opacity: 0.42 });
    const edgeMesh = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 28), edgeMaterial);
    edgeMesh.rotation.x = -Math.PI / 2;
    edgeMesh.position.y = 0.1;
    edgeMesh.visible = false;
    modelRoot.add(edgeMesh);
    structureLines.push(edgeMesh);

    const roofRise = Math.max(1.25, Math.min(3.2, Math.sqrt(ringAreaSquareMetres(building.ring)) * 0.1));
    const roof = new THREE.Mesh(
      roofGeometry(building.ring, spec.bounds, height + 0.1, roofRise),
      new THREE.MeshStandardMaterial({ color: 0x465970, roughness: 0.88, side: THREE.DoubleSide, flatShading: true })
    );
    roof.name = `procedural-roof-${index + 1}`;
    roof.castShadow = true;
    roof.receiveShadow = true;
    modelRoot.add(roof);
    addRingLine(building.ring, spec.bounds, height + 0.11, 0x1e293b, 0.85);
  });
}

function addPools(spec) {
  const material = new THREE.MeshPhysicalMaterial({
    color: 0x0ea5e9,
    roughness: 0.08,
    metalness: 0.05,
    transmission: 0.16,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide
  });
  spec.poolRings.forEach(ring => {
    addHorizontalShape(ring, spec.bounds, 0.125, material);
    addRingLine(ring, spec.bounds, 0.14, 0x7dd3fc, 1);
  });
}

function addTrees(spec) {
  spec.treePoints.forEach(([lon, lat], index) => {
    const local = geoToLocal(lon, lat, spec.bounds);
    const group = new THREE.Group();
    group.name = `mapped-tree-${index + 1}`;
    group.position.set(local.x, 0.08, -local.north);
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.15, 0.24, 3.4, 8),
      new THREE.MeshStandardMaterial({ color: 0x65452f, roughness: 1 })
    );
    trunk.position.y = 1.7;
    trunk.castShadow = true;
    const crown = new THREE.Mesh(
      new THREE.DodecahedronGeometry(1.55, 1),
      new THREE.MeshStandardMaterial({ color: 0x18733d, roughness: 1, flatShading: true })
    );
    crown.position.y = 4.1;
    crown.castShadow = true;
    group.add(trunk, crown);
    modelRoot.add(group);
  });
}

function addContextSurface(spec) {
  const widthM = Math.max(20, Math.abs(geoToLocal(spec.bounds.east, spec.bounds.centerLat, spec.bounds).x) * 2);
  const depthM = Math.max(20, Math.abs(geoToLocal(spec.bounds.centerLon, spec.bounds.north, spec.bounds).north) * 2);
  const surfaceSpan = Math.max(widthM, depthM) * 8;
  const landscape = new THREE.Mesh(
    new THREE.PlaneGeometry(surfaceSpan, surfaceSpan),
    new THREE.MeshStandardMaterial({ color: 0x193b31, roughness: 1 })
  );
  landscape.rotation.x = -Math.PI / 2;
  landscape.position.y = -0.34;
  landscape.receiveShadow = true;
  scene.add(landscape);
  const grid = new THREE.GridHelper(surfaceSpan, 28, 0x2f6c60, 0x2b5148);
  grid.position.y = -0.3;
  grid.material.transparent = true;
  grid.material.opacity = 0.24;
  scene.add(grid);
}

function fitModel() {
  const bounds = new THREE.Box3().setFromObject(modelRoot);
  if (bounds.isEmpty()) throw new Error('The selected property produced no renderable geometry.');
  const size = bounds.getSize(new THREE.Vector3());
  bounds.getCenter(modelCenter);
  modelCenter.y = Math.max(1, Math.min(modelCenter.y, size.y * 0.42));
  modelSpan = Math.max(size.x, size.z, size.y * 1.6, 18);
  controls.minDistance = Math.max(4, modelSpan * 0.18);
  controls.maxDistance = Math.max(120, modelSpan * 7);
  setCameraPreset('perspective');
}

function cameraDistance() {
  return Math.max(26, modelSpan * 1.75);
}

function setCameraPreset(preset) {
  const distance = cameraDistance();
  controls.target.copy(modelCenter);
  if (preset === 'top') {
    camera.position.set(modelCenter.x, modelCenter.y + distance, modelCenter.z + 0.01);
  } else if (preset === 'low') {
    camera.position.set(modelCenter.x + distance * 0.82, modelCenter.y + distance * 0.34, modelCenter.z + distance * 0.88);
  } else {
    camera.position.set(modelCenter.x + distance * 0.72, modelCenter.y + distance * 0.66, modelCenter.z + distance * 0.9);
  }
  camera.near = Math.max(0.1, distance / 1000);
  camera.far = Math.max(1000, distance * 20);
  camera.updateProjectionMatrix();
  controls.update();
  document.querySelectorAll('[data-camera]').forEach(button => {
    button.classList.toggle('active', button.dataset.camera === (preset === 'reset' ? 'perspective' : preset));
  });
}

function wireControls() {
  document.querySelectorAll('[data-camera]').forEach(button => {
    button.addEventListener('click', () => setCameraPreset(button.dataset.camera === 'reset' ? 'perspective' : button.dataset.camera));
  });
  const toggle = document.getElementById('wireframe-toggle');
  toggle.addEventListener('click', () => {
    const enabled = toggle.getAttribute('aria-pressed') !== 'true';
    toggle.setAttribute('aria-pressed', String(enabled));
    structureLines.forEach(line => { line.visible = enabled; });
  });
}

function updateInterface(spec) {
  const isProperty = Boolean(spec.propertyId);
  const source = spec.provenance.geometrySource;
  const title = isProperty ? spec.address : 'Selected area model';
  document.title = `3D Property Model — ${title}`;
  setText('page-title', title);
  setText('property-title', title);
  setText('property-subtitle', spec.locationLabel || (isProperty ? 'Selected property' : 'Exact analysis selection'));
  setText('property-id', spec.propertyId || selection || 'Current selected area');
  setText('geometry-source', sourceLabel(source));
  setText('scope-badge', isProperty ? 'Selected property only' : 'Exact selected area only');
  setText('metric-parcel', formatMetric(spec.metrics.parcelAreaSqm, 'm²'));
  setText('metric-building', formatMetric(spec.metrics.buildingAreaSqm, 'm²'));
  setText('metric-height', formatMetric(spec.metrics.buildingHeightM, 'm', 1));
  setText('metric-backyard', formatMetric(spec.metrics.backyardAreaSqm, 'm²'));
  setText('property-explanation', spec.explanation || 'No suitability narrative is available for this selection.');

  const indicative = source !== 'real';
  const lead = indicative ? 'Illustrative footprint-based reconstruction.' : 'Source-footprint reconstruction.';
  setText(
    'model-provenance',
    `${lead} ${spec.provenance.note} Building height: ${spec.provenance.heightSource}; roof: ${spec.provenance.roofSource}. This is massing for screening, not a survey or photogrammetric digital twin.`
  );
  const provenanceElement = document.getElementById('model-provenance');
  provenanceElement.dataset.accuracy = indicative ? 'illustrative' : 'source-footprint';
  document.querySelector('[data-legend="pool"]').hidden = spec.poolRings.length === 0;
  document.querySelector('[data-legend="tree"]').hidden = spec.treePoints.length === 0;
}

async function buildModel() {
  try {
    const payload = await loadSelectedPayload();
    const eligibility = propertyModelEligibility(payload);
    const spec = eligibility.spec;
    if (eligibility.reason === 'area_not_property') {
      throw new Error('This is a drawn analysis area, not a selected property. Return to the map, run the suburb Scan, click a property parcel/lead, then choose 3D Property Model.');
    }
    if (eligibility.reason === 'missing_parcel') {
      throw new Error('The selected record has no parcel polygon. No placeholder property was generated.');
    }
    if (eligibility.reason === 'missing_building') {
      throw new Error('This property has no mapped building footprint, so a defensible 3D structure cannot be generated. No placeholder house was substituted.');
    }

    initializeRenderer();
    addContextSurface(spec);
    addParcel(spec);
    addBackyard(spec);
    addBuildings(spec);
    addPools(spec);
    addTrees(spec);
    fitModel();
    wireControls();
    updateInterface(spec);
    resizeRenderer();

    const manifest = {
      ready: true,
      propertyId: spec.propertyId,
      selection,
      parcelCount: spec.parcelRings.length,
      parcelVertexCount: spec.parcelRings.reduce((sum, ring) => sum + ring.length, 0),
      buildingCount: spec.buildings.length,
      buildingHeightsM: spec.buildings.map(item => item.heightM),
      backyardCount: spec.backyardRings.length,
      poolCount: spec.poolRings.length,
      treeCount: spec.treePoints.length,
      geometrySource: spec.provenance.geometrySource,
      scope: spec.provenance.scope,
      usedPlaceholder: false
    };
    window.__property3DManifest = manifest;
    document.documentElement.dataset.modelStatus = 'ready';
    document.documentElement.dataset.modelPropertyId = spec.propertyId || selection || 'selected-area';
    document.documentElement.dataset.modelBuildingCount = String(spec.buildings.length);
    document.documentElement.dataset.modelUsedPlaceholder = 'false';
    loader.classList.add('hidden');
  } catch (error) {
    console.error('[3D MODEL] Failed to build selected-property model:', error);
    showError('Could not build this selected-property model', error?.message || 'The geometry could not be rendered.');
  }
}

buildModel();
