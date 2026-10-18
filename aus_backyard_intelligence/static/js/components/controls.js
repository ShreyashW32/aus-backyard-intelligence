// Floating map controls on the true MapLibre WebGL camera: 3D pitch/bearing
// ease, compass reset to north-flat, zoom capsule, and the Street View pegman.
import { state } from '../state.js';
import { getActiveCoords } from '../utils.js';
import { renderAILayers } from './map.js';

function setPegmanButtonActive(active) {
  const pegmanBtn = document.getElementById('ctrl-pegman');
  pegmanBtn.style.background = active ? '#eab308' : '';
  pegmanBtn.style.borderColor = active ? '#fbbf24' : '';
}

function googleMapFrames() {
  return Array.from(document.querySelectorAll('.google-map-frame'));
}

function activeGoogleFrame() {
  return document.getElementById(state.activeGoogleFrameId) || document.getElementById('map-iframe');
}

function toggleEmbeddedGoogleStreetView() {
  const iframe = activeGoogleFrame();
  const frames = googleMapFrames();
  const mapEl = document.getElementById('map');
  const select = document.getElementById('map-basemap');
  const isStreetView = iframe.dataset.streetView === 'true';

  if (isStreetView) {
    delete iframe.dataset.streetView;
    iframe.style.display = 'none';
    iframe.style.pointerEvents = 'none';
    iframe.src = '';
    mapEl.style.visibility = '';
    setPegmanButtonActive(false);
    select.dispatchEvent(new Event('change'));
    return;
  }

  const center = state.map.getCenter();
  window.clearTimeout(state.googleFrameSwapTimer);
  state.googleFrameLoadSequence += 1;
  frames.forEach(frame => {
    frame.onload = null;
    if (frame !== iframe) {
      frame.style.display = 'none';
      frame.src = '';
      delete frame.dataset.mapUrl;
    }
  });

  iframe.dataset.streetView = 'true';
  delete iframe.dataset.mapUrl;

  // Make the iframe interactive and visible for Street View
  iframe.style.display = 'block';
  iframe.style.opacity = '1';
  iframe.style.zIndex = '2';
  iframe.style.pointerEvents = 'auto';

  mapEl.style.visibility = 'hidden';
  iframe.src = `https://maps.google.com/maps?cbll=${center.lat},${center.lng}&layer=c&cbp=11,0,0,0,0&output=svembed`;
  setPegmanButtonActive(true);
  import('./ui.js').then(m => m.showToast('Street View opened at the map centre. Click Pegman again to return.', 'info'));
}

export function togglePegmanMode(enable) {
  if (enable === undefined) {
    state.isPegmanActive = !state.isPegmanActive;
  } else {
    state.isPegmanActive = enable;
  }

  const mapEl = document.getElementById('map');
  const pegmanBtn = document.getElementById('ctrl-pegman');

  if (state.isPegmanActive) {
    mapEl.classList.add('pegman-active');
    pegmanBtn.style.background = '#eab308';
    pegmanBtn.style.borderColor = '#fbbf24';
  } else {
    mapEl.classList.remove('pegman-active');
    pegmanBtn.style.background = '';
    pegmanBtn.style.borderColor = '';
  }
}

// True 3D camera: eases the WebGL camera to an oblique pitch/bearing and lights
// up the building fill-extrusions. No CSS transforms anywhere.
export function toggleMapTilt(enable) {
  const next = enable === undefined ? !state.isTilted : Boolean(enable);
  if (next === state.isTilted) {
    if (!next) return;  // already flat, nothing to reset
  }
  state.isTilted = next;

  const tiltBtn = document.getElementById('ctrl-tilt-mode');
  const ctrl3d = document.getElementById('ctrl-3d');

  if (state.isTilted) {
    state.map.easeTo({
      pitch: 45,
      bearing: 15,
      duration: 800,
      easing: (t) => t * (2 - t)
    });
    if (tiltBtn) {
      tiltBtn.style.background = '#6366f1';
      tiltBtn.style.borderColor = '#818cf8';
    }
  } else {
    state.map.easeTo({ pitch: 0, bearing: 0, duration: 800, easing: (t) => t * (2 - t) });
    if (tiltBtn) {
      tiltBtn.style.background = '';
      tiltBtn.style.borderColor = '';
    }
  }
  if (ctrl3d) {
    ctrl3d.textContent = state.isTilted ? '2D' : '3D';
    ctrl3d.title = state.isTilted ? 'Return to flat 2D view' : 'Tilt into 3D with extruded buildings';
  }
  // Extrusion visibility depends on the tilt state (render signature includes it).
  state.lastAIRenderKey = null;
  renderAILayers();
}

export function initControls() {
  console.log('[CONTROLS] initControls() called - wiring all buttons');
  document.getElementById('ctrl-pegman').addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleEmbeddedGoogleStreetView();
  });

  // Pegman click-drop on map
  state.map.on('click', (e) => {
    if (state.isPegmanActive) {
      import('./ui.js').then(m => m.loadStreetView(e.lngLat.lat, e.lngLat.lng));
      togglePegmanMode(false);
    }
  });

  document.getElementById('ctrl-tilt-mode').addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleMapTilt();
  });

  document.getElementById('ctrl-3d').addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleMapTilt();
  });

  // Compass: smoothly reset pitch and bearing to north-flat 2D.
  document.getElementById('ctrl-compass').addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleMapTilt(false);
    togglePegmanMode(false);

    googleMapFrames().forEach(frame => delete frame.dataset.streetView);
    setPegmanButtonActive(false);
    const ctrl3d = document.getElementById('ctrl-3d');
    if (ctrl3d) ctrl3d.textContent = '3D';
    const coords = getActiveCoords();
    state.map.easeTo({
      center: [coords.lng, coords.lat],
      pitch: 0,
      bearing: 0,
      duration: 800,
      easing: (t) => t * (2 - t)
    });
    const select = document.getElementById('map-basemap');
    select.dispatchEvent(new Event('change'));
  });

  document.getElementById('ctrl-zoom-in').addEventListener('click', (ev) => {
    ev.stopPropagation();
    state.map.zoomIn({ duration: 250 });
  });

  document.getElementById('ctrl-zoom-out').addEventListener('click', (ev) => {
    ev.stopPropagation();
    state.map.zoomOut({ duration: 250 });
  });

  // Live compass needle: rotate the icon with the camera bearing so the user
  // always sees which way north is while rotating/tilting in 3D.
  const compassSvg = document.querySelector('#ctrl-compass svg');
  if (compassSvg) {
    state.map.on('rotate', () => {
      compassSvg.style.transform = `rotate(${-state.map.getBearing()}deg)`;
    });
  }
}
