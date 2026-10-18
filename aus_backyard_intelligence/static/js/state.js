// Global state coordinator for the modular map dashboard (MapLibre GL build).
export const state = {
  map: null,                 // maplibregl.Map
  mapReady: false,           // style + sources loaded
  polygon: null,             // finished analysis selection: { vertices: [{lat, lng}, ...] }
  vertices: [],              // in-progress drawing vertices [{lat, lng}, ...]
  currentData: null,

  // Custom Control Modes
  drawingMode: null,
  isPegmanActive: false,
  isTilted: false,           // true 3D camera pitch active

  // Public feature and satellite detection cache
  cachedPoolsAndTrees: null,
  cachedBuildings: null,
  isFetchingAIData: false,
  poolFetchTimeout: null,
  hasExistingPool: false,
  manualPools: [],
  manualPoolSequence: 0,

  // Bounding Box Spatial Cache
  fetchedBBoxes: [],
  globalPools: new Map(),
  globalTrees: new Map(),
  globalPowerlines: new Map(),
  globalBuildings: new Map(),
  globalSolarPanels: new Map(),  // real detected arrays (satellite-yolo-solar)
  scanOwnedFeatureSnapshots: new Map(), // scan id -> cache keys plus prior non-scan features for snapshot reconciliation

  // Rendered-feature lookup: GeoJSON uid → cached feature object (popups/feedback).
  renderedPoolIndex: new Map(),
  renderedPanelIndex: new Map(),

  // Performance: Nearmap-inspired optimizations
  cacheDirty: false,               // Only rebuild cache arrays when new data arrives
  lastSuccessfulEndpoint: null,    // Sticky Overpass routing — try fastest server first
  lastFetchZoom: null,             // Adaptive debounce — faster at high zoom
  aiFetchController: null,         // Cancels stale viewport requests while panning
  aiFetchSequence: 0,              // Ignores late responses from superseded requests
  aiFeatureRevision: 0,            // Invalidates render signatures when data changes
  aiRenderFrame: null,             // Coalesces checkbox/map render bursts
  lastAIRenderKey: null,
  detectorSignature: null,
  segmentationMode: 'auto',
  lastAgentScanId: null,
  agentScanPollTimer: null,
  agentScanActiveId: null,
  agentScanScopeKey: null,
  agentScanPollGeneration: 0,    // invalidates in-flight poll responses when selection scope changes
  agentScanStartInFlight: false,
  agentScanCancellationInFlight: false,
  scanCameraFollow: true,          // false for user-drawn scans (don't hijack their view)
  scanAutoFocused: false,
  terrainGrid: null,
  terrainGridKey: null,
  lastTerrainGridKey: null,
  contourWorker: null,             // background Web Worker building contour GeoJSON
  contourWorkerBroken: false,      // fall back to sync contours if module workers fail
  activeGoogleFrameId: 'map-iframe',
  googleFrameLoadSequence: 0,
  googleFrameSwapTimer: null,

  // Tactical OSINT fusion layer
  osint: {
    flights: new Map(),            // icao24 → latest state vector
    flightTrails: new Map(),       // icao24 → [{lat, lon, t}, ...]
    cameras: [],
    wildfires: null,               // GeoJSON FeatureCollection passthrough
    incidents: [],
    earthquakes: [],
    flightTimer: null,
    slowTimer: null,
    pulsePhase: 0,
    pulseFrame: null,
    lastCameraBBoxKey: null
  }
};
