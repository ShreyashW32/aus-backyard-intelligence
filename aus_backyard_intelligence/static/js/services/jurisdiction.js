// Client mirror of aus_backyard_intelligence/jurisdiction.py — point → state
// code + per-jurisdiction pool-compliance link. Pool registration is state law,
// not national: QLD/NSW have public registers, VIC registers via councils,
// the rest are council-enforced barrier rules. Keep the two files in sync.

// Checked in order; first bbox containing the point wins (ACT before NSW,
// VIC Murray border approximated with three boxes, QLD Tweed coast with two).
const STATE_BBOXES = [
  ['act', -35.95, 148.74, -35.10, 149.42],
  ['tas', -44.0, 143.6, -39.2, 148.9],
  ['vic', -39.25, 140.96, -34.0, 143.0],
  ['vic', -39.25, 143.0, -35.6, 146.0],
  ['vic', -39.25, 146.0, -35.9, 150.05],
  ['qld', -28.18, 150.5, -8.9, 153.65],
  ['qld', -29.2, 137.999, -8.9, 150.5],
  ['sa', -38.2, 129.0, -26.0, 141.0],
  ['nsw', -37.6, 141.0, -28.15, 153.7],
  ['wa', -35.3, 112.5, -13.4, 129.0],
  ['nt', -26.0, 129.0, -10.4, 138.001]
];

export const POOL_REGISTERS = {
  qld: {
    label: 'QLD pool register check (QBCC)',
    url: 'https://www.qbcc.qld.gov.au/home-building-owners/pool-safety/pool-safety-register',
    title: 'Every regulated QLD pool must appear in the public QBCC register — check this address',
    sourceTitle: 'QBCC — Swimming pool safety laws',
    sourceUrl: 'https://www.qbcc.qld.gov.au/home-owner-hub/swimming-pools/swimming-pool-safety-laws',
    legislation: 'Building Act 1975 (Qld), chapter 8; Building Regulation 2021 (Qld)',
    reviewedAt: '2026-07-19'
  },
  nsw: {
    label: 'NSW pool register check',
    url: 'https://www.swimmingpoolregister.nsw.gov.au/',
    title: 'NSW pools must be on the state Swimming Pool Register — check this address',
    sourceTitle: 'NSW Swimming Pool Register — Further information',
    sourceUrl: 'https://www.swimmingpoolregister.nsw.gov.au/information/further-information',
    legislation: 'Swimming Pools Act 1992 (NSW); Swimming Pools Regulation 2018 (NSW)',
    reviewedAt: '2026-07-19'
  },
  vic: {
    label: 'VIC pool/spa rules (council registration)',
    url: 'https://www.bpc.vic.gov.au/home-owners/home-renovation-and-building-essentials/swimming-pools-and-spas',
    title: 'VIC pools must be registered with the local council; barrier certified every 4 years',
    sourceTitle: 'Victorian Building and Plumbing Commission — Swimming pools and spas',
    sourceUrl: 'https://www.bpc.vic.gov.au/home-owners/home-renovation-and-building-essentials/swimming-pools-and-spas',
    legislation: 'Building Act 1993 (Vic); Building Regulations 2018 (Vic), part 9A',
    reviewedAt: '2026-07-19'
  },
  wa: {
    label: 'WA pool barrier rules',
    url: 'https://www.wa.gov.au/organisation/building-and-energy/swimming-pool-and-spa-safety-barrier-requirements',
    title: 'WA: council barrier inspections at least every 4 years; no public register',
    sourceTitle: 'WA Building and Energy — Swimming pool and spa safety barriers',
    sourceUrl: 'https://www.wa.gov.au/organisation/building-and-energy/swimming-pool-and-spa-safety-barrier-requirements',
    legislation: 'Building Act 2011 (WA); Building Regulations 2012 (WA), part 8 division 2',
    reviewedAt: '2026-07-19'
  },
  sa: {
    label: 'SA pool safety rules (PlanSA)',
    url: 'https://plan.sa.gov.au/our_planning_system/instruments/building_rules',
    title: 'SA: pool-safety construction and inspection rules under the PDI framework',
    sourceTitle: 'PlanSA — Building rules',
    sourceUrl: 'https://plan.sa.gov.au/our_planning_system/instruments/building_rules',
    legislation: 'Planning, Development and Infrastructure Act 2016 (SA); PDI (Swimming Pool Safety) Regulations 2019',
    reviewedAt: '2026-07-19'
  },
  tas: {
    label: 'TAS pool barrier rules (CBOS)',
    url: 'https://www.cbos.tas.gov.au/topics/products-services/safety/pool-safety-barriers',
    title: 'TAS: pool barriers under the Building Act 2016 and applicable building standards',
    sourceTitle: 'Tasmania CBOS — Pool safety barriers',
    sourceUrl: 'https://www.cbos.tas.gov.au/topics/products-services/safety/pool-safety-barriers',
    legislation: 'Building Act 2016 (Tas); National Construction Code swimming-pool access requirements',
    reviewedAt: '2026-07-19'
  },
  act: {
    label: 'ACT home pool safety reforms',
    url: 'https://www.planning.act.gov.au/planning-projects/building-regulatory-system/home-swimming-pool-safety-reforms',
    title: 'ACT: transition to prescribed barrier standards runs to 1 May 2028; maintenance duties apply now',
    sourceTitle: 'ACT Planning — Home swimming pool safety reforms',
    sourceUrl: 'https://www.planning.act.gov.au/planning-projects/building-regulatory-system/home-swimming-pool-safety-reforms',
    legislation: 'Building Act 2004 (ACT), part 3A; Building (Swimming Pool Safety) Legislation Amendment Act 2023 (ACT)',
    reviewedAt: '2026-07-19'
  },
  nt: {
    label: 'NT pool safety rules',
    url: 'https://nt.gov.au/property/swimming-pools',
    title: 'NT: compliance certificates under the Swimming Pool Safety Act',
    sourceTitle: 'NT Government — Buying or selling property with pools or spas',
    sourceUrl: 'https://nt.gov.au/property/buying-and-selling-a-home/settle-the-sale/buying-or-selling-property-with-pools-or-spas',
    legislation: 'Swimming Pool Safety Act 2004 (NT); Swimming Pool Safety Regulations 2004 (NT)',
    reviewedAt: '2026-07-19'
  }
};

// Lower-case state code for a coordinate, or '' outside Australia.
export function stateForPoint(lat, lon) {
  for (const [code, s, w, n, e] of STATE_BBOXES) {
    if (lat >= s && lat <= n && lon >= w && lon <= e) return code;
  }
  return '';
}

// HTML fragment for a popup: the jurisdiction-correct compliance link.
export function poolRegisterLinkHtml(lat, lon) {
  const info = POOL_REGISTERS[stateForPoint(lat, lon)];
  if (!info) return '';
  const reviewed = new Date(`${info.reviewedAt}T00:00:00Z`).toLocaleDateString('en-AU', {
    day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'
  });
  return `<br><a href="${info.url}" target="_blank" rel="noopener" title="${info.title}">${info.label}</a>` +
    `<span class="pool-compliance-provenance">Official guidance: ` +
    `<a href="${info.sourceUrl}" target="_blank" rel="noopener">${info.sourceTitle}</a>` +
    ` · reviewed ${reviewed}<br>${info.legislation}</span>`;
}
