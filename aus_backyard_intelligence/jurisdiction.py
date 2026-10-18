"""Australian state/territory awareness: point -> jurisdiction, plus the
per-jurisdiction pool-compliance matrix.

Pool/spa barrier regulation is NOT national. The National Construction Code sets
barrier performance standards, but registration and inspection are state law:
QLD and NSW run public state registers, VIC requires council registration with
4-yearly barrier certification, the rest rely on council inspection/enforcement.
Every lead therefore carries its jurisdiction and the lawful check link for that
jurisdiction rather than a QLD-only assumption.

Boundaries are the bbox approximations already verified for the basemap router
(map.js AU_GOV_REGIONS), extended with WA/NT/ACT. Good to a few km along the
Murray and state corners — fine for compliance-link routing, not for cadastre.
"""

from __future__ import annotations

from typing import Any

# Checked in order; first bbox containing the point wins. ACT must precede NSW
# (it is inside NSW's bbox); VIC's Murray border is approximated with three
# boxes; QLD's Tweed coast border with two.
_STATE_BBOXES: list[tuple[str, float, float, float, float]] = [
    # (code, south, west, north, east)
    ("act", -35.95, 148.74, -35.10, 149.42),
    ("tas", -44.0, 143.6, -39.2, 148.9),
    ("vic", -39.25, 140.96, -34.0, 143.0),   # Mallee
    ("vic", -39.25, 143.0, -35.6, 146.0),    # mid-Murray
    ("vic", -39.25, 146.0, -35.9, 150.05),   # east
    ("qld", -28.18, 150.5, -8.9, 153.65),    # coastal (Tweed border)
    ("qld", -29.2, 137.999, -8.9, 150.5),    # inland
    ("sa", -38.2, 129.0, -26.0, 141.0),
    ("nsw", -37.6, 141.0, -28.15, 153.7),
    ("wa", -35.3, 112.5, -13.4, 129.0),
    ("nt", -26.0, 129.0, -10.4, 138.001),
]

STATE_NAMES = {
    "act": "Australian Capital Territory",
    "nsw": "New South Wales",
    "nt": "Northern Territory",
    "qld": "Queensland",
    "sa": "South Australia",
    "tas": "Tasmania",
    "vic": "Victoria",
    "wa": "Western Australia",
}

# The compliance matrix. register_kind:
#   state_register    — public, searchable state register (strongest lead signal:
#                       a detected pool absent from the register is a compliance lead)
#   council_register  — mandatory registration held by local council, not publicly
#                       searchable statewide
#   council_enforced  — barrier rules enforced by council inspection; no register
POOL_COMPLIANCE: dict[str, dict[str, str]] = {
    "qld": {
        "register_kind": "state_register",
        "register_label": "QBCC pool safety register",
        "register_url": "https://www.qbcc.qld.gov.au/home-building-owners/pool-safety/pool-safety-register",
        "source_title": "QBCC — Swimming pool safety laws",
        "source_url": "https://www.qbcc.qld.gov.au/home-owner-hub/swimming-pools/swimming-pool-safety-laws",
        "legislation": "Building Act 1975 (Qld), chapter 8; Building Regulation 2021 (Qld)",
        "reviewed_at": "2026-07-19",
        "note": "Every regulated QLD pool must appear in the public QBCC register; "
                "a pool safety certificate is required to sell or lease.",
    },
    "nsw": {
        "register_kind": "state_register",
        "register_label": "NSW Swimming Pool Register",
        "register_url": "https://www.swimmingpoolregister.nsw.gov.au/",
        "source_title": "NSW Swimming Pool Register — Further information",
        "source_url": "https://www.swimmingpoolregister.nsw.gov.au/information/further-information",
        "legislation": "Swimming Pools Act 1992 (NSW); Swimming Pools Regulation 2018 (NSW)",
        "reviewed_at": "2026-07-19",
        "note": "Registration is mandatory; a certificate of compliance is required "
                "to sell or lease.",
    },
    "vic": {
        "register_kind": "council_register",
        "register_label": "VIC pool/spa registration (local council)",
        "register_url": "https://www.bpc.vic.gov.au/home-owners/home-renovation-and-building-essentials/swimming-pools-and-spas",
        "source_title": "Victorian Building and Plumbing Commission — Swimming pools and spas",
        "source_url": "https://www.bpc.vic.gov.au/home-owners/home-renovation-and-building-essentials/swimming-pools-and-spas",
        "legislation": "Building Act 1993 (Vic); Building Regulations 2018 (Vic), part 9A",
        "reviewed_at": "2026-07-19",
        "note": "Pools/spas must be registered with the local council and the barrier "
                "certified every four years.",
    },
    "wa": {
        "register_kind": "council_enforced",
        "register_label": "WA pool barrier rules (Building and Energy)",
        "register_url": "https://www.wa.gov.au/organisation/building-and-energy/swimming-pool-and-spa-safety-barrier-requirements",
        "source_title": "WA Building and Energy — Swimming pool and spa safety barriers",
        "source_url": "https://www.wa.gov.au/organisation/building-and-energy/swimming-pool-and-spa-safety-barrier-requirements",
        "legislation": "Building Act 2011 (WA); Building Regulations 2012 (WA), part 8 division 2",
        "reviewed_at": "2026-07-19",
        "note": "No public register; local governments must inspect pool barriers at "
                "least every four years.",
    },
    "sa": {
        "register_kind": "council_enforced",
        "register_label": "SA swimming pool safety (PlanSA)",
        "register_url": "https://plan.sa.gov.au/our_planning_system/instruments/building_rules",
        "source_title": "PlanSA — Building rules",
        "source_url": "https://plan.sa.gov.au/our_planning_system/instruments/building_rules",
        "legislation": "Planning, Development and Infrastructure Act 2016 (SA); PDI (Swimming Pool Safety) Regulations 2019",
        "reviewed_at": "2026-07-19",
        "note": "Barrier rules under the Planning, Development and Infrastructure Act; "
                "councils inspect new pools. No public register.",
    },
    "tas": {
        "register_kind": "council_enforced",
        "register_label": "TAS pool barrier standards (CBOS)",
        "register_url": "https://www.cbos.tas.gov.au/topics/products-services/safety/pool-safety-barriers",
        "source_title": "Tasmania CBOS — Pool safety barriers",
        "source_url": "https://www.cbos.tas.gov.au/topics/products-services/safety/pool-safety-barriers",
        "legislation": "Building Act 2016 (Tas); National Construction Code swimming-pool access requirements",
        "reviewed_at": "2026-07-19",
        "note": "Barriers regulated under the Building Act 2016, enforced by councils. "
                "No public register.",
    },
    "act": {
        "register_kind": "council_enforced",
        "register_label": "ACT pool safety (Access Canberra)",
        "register_url": "https://www.planning.act.gov.au/planning-projects/building-regulatory-system/home-swimming-pool-safety-reforms",
        "source_title": "ACT Planning — Home swimming pool safety reforms",
        "source_url": "https://www.planning.act.gov.au/planning-projects/building-regulatory-system/home-swimming-pool-safety-reforms",
        "legislation": "Building Act 2004 (ACT), part 3A; Building (Swimming Pool Safety) Legislation Amendment Act 2023 (ACT)",
        "reviewed_at": "2026-07-19",
        "note": "The scheme is in a transition period: barrier maintenance and secure-access duties apply now; "
                "prescribed barrier standards and compliance evidence phase in by 1 May 2028.",
    },
    "nt": {
        "register_kind": "council_enforced",
        "register_label": "NT pool safety (Swimming Pool Safety Act)",
        "register_url": "https://nt.gov.au/property/swimming-pools",
        "source_title": "NT Government — Buying or selling property with pools or spas",
        "source_url": "https://nt.gov.au/property/buying-and-selling-a-home/settle-the-sale/buying-or-selling-property-with-pools-or-spas",
        "legislation": "Swimming Pool Safety Act 2004 (NT); Swimming Pool Safety Regulations 2004 (NT)",
        "reviewed_at": "2026-07-19",
        "note": "Pool safety compliance certificates under the Swimming Pool Safety Act; "
                "required on sale or lease in regulated areas.",
    },
}


def state_for_point(lat: float, lon: float) -> str:
    """Lower-case state/territory code for a coordinate, or '' outside Australia."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return ""
    for code, s, w, n, e in _STATE_BBOXES:
        if s <= lat <= n and w <= lon <= e:
            return code
    return ""


def pool_compliance(lat: float, lon: float) -> dict[str, Any] | None:
    """Jurisdiction + pool-compliance info for a coordinate, or None outside AU."""
    code = state_for_point(lat, lon)
    if not code:
        return None
    rules = POOL_COMPLIANCE[code]
    return {
        "state": code.upper(),
        "state_name": STATE_NAMES[code],
        **rules,
    }
