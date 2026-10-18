from __future__ import annotations

import re
from typing import Any

PRIVATE_OWNERSHIP_KEYWORDS = {
    "private",
    "freehold",
    "owner occupied",
    "individual",
    "proprietor",
    "residential owner",
    "strata",
}

PUBLIC_OWNERSHIP_KEYWORDS = {
    "airport",
    "moreton bay regional council",
    "moreton bay city council",
    "moreton bay council",
    "council",
    "crown",
    "department",
    "government",
    "hospital",
    "housing authority",
    "public housing",
    "queensland rail",
    "school",
    "state of queensland",
    "university",
}

RESIDENTIAL_USE_KEYWORDS = {
    "detached house",
    "dwelling",
    "granny flat",
    "house",
    "house and land",
    "low density residential",
    "private residence",
    "residence",
    "residential",
    "rural residential",
    "single family",
    "suburban neighbourhood",
    "townhouse",
    "unit",
    "villa",
}

NON_RESIDENTIAL_USE_KEYWORDS = {
    "aged care",
    "airport",
    "car park",
    "cemetery",
    "church",
    "civic",
    "club",
    "commercial",
    "community",
    "council depot",
    "education",
    "government",
    "hospital",
    "industrial",
    "library",
    "mixed use commercial",
    "open space",
    "park",
    "playground",
    "police",
    "public",
    "rail",
    "recreation",
    "reserve",
    "retail",
    "school",
    "shopping",
    "sport",
    "stadium",
    "state facility",
    "stormwater",
    "transport",
    "utility",
    "warehouse",
    "water treatment",
}

PUBLIC_ADDRESS_KEYWORDS = {
    "college",
    "hospital",
    "library",
    "park",
    "reserve",
    "school",
    "state school",
    "university",
}

VILLA_KEYWORDS = {"villa", "duplex", "townhouse", "terrace"}


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _text_blob(values: list[Any]) -> str:
    parts = [str(value).strip().lower() for value in values if value not in (None, "")]
    return " ".join(parts)


def _contains_keyword(blob: str, keywords: set[str]) -> bool:
    return any(keyword in blob for keyword in keywords)


def assess_private_property(record: dict[str, Any]) -> dict[str, Any]:
    address = str(record.get("address") or "").strip()
    address_blob = address.lower()
    ownership_blob = _text_blob(
        [
            record.get("ownership_class"),
            record.get("owner_type"),
            record.get("owner_name"),
            record.get("tenure"),
            record.get("owner_category"),
        ]
    )
    use_blob = _text_blob(
        [
            record.get("property_type"),
            record.get("residential_type"),
            record.get("land_use"),
            record.get("site_use"),
            record.get("zoning"),
            record.get("planning_zone"),
        ]
    )
    explicit_private = _optional_bool(record.get("is_private_property"))
    if explicit_private is None:
        explicit_private = _optional_bool(record.get("private_property"))
    explicit_public = _optional_bool(record.get("is_public_property"))
    if explicit_public is None:
        explicit_public = _optional_bool(record.get("public_property"))

    if explicit_public is True and explicit_private is not True:
        return {
            "is_private_property_candidate": False,
            "is_public_property": True,
            "ownership_class": "public",
            "property_classification": "public_or_excluded_site",
            "residential_type": "non_residential",
            "candidate_label": "Excluded public or non-residential site",
            "eligibility_confidence": 1.0,
            "eligibility_reason": "Explicitly marked as a public property.",
        }

    if _contains_keyword(ownership_blob, PUBLIC_OWNERSHIP_KEYWORDS):
        return {
            "is_private_property_candidate": False,
            "is_public_property": True,
            "ownership_class": "public",
            "property_classification": "public_or_institutional_site",
            "residential_type": "non_residential",
            "candidate_label": "Excluded public or institutional site",
            "eligibility_confidence": 0.95,
            "eligibility_reason": "Ownership clues suggest the parcel is public, institutional, or otherwise not a private backyard site.",
        }

    if _contains_keyword(use_blob, NON_RESIDENTIAL_USE_KEYWORDS) or _contains_keyword(address_blob, PUBLIC_ADDRESS_KEYWORDS):
        return {
            "is_private_property_candidate": False,
            "is_public_property": True,
            "ownership_class": "public_or_non_residential",
            "property_classification": "public_or_non_residential_site",
            "residential_type": "non_residential",
            "candidate_label": "Excluded public or non-residential site",
            "eligibility_confidence": 0.9,
            "eligibility_reason": "Land-use or address clues point to a public, civic, commercial, or other non-residential place where a private pool/fence recommendation would be inappropriate.",
        }

    residential_hit = explicit_private is True or _contains_keyword(use_blob, RESIDENTIAL_USE_KEYWORDS)
    address_hit = bool(re.match(r"^\s*\d+[A-Za-z]?(?:/\d+[A-Za-z]?)?\s+", address))
    has_building = bool(record.get("building_wkt") or record.get("building_rect"))
    has_backyard_geometry = bool(
        record.get("backyard_wkt")
        or record.get("backyard_rect")
        or (record.get("parcel_rect") and record.get("building_rect"))
        or (record.get("parcel_wkt") and record.get("building_wkt"))
    )

    if residential_hit or (address_hit and has_building):
        residential_type = "villa" if _contains_keyword(f"{use_blob} {address_blob}", VILLA_KEYWORDS) else "house"
        confidence = 0.98 if explicit_private is True else 0.9 if residential_hit else 0.72
        reason = (
            "Explicit private or residential flags confirm this is a private dwelling site."
            if explicit_private is True
            else "Residential land-use clues point to a private dwelling suitable for backyard analysis."
            if residential_hit
            else "A normal street address plus a building footprint strongly suggests a private dwelling rather than public land."
        )
        if has_backyard_geometry:
            reason += " Building and rear-yard geometry are available for pool and fence scoring."
        return {
            "is_private_property_candidate": True,
            "is_public_property": False,
            "ownership_class": "private",
            "property_classification": f"private_residential_{residential_type}",
            "residential_type": residential_type,
            "candidate_label": f"Private residential {residential_type} pool/fence candidate",
            "eligibility_confidence": confidence,
            "eligibility_reason": reason,
        }

    return {
        "is_private_property_candidate": False,
        "is_public_property": False,
        "ownership_class": "unknown",
        "property_classification": "unverified_property_type",
        "residential_type": "unknown",
        "candidate_label": "Excluded unverified property type",
        "eligibility_confidence": 0.45,
        "eligibility_reason": "The record could not be verified confidently enough as a private residential property, so it was excluded from pool and fence recommendations.",
    }


def enrich_property_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    assessment = assess_private_property(enriched)
    enriched.setdefault("ownership_class", assessment["ownership_class"])
    enriched.setdefault("residential_type", assessment["residential_type"])
    enriched["is_private_property"] = assessment["is_private_property_candidate"]
    enriched["is_public_property"] = assessment["is_public_property"]
    enriched["property_classification"] = assessment["property_classification"]
    enriched["candidate_label"] = assessment["candidate_label"]
    enriched["eligibility_confidence"] = assessment["eligibility_confidence"]
    enriched["eligibility_reason"] = assessment["eligibility_reason"]
    enriched["pool_fence_candidate"] = assessment["is_private_property_candidate"]
    return enriched
