import json
import re
from pathlib import Path

from aus_backyard_intelligence.jurisdiction import (
    _STATE_BBOXES,
    POOL_COMPLIANCE,
    STATE_NAMES,
    pool_compliance,
    state_for_point,
)

SAMPLE_DATA = (
    Path(__file__).resolve().parents[1]
    / "aus_backyard_intelligence"
    / "data"
    / "derived"
    / "sample_properties.json"
)


def test_state_for_point_capitals():
    assert state_for_point(-27.470, 153.026) == "qld"  # Brisbane
    assert state_for_point(-33.868, 151.209) == "nsw"  # Sydney
    assert state_for_point(-37.813, 144.963) == "vic"  # Melbourne
    assert state_for_point(-31.953, 115.857) == "wa"   # Perth
    assert state_for_point(-34.928, 138.600) == "sa"   # Adelaide
    assert state_for_point(-42.882, 147.327) == "tas"  # Hobart
    assert state_for_point(-12.463, 130.844) == "nt"   # Darwin


def test_act_wins_over_surrounding_nsw():
    assert state_for_point(-35.281, 149.128) == "act"  # Canberra
    # Just outside the ACT box is NSW again.
    assert state_for_point(-34.75, 149.72) == "nsw"    # Goulburn


def test_outside_australia_is_empty():
    # Warner, New Hampshire — the exact bad records that polluted the demo data.
    assert state_for_point(43.290922, -71.861355) == ""
    assert state_for_point(-36.85, 174.76) == ""  # Auckland
    assert state_for_point(None, None) == ""


def test_pool_compliance_matrix_covers_all_jurisdictions():
    assert set(POOL_COMPLIANCE) == set(STATE_NAMES)
    for code, rules in POOL_COMPLIANCE.items():
        assert rules["register_url"].startswith("https://"), code
        assert rules["source_url"].startswith("https://"), code
        assert rules["source_title"], code
        assert rules["legislation"], code
        assert rules["reviewed_at"] == "2026-07-19", code
        assert rules["register_kind"] in {"state_register", "council_register", "council_enforced"}
    # The two public state registers are the strongest compliance-lead signal.
    assert POOL_COMPLIANCE["qld"]["register_kind"] == "state_register"
    assert POOL_COMPLIANCE["nsw"]["register_kind"] == "state_register"


def test_pool_compliance_lookup():
    info = pool_compliance(-27.470, 153.026)
    assert info["state"] == "QLD"
    assert "qbcc" in info["register_url"]
    assert info["reviewed_at"] == "2026-07-19"
    assert "Building Act 1975" in info["legislation"]
    assert pool_compliance(51.5, -0.1) is None  # London


def test_client_jurisdiction_mirror_in_sync():
    """static/js/services/jurisdiction.js mirrors jurisdiction.py; the two are
    maintained by hand, so this guards against silent drift in the bbox table
    (order matters) and in which jurisdictions carry a register link."""
    js_path = (
        Path(__file__).resolve().parents[1]
        / "aus_backyard_intelligence"
        / "static"
        / "js"
        / "services"
        / "jurisdiction.js"
    )
    js = js_path.read_text(encoding="utf-8")

    js_rows = re.findall(
        r"\['(\w+)',\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]", js
    )
    js_bboxes = [(code, float(s), float(w), float(n), float(e)) for code, s, w, n, e in js_rows]
    assert js_bboxes == list(_STATE_BBOXES)

    for code, rules in POOL_COMPLIANCE.items():
        assert f"{code}:" in js, f"{code} missing from client POOL_REGISTERS"
        assert rules["register_url"] in js, f"{code} register URL differs in client mirror"
        assert rules["source_url"] in js, f"{code} source URL differs in client mirror"
        assert rules["reviewed_at"] in js, f"{code} review date differs in client mirror"

    assert "pool-compliance-provenance" in js
    assert "reviewed ${reviewed}" in js


def test_sample_data_is_clean():
    """The demo dataset must never carry personal contact details (fabricated
    owner identities are an APP 7 hazard) and every record must be in Australia."""
    records = json.loads(SAMPLE_DATA.read_text(encoding="utf-8"))
    assert records, "sample dataset must not be empty"
    for record in records:
        assert record.get("owner_name", "") == ""
        assert record.get("owner_phone", "") == ""
        assert record.get("owner_email", "") == ""
        assert record.get("state"), f"record {record.get('id')} has no state"
        assert state_for_point(record["real_lat"], record["real_lon"]) == record["state"].lower()
