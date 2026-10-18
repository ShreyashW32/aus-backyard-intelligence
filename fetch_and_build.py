"""One-shot fetcher: grabs real houses from OSM Overpass for greater Brisbane area."""
import json, math, random, time, urllib.request, urllib.parse
from pathlib import Path

OUTPUT = Path(__file__).parent / "aus_backyard_intelligence" / "data" / "derived" / "sample_properties.json"

# Single broad query - houses in Brisbane bounding box
QUERY = """
[out:json][timeout:90];
(
  way["building"="house"](-27.65,152.85,-27.20,153.20);
  way["building"="residential"](-27.65,152.85,-27.20,153.20);
  way["building"="detached"](-27.65,152.85,-27.20,153.20);
);
out geom 200;
"""

POOL_QUERY = """
[out:json][timeout:90];
(
  way["leisure"="swimming_pool"](-27.65,152.85,-27.20,153.20);
  relation["leisure"="swimming_pool"](-27.65,152.85,-27.20,153.20);
);
out geom;
"""

# Suburb lookup by lat/lon
SUBURB_ZONES = [
    ((-27.26,-27.23, 153.00,153.05), "North Lakes","Moreton Bay","4509"),
    ((-27.29,-27.26, 152.99,153.03), "Murrumba Downs","Moreton Bay","4503"),
    ((-27.39,-27.35, 152.86,152.92), "Samford Valley","Moreton Bay","4520"),
    ((-27.25,-27.21, 153.08,153.13), "Redcliffe","Moreton Bay","4020"),
    ((-27.28,-27.26, 152.97,153.00), "Petrie","Moreton Bay","4502"),
    ((-27.32,-27.28, 152.94,152.98), "Warner","Moreton Bay","4500"),
]

def find_suburb(lat, lon):
    for (s_lat, n_lat, w_lon, e_lon), sub, lga, pc in SUBURB_ZONES:
        if s_lat <= lat <= n_lat and w_lon <= lon <= e_lon:
            return sub, lga, pc
    return None, None, None

def estimate_elev(lat, lon):
    base = max(5.0, 80.0 - (lon - 152.9) * 350.0)
    base += abs(lat + 27.47) * 20.0
    return round(base + random.uniform(-3, 3), 1)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

print("Fetching real houses from OpenStreetMap (greater Brisbane area)...")
url = "https://overpass-api.de/api/interpreter"
data = urllib.parse.urlencode({"data": QUERY}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"User-Agent": "QLD-Backyard-Recommender/1.0"})

with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read().decode("utf-8"))

elements = result.get("elements", [])
print(f"Received {len(elements)} buildings from OSM")

print("Fetching existing pools from OpenStreetMap...")
data_pools = urllib.parse.urlencode({"data": POOL_QUERY}).encode("utf-8")
req_pools = urllib.request.Request(url, data=data_pools, headers={"User-Agent": "QLD-Backyard-Recommender/1.0"})
try:
    with urllib.request.urlopen(req_pools, timeout=120) as resp:
        result_pools = json.loads(resp.read().decode("utf-8"))
    elements_pools = result_pools.get("elements", [])
except Exception as e:
    print(f"Failed to fetch pools: {e}")
    elements_pools = []

pools = []
for el in elements_pools:
    if "geometry" in el:
        lats_p = [pt["lat"] for pt in el["geometry"]]
        lons_p = [pt["lon"] for pt in el["geometry"]]
        if lats_p:
            pools.append(((min(lats_p) + max(lats_p)) / 2, (min(lons_p) + max(lons_p)) / 2))
    elif "lat" in el and "lon" in el:
        pools.append((el["lat"], el["lon"]))
print(f"Loaded {len(pools)} existing pools from OSM")

props = []
for el in elements:
    if "geometry" not in el:
        continue
    nodes = el["geometry"]
    lats = [p["lat"] for p in nodes]
    lons = [p["lon"] for p in nodes]
    if len(lats) < 3:
        continue
    
    clat = (min(lats)+max(lats))/2
    clon = (min(lons)+max(lons))/2
    
    cos_lat = math.cos(math.radians(clat))
    bw = (max(lons)-min(lons)) * 111320 * cos_lat
    bd = (max(lats)-min(lats)) * 110540
    if bw < 4 or bd < 4:
        continue
    
    suburb, lga, postcode = find_suburb(clat, clon)
    if suburb is None:
        # Assign to nearest known suburb anyway
        suburb = "Moreton Bay Suburb"
        lga = "Moreton Bay"
        postcode = "4500"
    
    fs = random.uniform(5,8)
    rs = random.uniform(8,18)
    sl = random.uniform(1,4)
    sr = random.uniform(1,4)
    pw = bw+sl+sr
    pd = bd+fs+rs
    
    base_e = estimate_elev(clat, clon)
    tags = el.get("tags",{})
    hn = tags.get("addr:housenumber","")
    st = tags.get("addr:street","")
    addr = f"{hn} {st}, {suburb} QLD {postcode}".strip() if hn or st else f"Property OSM-{el['id']}, {suburb} QLD {postcode}"
    if addr.startswith(" "):
        addr = addr.strip()
        
    has_existing_pool = False
    for p_lat, p_lon in pools:
        if haversine_distance(clat, clon, p_lat, p_lon) <= 25.0:
            has_existing_pool = True
            break
    
    props.append({
        "id": f"OSM-{el['id']}",
        "lga": lga, "suburb": suburb,
        "address": addr,
        "lot_plan": f"OSM{el['id']}", "valuation_ref": f"VAL-{el['id']}",
        "parcel_rect": [0,0, round(pw,1), round(pd,1)],
        "building_rect": [round(sl,1), round(fs,1), round(sl+bw,1), round(fs+bd,1)],
        "elevation_corners_m": {
            "sw": base_e,
            "se": round(base_e + random.uniform(0,1.5),1),
            "nw": round(base_e + random.uniform(0.3,3),1),
            "ne": round(base_e + random.uniform(0.5,4),1)
        },
        "vegetation_pct": round(random.uniform(5,45),1),
        "road_frontage_m": round(pw,1),
        "corner_lot": random.random()<0.18,
        "flood_risk": 1 if base_e<12 and random.random()<0.4 else 0,
        "heritage_overlay": random.random()<0.06,
        "overhead_powerline": random.random()<0.12,
        "solar_access_pct": round(random.uniform(60,95),1),
        "view_quality": random.randint(3,9),
        "real_lat": round(clat,6), "real_lon": round(clon,6),
        "property_type": "residential",
        "ownership_class": "private",
        "is_private_property": True,
        "has_existing_pool": has_existing_pool,
        "exclude_from_pool_candidates": has_existing_pool,
    })

print(f"Built {len(props)} valid property records")

# Count by suburb
from collections import Counter
sc = Counter(p["suburb"] for p in props)
for sub, cnt in sc.most_common():
    print(f"  {sub}: {cnt}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(props, indent=2), encoding="utf-8")
print(f"\nSaved to {OUTPUT}")
