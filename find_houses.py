import urllib.request
import json
import time

suburbs = [
    'Carindale', 'Kenmore', 'Camp Hill', 'Chapel Hill', 
    'Indooroopilly', 'Bulimba', 'The Gap', 'Bridgeman Downs',
    'North Lakes', 'Murrumba Downs', 'Samford Valley', 'Redcliffe'
]

results = {}
for sub in suburbs:
    query = f"""
    [out:json];
    area[name="{sub}"][place="suburb"]->.searchArea;
    way(area.searchArea)["building"="house"];
    out center limit 1;
    """
    url = 'https://overpass-api.de/api/interpreter'
    data = query.encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read())
            if res_data['elements']:
                center = res_data['elements'][0]['center']
                print(f'"{sub}": ({center["lon"]}, {center["lat"]}),')
            else:
                print(f'"{sub}": Not found')
    except Exception as e:
        print(f'Error for {sub}: {e}')
    time.sleep(1)
