# Rebuild site_assets/us_states.json: npm pack us-atlas@3.0.1 && tar xzf us-atlas-3.0.1.tgz
#   python make_us_states.py package/states-albers-10m.json ../polisy_lab/site_assets/us_states.json 0.4
"""Turn us-atlas states-albers-10m (pre-projected Albers USA, 975x610) into compact SVG paths."""
import json, math, sys
src, out, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
t = json.load(open(src))
sx, sy = t["transform"]["scale"]; tx, ty = t["transform"]["translate"]

def decode(arc):
    x = y = 0; pts = []
    for dx, dy in arc:
        x += dx; y += dy
        pts.append((x * sx + tx, y * sy + ty))
    return pts

def dp(pts, tol):
    if len(pts) < 3: return pts
    keep = [False] * len(pts); keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1; L = math.hypot(dx, dy)
        best, bi = -1, -1
        for i in range(a + 1, b):
            px, py = pts[i]
            d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / L if L else math.hypot(px - x1, py - y1)
            if d > best: best, bi = d, i
        if best > tol:
            keep[bi] = True; stack += [(a, bi), (bi, b)]
    return [p for p, k in zip(pts, keep) if k]

arcs = [dp(decode(a), tol) for a in t["arcs"]]

def ring(idx):
    pts = []
    for i in idx:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        pts += a if not pts else a[1:]
    return pts

def path(geom):
    polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
    parts = []
    for poly in polys:
        for r in poly:
            pts = ring(r)
            if len(pts) < 4: continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            if max(xs) - min(xs) < 0.6 and max(ys) - min(ys) < 0.6: continue  # specks
            parts.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
    return "".join(parts)

ABBR = {"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","60":"AS","66":"GU","69":"MP","72":"PR","78":"VI"}
states = []
for g in t["objects"]["states"]["geometries"]:
    if g["id"] not in ABBR or int(g["id"]) > 56: continue
    d = path(g)
    states.append({"fips": g["id"], "abbr": ABBR[g["id"]], "name": g["properties"]["name"], "d": d})
res = {"source": "us-atlas 3.0.1 states-albers-10m (US Census Bureau cartographic boundaries; d3.geoAlbersUsa scale 1300, translate [487.5, 305])",
       "width": 975, "height": 610, "states": sorted(states, key=lambda s: s["fips"])}
json.dump(res, open(out, "w"), separators=(",", ":"))
print(len(states), "states;", sum(len(s["d"]) for s in states), "chars")
