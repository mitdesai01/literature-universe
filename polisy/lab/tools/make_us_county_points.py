# Rebuild site_assets/us_county_xy.json: npm pack us-atlas@3.0.1 && tar xzf us-atlas-3.0.1.tgz
#   python make_us_county_points.py package/counties-albers-10m.json ../polisy_lab/site_assets/us_county_xy.json
"""County FIPS -> [x, y] on the site's Albers USA canvas (975 x 610): the area-weighted centroid of the county's
largest polygon, from us-atlas counties-albers-10m (US Census Bureau cartographic boundaries)."""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
t = json.load(open(src))
sx, sy = t["transform"]["scale"]
tx, ty = t["transform"]["translate"]


def decode(arc):
    x = y = 0
    pts = []
    for dx, dy in arc:
        x += dx
        y += dy
        pts.append((x * sx + tx, y * sy + ty))
    return pts


arcs = [decode(a) for a in t["arcs"]]


def ring(idx):
    pts = []
    for i in idx:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        pts += a if not pts else a[1:]
    return pts


def centroid(pts):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        f = x0 * y1 - x1 * y0
        a += f
        cx += (x0 + x1) * f
        cy += (y0 + y1) * f
    if abs(a) < 1e-9:
        return abs(a), sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return abs(a) / 2, cx / (3 * a), cy / (3 * a)


res = {}
for g in t["objects"]["counties"]["geometries"]:
    if "arcs" not in g:
        continue
    polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
    best = max((centroid(ring(p[0])) for p in polys), key=lambda c: c[0])
    res[g["id"]] = [round(best[1], 1), round(best[2], 1)]
json.dump({"source": "us-atlas 3.0.1 counties-albers-10m (US Census Bureau), largest-polygon centroids", "points": res},
          open(out, "w"), separators=(",", ":"))
print(len(res), "counties")
