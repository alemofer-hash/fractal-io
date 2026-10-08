from __future__ import annotations
from urllib.parse import quote

def _polygon(item: dict, height: int) -> str:
    color = item["colors"].mean(axis=0)
    pts = ",".join("{%0.4f,%0.4f}" % (p[0], 1 - p[1] / max(1, height - 1)) for p in item["vertices"])
    return "{RGBColor(%0.3f,%0.3f,%0.3f),Polygon[{%s}]}" % (*color, pts)

def emit_wolfram_expression(planes: list[dict], height: int) -> str:
    polygons = [_polygon(item, height) for plane in planes for item in plane["triangles"]]
    return "Graphics[{" + ",".join(polygons) + "},PlotRange->{{0,1},{0,1}},ImageSize->600]"

def compact_wolfram_expression(planes: list[dict], height: int, max_length: int = 7000) -> tuple[str, bool]:
    polygons = [_polygon(item, height) for plane in planes for item in plane["triangles"]]
    full = "Graphics[{" + ",".join(polygons) + "},PlotRange->{{0,1},{0,1}},ImageSize->600]"
    if len(full) <= max_length: return full, False
    stride = max(1, len(polygons) // 80)
    compact = "Graphics[{" + ",".join(polygons[::stride]) + "},PlotRange->{{0,1},{0,1}},ImageSize->600]"
    return compact, True

def build_wolfram_url(expression: str) -> str:
    return "https://www.wolframalpha.com/input?i=" + quote(expression, safe="")

