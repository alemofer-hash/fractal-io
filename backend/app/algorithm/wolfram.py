from __future__ import annotations
from urllib.parse import quote

def _expression_items(planes: list[dict], width: int, height: int) -> tuple[list[str], list[str]]:
    """Build a normalized GraphicsComplex; coordinates are never raw image pixels."""
    points: list[str] = []; point_index: dict[tuple[float, float], int] = {}; primitives: list[str] = []
    for plane in planes:
        for item in plane["triangles"]:
            indices = []
            for point in item["vertices"]:
                normalized = (round(float(point[0]) / max(1, width - 1), 6), round(1 - float(point[1]) / max(1, height - 1), 6))
                if normalized not in point_index:
                    point_index[normalized] = len(points) + 1; points.append("{%0.6f,%0.6f}" % normalized)
                indices.append(point_index[normalized])
            color = item.get("field_color", item["colors"].mean(axis=0))
            primitives.append("{Directive[RGBColor(%0.4f,%0.4f,%0.4f],Opacity(%0.4f)],Polygon[{%s}]}" % (*color, item.get("opacity", 1.0), ",".join(map(str, indices))))
    return points, primitives

def emit_wolfram_expression(planes: list[dict], width: int, height: int) -> str:
    points, primitives = _expression_items(planes, width, height)
    return "Graphics[GraphicsComplex[{" + ",".join(points) + "},{" + ",".join(primitives) + "}],PlotRange->{{0,1},{0,1}},ImageSize->600]"

def compact_wolfram_expression(planes: list[dict], width: int, height: int, max_length: int = 7000) -> tuple[str, bool]:
    full = emit_wolfram_expression(planes, width, height)
    if len(full) <= max_length: return full, False
    total = sum(len(plane["triangles"]) for plane in planes)

    def evenly_spaced(items: list[dict], count: int) -> list[dict]:
        if len(items) <= count: return items
        return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]

    # A GraphicsComplex triangle usually takes 120–180 source characters once
    # its colour directive and its distinct points are included.  Iterate rather
    # than assuming a stride: the URL expression must actually meet the bound.
    target = min(total, 36)
    compact = full
    while target >= 3:
        compact_planes = []
        for plane in planes:
            triangles = plane["triangles"]
            if not triangles: continue
            allocation = max(1, round(target * len(triangles) / max(total, 1)))
            compact_planes.append({**plane, "triangles": evenly_spaced(triangles, allocation)})
        compact = emit_wolfram_expression(compact_planes, width, height)
        if len(compact) <= max_length: return compact, True
        target -= 2
    return compact, True

def build_wolfram_url(expression: str) -> str:
    return "https://www.wolframalpha.com/input?i=" + quote(expression, safe="")

