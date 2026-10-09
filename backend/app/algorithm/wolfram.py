from __future__ import annotations
from urllib.parse import quote

DOWNLOAD_FILENAME = "fractal-expression.wl"

def _normalized_point(point, width: int, height: int) -> tuple[float, float]:
    x = min(1.0, max(0.0, float(point[0]) / max(1, width - 1)))
    y = min(1.0, max(0.0, 1.0 - float(point[1]) / max(1, height - 1)))
    return round(x, 6), round(y, 6)

def _expression_items(planes: list[dict], width: int, height: int) -> tuple[list[str], list[str]]:
    """Build a normalized GraphicsComplex; coordinates are never raw image pixels."""
    points: list[str] = []; point_index: dict[tuple[float, float], int] = {}; primitives: list[str] = []
    for plane in planes:
        for item in plane["triangles"]:
            indices = []
            for point in item["vertices"]:
                normalized = _normalized_point(point, width, height)
                if normalized not in point_index:
                    point_index[normalized] = len(points) + 1; points.append("{%0.6f,%0.6f}" % normalized)
                indices.append(point_index[normalized])
            color = item.get("field_color", item["colors"].mean(axis=0))
            primitives.append("{Directive[RGBColor[%0.4f,%0.4f,%0.4f],Opacity[%0.4f]],Polygon[{%s}]}" % (*color, item.get("opacity", 1.0), ",".join(map(str, indices))))
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

def readable_latex() -> str:
    return (r"\begin{aligned}"
            r"\mathcal{I}(x,y) &= \sum_{p=1}^{P}\sum_{t=1}^{T_p} "
            r"\mathbf{1}_{\Delta_{p,t}}(x,y)"
            r"\left[\sum_{k=1}^{3}\lambda_{kpt}(x,y)c_{kpt}\right]\\"
            r"\Delta^{(d+1)}_{p,t} &= \bigcup_{j=1}^{4}S_{p,t,j}(\Delta^{(d)}_{p,t})"
            r"\end{aligned}")

def plain_text_expression() -> str:
    return ("I(x,y) = sum_p sum_t 1_[Delta(p,t)](x,y) "
            "[sum_k lambda(k,p,t)(x,y) c(k,p,t)]\n"
            "Delta^(d+1)(p,t) = union_(j=1..4) S(p,t,j)(Delta^d(p,t))")

def build_expression_model(planes: list[dict], width: int, height: int) -> dict:
    """Serialize one structured triangular field into all user-facing forms."""
    complete = emit_wolfram_expression(planes, width, height)
    compact, compacted = compact_wolfram_expression(planes, width, height)
    external = compact if compacted else complete
    return {
        "latex": readable_latex(),
        "plain_text": plain_text_expression(),
        "wolfram_code": complete,
        "wolfram_url": build_wolfram_url(external),
        "download_filename": DOWNLOAD_FILENAME,
        "wolfram_uses_compact_expression": compacted,
        "wolfram_url_expression": external,
    }

