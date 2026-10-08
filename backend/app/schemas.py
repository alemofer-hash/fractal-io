from pydantic import BaseModel

class RuleSummary(BaseModel):
    domain: str = "[0,1]^2"
    contraction_limit: float = 0.85
    subdivision_maps: int = 4

class TransformResponse(BaseModel):
    success: bool
    original: dict
    processing_time_ms: int
    plane_count: int
    contour_count: int
    triangle_count: int
    fractal_depth: int
    preview_data_url: str
    expression: str
    wolfram_expression: str
    wolfram_url: str
    wolfram_uses_compact_expression: bool
    download_filename: str = "fractal_x2_result.wl"
    rule_summary: RuleSummary = RuleSummary()
    planes: list[dict]
    quality_reduced: bool = False

