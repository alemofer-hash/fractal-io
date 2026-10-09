from pydantic import BaseModel

class RuleSummary(BaseModel):
    domain: str = "[0,1]^2"
    contraction_limit: float = 0.85
    subdivision_maps: int = 4

class ExpressionPayload(BaseModel):
    latex: str
    plain_text: str
    wolfram_code: str
    wolfram_url: str
    download_filename: str = "fractal-expression.wl"
    wolfram_uses_compact_expression: bool = False
    wolfram_url_expression: str | None = None

class TransformMetadata(BaseModel):
    width: int
    height: int
    planes: int
    contours: int
    triangles: int
    recursion_depth: int
    render_time_ms: int

class TransformResponse(BaseModel):
    success: bool
    image_url: str
    expression: ExpressionPayload
    metadata: TransformMetadata
    original: dict
    processing_time_ms: int
    plane_count: int
    contour_count: int
    triangle_count: int
    base_triangle_count: int
    leaf_triangle_count: int
    adaptive_refinement: bool
    mean_local_error: float
    maximum_local_error: float
    high_detail_regions: int
    fallback_used: bool
    source_resolution: dict
    analysis_resolution: dict
    mesh_resolution: dict
    render_resolution: dict
    preview_resolution: dict
    fractal_depth: int
    preview_data_url: str
    # Legacy aliases retained for existing local clients.
    wolfram_expression: str
    wolfram_url: str
    wolfram_uses_compact_expression: bool
    download_filename: str = "fractal_x2_result.wl"
    rule_summary: RuleSummary = RuleSummary()
    planes: list[dict]
    quality_reduced: bool = False

