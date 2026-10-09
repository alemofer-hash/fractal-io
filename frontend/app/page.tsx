"use client";
import { useEffect, useState } from "react";
import ImageDropzone from "@/components/ImageDropzone";
import ProcessingStages from "@/components/ProcessingStages";
import ImageComparison from "@/components/ImageComparison";
import PlaneStack from "@/components/PlaneStack";
import ExpressionCard from "@/components/ExpressionCard";
import { transform, type TransformResult } from "@/lib/api";
type State="empty"|"selected"|"processing"|"success"|"error";
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [state, setState] = useState<State>("empty");
  const [error, setError] = useState("");
  const [result, setResult] = useState<TransformResult | null>(null);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  function choose(next: File) { if (preview) URL.revokeObjectURL(preview); setFile(next); setPreview(URL.createObjectURL(next)); setState("selected"); setResult(null); setError(""); }
  function remove() { if (preview) URL.revokeObjectURL(preview); setFile(null); setPreview(null); setState("empty"); setResult(null); }
  async function run() { if (!file) return; setState("processing"); setError(""); try { setResult(await transform(file)); setState("success"); } catch (caught) { setError(caught instanceof Error ? caught.message : "Transformation failed."); setState("error"); } }
  return <main className="shell">
    <div className="brand"><b>Fractal</b>.io / x²</div>
    <section className="hero"><h1>Turn a picture into mathematics.</h1><p>Fractal.io decomposes your image into closed contours, triangular planes and a finite fractal construction.</p></section>
    <section className="panel section"><ImageDropzone file={file} preview={preview} onFile={choose} onRemove={remove} disabled={state === "processing"}/>{file && <div className="actions" style={{justifyContent:"center",marginTop:18}}><button className="button" disabled={state === "processing"} onClick={run}>{state === "processing" ? "Constructing mathematics…" : "Transform me into mathematics"}</button></div>}{error && <p className="error">{error}</p>}</section>
    {state === "processing" && <ProcessingStages active/>}
    {result && preview && <>
      <ImageComparison original={preview} reconstructed={result.preview_data_url}/>
      <div className="metrics">{[[result.plane_count,"planes"],[result.contour_count,"contours"],[result.base_triangle_count,"base triangles"],[result.leaf_triangle_count,"adaptive triangles"],[result.fractal_depth,"fractal depth"],[`${result.processing_time_ms}ms`,"process time"]].map(([value,label]) => <div className="metric" key={String(label)}><strong>{value}</strong><span>{label}</span></div>)}</div>
      <p className="notice">Adaptive refinement: mean local error {result.mean_local_error.toFixed(5)} · high-detail regions {result.high_detail_regions}</p>
      <p className="notice">Resolution pipeline: source {result.source_resolution.width}×{result.source_resolution.height} · analysis {result.analysis_resolution.width}×{result.analysis_resolution.height} · mesh {result.mesh_resolution.width}×{result.mesh_resolution.height} · render {result.render_resolution.width}×{result.render_resolution.height}</p>
      {result.quality_reduced && <p className="notice">The engine reduced detail to remain inside the processing budget.</p>}
      <PlaneStack result={result}/><ExpressionCard result={result}/>
    </>}
  </main>;
}
