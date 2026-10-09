"use client";

import { useMemo, useState } from "react";
import katex from "katex";
import type { TransformResult } from "@/lib/api";
import { downloadWolfram } from "@/lib/wolfram";

type Tab = "formula" | "wolfram" | "technical";

export default function ExpressionCard({ result }: { result: TransformResult }) {
  const [tab, setTab] = useState<Tab>("formula");
  const [copied, setCopied] = useState(false);
  const formula = useMemo(() => katex.renderToString(result.expression.latex, {
    displayMode: true,
    throwOnError: false,
    output: "html",
  }), [result.expression.latex]);

  async function copyExpression() {
    await navigator.clipboard.writeText(result.expression.wolfram_code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  const chip = (label: string, value: string | number) => <span className="formula-chip" key={label}><b>{value}</b> {label}</span>;

  return <section className="panel expression-card section" aria-label="Fractal expression">
    <div className="expression-heading">
      <div><div className="label">Mathematical construction</div><h2>Fractal Image Function</h2></div>
      <div className="expression-mark">𝓘(x,y)</div>
    </div>
    <nav className="expression-tabs" aria-label="Expression views">
      <button className={tab === "formula" ? "expression-tab active" : "expression-tab"} onClick={() => setTab("formula")}>Readable Formula</button>
      <button className={tab === "wolfram" ? "expression-tab active" : "expression-tab"} onClick={() => setTab("wolfram")}>Wolfram Language</button>
      <button className={tab === "technical" ? "expression-tab active" : "expression-tab"} onClick={() => setTab("technical")}>Technical Details</button>
    </nav>

    {tab === "formula" && <div className="formula-view">
      <div className="formula-render" dangerouslySetInnerHTML={{ __html: formula }} />
      <p className="formula-description">Piecewise fractal image field composed of layered triangular domains with barycentric color interpolation.</p>
      <div className="formula-chips">
        {chip("planes", result.metadata.planes)}{chip("contours", result.metadata.contours)}{chip("triangles", result.metadata.triangles)}{chip("recursion depth", result.metadata.recursion_depth)}{chip("render time", `${result.metadata.render_time_ms} ms`)}
      </div>
      <p className="formula-recursion">Recursive rule: <span>Δ<sup>(d+1)</sup><sub>p,t</sub> = ⋃<sub>j=1..4</sub> S<sub>p,t,j</sub>(Δ<sup>(d)</sup><sub>p,t</sub>)</span></p>
    </div>}

    {tab === "wolfram" && <div className="wolfram-view">
      <p className="formula-description">Exact Wolfram Language representation of the final triangular field.</p>
      <pre className="code expression-code"><code>{result.expression.wolfram_code}</code></pre>
      {result.expression.wolfram_uses_compact_expression && <p className="notice">The Wolfram|Alpha link uses a compact expression for browser URL safety. The download contains the complete construction.</p>}
      <div className="actions">
        <button className="button secondary" onClick={copyExpression}>{copied ? "Copied" : "Copy expression"}</button>
        <button className="button secondary" onClick={() => downloadWolfram(result.expression.wolfram_code, result.expression.download_filename)}>Download .wl</button>
        {result.expression.wolfram_url && <a className="button" href={result.expression.wolfram_url} target="_blank" rel="noopener noreferrer">Open in Wolfram|Alpha ↗</a>}
      </div>
    </div>}

    {tab === "technical" && <div className="technical-view">
      <div className="technical-grid">
        <div><span>Coordinate system</span><b>Ω = [0,1]²</b></div>
        <div><span>Color interpolation</span><b>Barycentric RGB field</b></div>
        <div><span>Planes / contours</span><b>{result.metadata.planes} / {result.metadata.contours}</b></div>
        <div><span>Triangles</span><b>{result.metadata.triangles} adaptive leaves</b></div>
        <div><span>Recursion depth</span><b>{result.metadata.recursion_depth}</b></div>
        <div><span>Generation time</span><b>{result.metadata.render_time_ms} ms</b></div>
      </div>
      <p className="notice">Each leaf stores its vertices, RGB controls, opacity, depth and contraction factor. The preview and Wolfram export use the same normalized triangle geometry.</p>
    </div>}
  </section>;
}
