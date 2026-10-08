"use client";
import type { TransformResult } from "@/lib/api";
import { downloadWolfram } from "@/lib/wolfram";
import WolframButton from "./WolframButton";
export default function ExpressionPanel({result}:{result:TransformResult}){async function copy(){await navigator.clipboard.writeText(result.expression)}return <div className="panel section"><div className="label">Wolfram Language construction</div><pre className="code">{result.expression}</pre>{result.wolfram_uses_compact_expression&&<p className="notice">The external link uses a compact visual expression for URL safety. Your .wl download contains the complete construction.</p>}<div className="actions"><button className="button secondary" onClick={copy}>Copy expression</button><button className="button secondary" onClick={()=>downloadWolfram(result.expression,result.download_filename)}>Download .wl file</button><WolframButton url={result.wolfram_url}/></div></div>}

