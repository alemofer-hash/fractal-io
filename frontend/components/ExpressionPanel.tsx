"use client";
import type { TransformResult } from "@/lib/api";
import { downloadWolfram } from "@/lib/wolfram";
import WolframButton from "./WolframButton";
export default function ExpressionPanel({result}:{result:TransformResult}){const visualExpression=result.wolfram_uses_compact_expression?result.wolfram_expression:result.expression;async function copy(){await navigator.clipboard.writeText(visualExpression)}return <div className="panel section"><div className="label">Wolfram Language construction</div><pre className="code">{visualExpression}</pre>{result.wolfram_uses_compact_expression&&<p className="notice">This is the compact visual expression used by Wolfram|Alpha. Your .wl download keeps the complete construction.</p>}<div className="actions"><button className="button secondary" onClick={copy}>Copy expression</button><button className="button secondary" onClick={()=>downloadWolfram(result.expression,result.download_filename)}>Download .wl file</button><WolframButton url={result.wolfram_url}/></div></div>}

