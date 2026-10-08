export function downloadWolfram(expression:string, filename:string) { const url=URL.createObjectURL(new Blob([expression],{type:"text/plain;charset=utf-8"})); const link=document.createElement("a"); link.href=url;link.download=filename;link.click();URL.revokeObjectURL(url); }

