import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Fractal.io — Picture to mathematics", description: "Finite fractal construction from an image." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }

