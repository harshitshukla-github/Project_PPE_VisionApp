/**
 * A tiny hash router: `#/workers` is the Workers page. The built dashboard is
 * a static app served by FastAPI at /admin, and a hash needs no server-side
 * fallback for deep links, so there is nothing to configure.
 */
import { useEffect, useState } from "react";

export function currentPath(): string {
  const hash = window.location.hash.replace(/^#/, "");
  return hash.split("?")[0] || "/";
}

export function navigate(path: string): void {
  window.location.hash = path;
}

/** The current page's path, kept up to date when the hash changes. */
export function useHashPath(): string {
  const [path, setPath] = useState(currentPath);
  useEffect(() => {
    const update = () => setPath(currentPath());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return path;
}
