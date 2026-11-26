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

/** The `?…` part of the hash, such as `worker=W-0001` in `#/checks?worker=W-0001`. */
export function currentQuery(): URLSearchParams {
  const hash = window.location.hash;
  const at = hash.indexOf("?");
  return new URLSearchParams(at === -1 ? "" : hash.slice(at + 1));
}

export function navigate(path: string): void {
  window.location.hash = path;
}

/** Like `navigate`, but in place of the current history entry, so Back doesn't return to the address being left. */
export function replacePath(path: string): void {
  window.location.replace(`#${path}`);
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
