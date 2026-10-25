/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "mock" fakes the /admin/* API in the browser; "real" uses the server. Unset: mock in `npm run dev`, real in a build. */
  readonly VITE_ADMIN_API?: "mock" | "real";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
