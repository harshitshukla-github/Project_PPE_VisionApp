/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "mock" fakes the /admin/* API in the browser with sample data. Anything else, or unset: the real server. */
  readonly VITE_ADMIN_API?: "mock" | "real";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
