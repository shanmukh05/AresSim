/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEPLOYED?: string;
  readonly VITE_BASE_PATH?: string;
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
