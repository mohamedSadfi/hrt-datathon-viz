import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `base` is the public URL prefix Vite bakes into asset paths.
//   - dev / preview / root deploys: '/'
//   - GitHub Pages project page:    '/<repo-name>/'
// CI sets VITE_BASE='/${{ github.event.repository.name }}/' in pages.yml so
// the same source tree deploys correctly regardless of repo name.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || '/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
  },
  server: {
    port: 5173,
    strictPort: false,
  },
});
