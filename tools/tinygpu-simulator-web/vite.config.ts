import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 网页仿真器完全位于本教程 tools 目录，不依赖 tiny-gpu-compiler 工作区。
// 开发时把 /api 请求转给本机 Triton 编译服务，避免浏览器跨域。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
