/**
 * 前端应用配置文件
 * ================
 * 部署到生产环境时，修改此文件中的后端地址即可，无需重新构建。
 *
 * 开发环境（npm run dev）：
 *   - Vite 自动代理 /api、/videos、/health 到 apiBaseURL
 *   - apiBaseURL 保持默认 '/' 即可
 *
 * 生产环境（npm run build 后部署）：
 *   - 将 apiBaseURL 修改为后端实际地址
 *   - 例如: 'http://192.168.1.100:8010' 或 'https://api.example.com'
 */
window.__APP_CONFIG__ = {
  /**
   * 后端 API 基础地址
   * 开发时保持 '/'（Vite proxy 自动转发）
   * 生产部署时改为后端实际地址
   */
  apiBaseURL: '/',

  /**
   * 视频文件基础地址
   * - 默认与 apiBaseURL 相同（视频由后端同一服务提供）
   * - 如果视频托管在 CDN 或其他服务器，可单独配置
   * - 开发环境保持 '/' 即可（Vite proxy 自动转发 /videos）
   */
  videoBaseURL: '/',

  /**
   * 请求超时时间（毫秒）
   */
  timeout: 30000,

  /**
   * 后端服务端口（供运维参考，代码不直接使用）
   */
  backendPort: 8010,
}
