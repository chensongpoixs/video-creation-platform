/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// 全局应用配置类型（public/config.js）
interface AppConfig {
  apiBaseURL: string
  videoBaseURL: string
  timeout: number
  backendPort: number
}

declare module 'element-plus/dist/locale/zh-cn.mjs' {
  const locale: any
  export default locale
}
