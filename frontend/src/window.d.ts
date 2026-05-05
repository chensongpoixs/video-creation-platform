// Window 类型扩展（必须是模块文件，所以有 export {}）
declare global {
  interface Window {
    __APP_CONFIG__?: AppConfig
  }
}

export {}
