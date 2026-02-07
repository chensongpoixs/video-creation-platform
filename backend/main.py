from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from api.tasks import router as tasks_router
from api.auth import router as auth_router
from utils.logger import setup_logger
from config import API_HOST, API_PORT, LLM_CONFIG

# 初始化日志
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("应用启动，开始初始化...")
    logger.info("=" * 60)
    
    try:
        from services.model_loader import llm_loader, video_loader
        
        # 加载 LLM 模型
        logger.info("开始加载 LLM 模型...")
        llm_success = llm_loader.load_model()
        if llm_success:
            logger.info("✅ LLM 模型加载成功")
        else:
            logger.warning("⚠️ LLM 模型加载失败，将使用备用方案")
        
        # 加载视频模型
        logger.info("开始加载视频生成模型...")
        video_success = video_loader.load_model()
        if video_success:
            logger.info("✅ 视频生成模型加载成功")
        else:
            logger.warning("⚠️ 视频生成模型加载失败，将使用备用方案")
            
    except Exception as e:
        logger.error(f"❌ 模型初始化失败: {str(e)}")
        logger.info("系统将继续运行，但部分功能将不可用")
    
    logger.info("=" * 60)
    logger.info("应用启动完成")
    logger.info("=" * 60)
    
    yield
    
    # 关闭时卸载模型
    logger.info("应用关闭，卸载模型...")
    try:
        from services.model_loader import llm_loader, video_loader
        llm_loader.unload_model()
        video_loader.unload_model()
    except:
        pass

# 创建FastAPI应用
app = FastAPI(
    title="多模态视频创作平台",
    description="基于本地私有化部署的多模态视频生成系统",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)  # 认证路由
app.include_router(tasks_router)  # 任务路由

# 挂载静态文件（前端）
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# 挂载视频文件目录
videos_path = os.path.join(os.path.dirname(__file__), "videos")
if os.path.exists(videos_path):
    app.mount("/videos", StaticFiles(directory=videos_path), name="videos")

@app.get("/")
def read_root():
    """返回前端页面"""
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/health")
def health_check():
    """健康检查接口"""
    from services.model_loader import llm_loader
    
    return {
        "status": "ok",
        "message": "服务运行正常",
        "llm_loaded": llm_loader.is_loaded,
        "device": llm_loader.device
    }

@app.get("/api/model/status")
def model_status():
    """获取模型状态"""
    from services.model_loader import llm_loader, video_loader
    import torch
    
    status = {
        "llm_loaded": llm_loader.is_loaded,
        "video_loaded": video_loader.is_loaded,
        "device": llm_loader.device,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        status["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    
    return status

if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动服务: {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
