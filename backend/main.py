from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from api.tasks import router as tasks_router
from api.auth import router as auth_router
from utils.logger import setup_logger
from config import API_HOST, API_PORT, LLM_CONFIG, VIDEO_CONFIG, MEMORY_CONFIG

# 初始化日志
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("应用启动，开始初始化...")
    logger.info("=" * 60)

    try:
        import torch
        from models.database import init_db

        # GPU 诊断
        logger.info("=" * 50)
        logger.info("GPU 环境检测:")
        logger.info(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU[{i}]: {props.name}")
                logger.info(f"    显存: {props.total_memory / 1024**3:.1f} GB")
            logger.info(f"  PyTorch CUDA: {torch.version.cuda}")
        else:
            logger.warning("  CUDA 不可用，如果 config.py 中 device='cuda' 则模型加载将失败")
        logger.info(f"  LLM 配置 device: {LLM_CONFIG['device']}")
        logger.info(f"  Video 配置 device: {VIDEO_CONFIG['device']}")
        logger.info("=" * 50)

        # 初始化数据库表
        logger.info("初始化数据库...")
        init_db()
        logger.info("数据库初始化完成")

        from services.model_loader import llm_loader, video_loader

        # 加载 LLM 模型
        logger.info("开始加载 LLM 模型...")
        llm_success = llm_loader.load_model()
        if llm_success:
            logger.info("✅ LLM 模型加载成功")
        else:
            logger.warning("⚠️ LLM 模型未加载，脚本生成使用备用方案")

        # 加载视频模型
        logger.info("开始加载视频生成模型...")
        video_success = video_loader.load_model()
        if video_success:
            logger.info("✅ 视频生成模型加载成功")
        else:
            logger.warning("⚠️ 视频模型未加载，视频生成使用备用方案")

    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
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
    version="2.0.0",
    lifespan=lifespan,
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

# 确定前端目录
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# 挂载静态文件
if os.path.exists(os.path.join(FRONTEND_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

# 挂载视频文件目录
videos_path = os.path.join(os.path.dirname(__file__), "videos")
if os.path.exists(videos_path):
    app.mount("/videos", StaticFiles(directory=videos_path), name="videos")


@app.get("/health")
def health_check():
    """健康检查接口"""
    from services.model_loader import llm_loader

    return {
        "status": "ok",
        "message": "服务运行正常",
        "llm_loaded": llm_loader.is_loaded,
        "device": llm_loader.device,
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


@app.get("/{full_path:path}")
async def serve_spa(full_path: str = ""):
    """Serve Vue SPA - all non-API routes return index.html"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not built. Run: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动服务: {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
