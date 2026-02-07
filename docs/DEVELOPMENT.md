# 开发指南

## 开发环境配置

### 1. 安装依赖
```bash
cd backend
pip install -r requirements.txt
pip install pytest pytest-cov  # 测试工具
```

### 2. 配置IDE
推荐使用VSCode，安装以下插件：
- Python
- Pylance
- Python Test Explorer

### 3. 代码规范
项目遵循PEP 8规范，使用以下工具：
```bash
pip install black flake8 mypy
```

## 项目结构

```
video-creation-platform/
├── backend/                 # 后端代码
│   ├── api/                # API路由
│   │   ├── __init__.py
│   │   └── tasks.py        # 任务管理接口
│   ├── models/             # 数据模型
│   │   ├── __init__.py
│   │   └── database.py     # 数据库模型
│   ├── services/           # 业务逻辑
│   │   ├── __init__.py
│   │   ├── llm_service.py  # LLM服务
│   │   ├── video_service.py # 视频生成服务
│   │   ├── task_processor.py # 任务处理器
│   │   └── model_loader.py  # 模型加载器
│   ├── utils/              # 工具函数
│   │   ├── __init__.py
│   │   └── logger.py       # 日志工具
│   ├── config.py           # 配置文件
│   ├── main.py             # 应用入口
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端代码
│   ├── index.html          # 主页面
│   ├── app.js              # 前端逻辑
│   └── style.css           # 样式文件
├── tests/                  # 测试代码
│   ├── __init__.py
│   └── test_api.py         # API测试
├── docs/                   # 文档
│   ├── API.md              # API文档
│   ├── ARCHITECTURE.md     # 架构文档
│   ├── DEPLOYMENT.md       # 部署文档
│   └── DEVELOPMENT.md      # 开发文档
├── scripts/                # 脚本
│   ├── setup.sh            # 环境配置
│   └── run.sh              # 启动脚本
└── README.md               # 项目说明
```

## 开发流程

### 1. 创建功能分支
```bash
git checkout -b feature/your-feature-name
```

### 2. 编写代码
遵循以下原则：
- 单一职责原则
- 代码注释清晰
- 函数命名语义化
- 避免硬编码

### 3. 编写测试
```python
# tests/test_your_feature.py
def test_your_function():
    result = your_function()
    assert result == expected_value
```

### 4. 运行测试
```bash
pytest tests/
pytest --cov=backend tests/  # 带覆盖率
```

### 5. 代码格式化
```bash
black backend/
flake8 backend/
mypy backend/
```

### 6. 提交代码
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

## 常见开发任务

### 添加新的API接口
1. 在`backend/api/`创建或修改路由文件
2. 定义Pydantic模型
3. 实现路由处理函数
4. 在`main.py`中注册路由
5. 编写测试用例

示例：
```python
# backend/api/new_feature.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/feature", tags=["feature"])

class FeatureRequest(BaseModel):
    param: str

@router.post("/")
async def create_feature(req: FeatureRequest):
    return {"result": "success"}
```

### 添加新的服务模块
1. 在`backend/services/`创建服务文件
2. 实现服务类或函数
3. 添加日志记录
4. 处理异常情况
5. 编写单元测试

### 集成新的模型
1. 在`config.py`添加模型配置
2. 在`model_loader.py`添加加载逻辑
3. 在对应服务中调用模型
4. 优化显存使用
5. 测试推理性能

## 调试技巧

### 1. 使用日志
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)

logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### 2. 使用断点
```python
import pdb; pdb.set_trace()  # 设置断点
```

### 3. 查看API文档
启动服务后访问: `http://localhost:8000/docs`

### 4. 监控GPU使用
```bash
watch -n 1 nvidia-smi
```

## 性能优化

### 1. 模型优化
- 使用FP16半精度
- 启用模型量化
- 批量推理

### 2. 代码优化
- 使用异步IO
- 避免阻塞操作
- 合理使用缓存

### 3. 数据库优化
- 添加索引
- 使用连接池
- 批量操作

## 常见问题

### Q: 如何切换模型？
A: 修改`config.py`中的模型路径，重启服务

### Q: 如何增加并发任务数？
A: 修改`config.py`中的`MAX_CONCURRENT_TASKS`

### Q: 如何查看详细日志？
A: 查看`backend/logs/app.log`文件

### Q: 如何清理生成的视频？
A: 删除`backend/videos/`目录下的文件
