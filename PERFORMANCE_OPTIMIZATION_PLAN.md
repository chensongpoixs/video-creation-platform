# 性能优化和压力测试实施方案

## 📋 目录
1. [需求分析](#需求分析)
2. [性能分析](#性能分析)
3. [优化策略](#优化策略)
4. [压力测试](#压力测试)
5. [实施方案](#实施方案)

---

## 1. 需求分析

### 1.1 当前性能状况

#### 已知性能瓶颈
- ❌ 模型加载时间长（LLM + Video Model）
- ❌ 视频生成耗时（单场景 ~30秒）
- ❌ 后处理串行执行
- ❌ 数据库查询未优化
- ❌ 无并发控制
- ❌ 内存占用高

#### 性能目标
- ✅ API 响应时间 < 100ms
- ✅ 视频生成时间 < 2分钟（3场景）
- ✅ 并发支持 10+ 用户
- ✅ 内存占用 < 16GB
- ✅ CPU 利用率 < 80%

### 1.2 优化需求

#### 核心优化
1. **模型优化**: 模型预加载、缓存、量化
2. **并发优化**: 异步处理、任务队列、线程池
3. **数据库优化**: 索引、连接池、查询优化
4. **缓存优化**: Redis、内存缓存、结果缓存
5. **资源优化**: 内存管理、GPU 利用、文件清理

#### 压力测试
1. **API 压力测试**: 并发请求、响应时间
2. **视频生成压力测试**: 多任务并发
3. **数据库压力测试**: 高并发读写
4. **内存压力测试**: 内存泄漏检测
5. **长时间运行测试**: 稳定性测试

---

## 2. 性能分析

### 2.1 性能瓶颈分析

#### 2.1.1 模型加载
```
问题: 首次加载耗时 30-60秒
原因:
- 模型文件大（LLM 6GB + Video 10GB）
- 从磁盘加载慢
- 初始化耗时

优化方案:
- 预加载模型
- 使用模型缓存
- 延迟加载
```

#### 2.1.2 视频生成
```
问题: 单场景生成 30秒，3场景 90秒
原因:
- Diffusion 模型推理慢
- 串行处理场景
- 后处理串行

优化方案:
- 并行生成场景
- 批量处理
- 异步后处理
```

#### 2.1.3 数据库操作
```
问题: 高并发时响应慢
原因:
- 无连接池
- 无索引优化
- N+1 查询问题

优化方案:
- 添加连接池
- 优化索引
- 使用 ORM 优化
```

#### 2.1.4 内存占用
```
问题: 内存占用 20GB+
原因:
- 模型占用大
- 视频帧缓存
- 无内存释放

优化方案:
- FP16 量化
- 及时释放内存
- 流式处理
```

### 2.2 性能指标

#### 当前性能

| 指标 | 当前值 | 目标值 | 差距 |
|------|--------|--------|------|
| API 响应 | ~200ms | <100ms | 2x |
| 视频生成 | ~90s | <120s | ✅ |
| 并发用户 | 1-2 | 10+ | 5x |
| 内存占用 | 20GB | <16GB | 1.25x |
| CPU 利用率 | 60% | <80% | ✅ |

#### 优化目标

| 优化项 | 预期提升 |
|--------|----------|
| 模型预加载 | 启动时间 -50% |
| 并行处理 | 吞吐量 +200% |
| 数据库优化 | 查询速度 +50% |
| 缓存优化 | 响应时间 -30% |
| 内存优化 | 内存占用 -20% |

---

## 3. 优化策略

### 3.1 模型优化

#### 3.1.1 模型预加载
```python
# 应用启动时预加载模型
@app.on_event("startup")
async def startup_event():
    # 预加载 LLM 模型
    llm_loader.load_model()
    # 预加载视频模型
    video_loader.load_model()
```

#### 3.1.2 模型缓存
```python
# 使用单例模式缓存模型
class ModelCache:
    _instance = None
    _models = {}
    
    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._models:
            cls._models[model_name] = load_model(model_name)
        return cls._models[model_name]
```

#### 3.1.3 批量推理
```python
# 批量处理多个场景
def generate_scenes_batch(scenes, batch_size=2):
    for i in range(0, len(scenes), batch_size):
        batch = scenes[i:i+batch_size]
        results = model.generate_batch(batch)
        yield results
```

### 3.2 并发优化

#### 3.2.1 异步处理
```python
# 使用 asyncio 异步处理
async def generate_video_async(script, task_id):
    tasks = []
    for scene in script['scenes']:
        task = asyncio.create_task(generate_scene(scene))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

#### 3.2.2 任务队列
```python
# 使用 Celery 任务队列
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def generate_video_task(script, task_id):
    return generate_video(script, task_id)
```

#### 3.2.3 线程池
```python
# 使用线程池并行处理
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_scene, scene) 
               for scene in scenes]
    results = [f.result() for f in futures]
```

### 3.3 数据库优化

#### 3.3.1 连接池
```python
# SQLAlchemy 连接池配置
engine = create_engine(
    DATABASE_URL,
    pool_size=10,           # 连接池大小
    max_overflow=20,        # 最大溢出连接
    pool_timeout=30,        # 连接超时
    pool_recycle=3600,      # 连接回收时间
)
```

#### 3.3.2 索引优化
```python
# 添加索引
class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # 添加索引
    status = Column(String, index=True)    # 添加索引
    created_at = Column(DateTime, index=True)  # 添加索引
```

#### 3.3.3 查询优化
```python
# 使用 joinedload 避免 N+1 查询
from sqlalchemy.orm import joinedload

tasks = session.query(Task)\
    .options(joinedload(Task.user))\
    .filter(Task.status == 'pending')\
    .all()
```

### 3.4 缓存优化

#### 3.4.1 内存缓存
```python
# 使用 functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def get_user_by_id(user_id):
    return db.query(User).filter(User.id == user_id).first()
```

#### 3.4.2 Redis 缓存
```python
# 使用 Redis 缓存结果
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def get_cached_result(key):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def set_cached_result(key, value, expire=3600):
    redis_client.setex(key, expire, json.dumps(value))
```

#### 3.4.3 结果缓存
```python
# 缓存视频生成结果
def generate_video_cached(script_hash, task_id):
    cache_key = f"video:{script_hash}"
    cached = get_cached_result(cache_key)
    
    if cached:
        return cached
    
    result = generate_video(script, task_id)
    set_cached_result(cache_key, result)
    return result
```

### 3.5 资源优化

#### 3.5.1 内存管理
```python
# 及时释放内存
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### 3.5.2 文件清理
```python
# 定期清理临时文件
import os
import time

def cleanup_old_files(directory, max_age_days=7):
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            age = now - os.path.getmtime(filepath)
            if age > max_age_days * 86400:
                os.remove(filepath)
```

#### 3.5.3 GPU 利用
```python
# 优化 GPU 利用率
torch.backends.cudnn.benchmark = True  # 自动优化
torch.backends.cudnn.enabled = True
```

---

## 4. 压力测试

### 4.1 测试工具

#### 4.1.1 Locust（推荐）
```python
# 使用 Locust 进行压力测试
from locust import HttpUser, task, between

class VideoUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def create_task(self):
        self.client.post("/api/tasks", json={
            "prompt": "测试视频",
            "num_scenes": 3
        })
    
    @task
    def get_task(self):
        self.client.get("/api/tasks/1")
```

#### 4.1.2 Apache Bench
```bash
# 简单的 HTTP 压力测试
ab -n 1000 -c 10 http://localhost:8000/api/tasks
```

#### 4.1.3 pytest-benchmark
```python
# 性能基准测试
def test_api_performance(benchmark):
    result = benchmark(api_call)
    assert result.status_code == 200
```

### 4.2 测试场景

#### 4.2.1 API 压力测试
```
测试目标: API 响应时间和并发能力
测试方法:
- 并发用户: 10, 50, 100
- 请求数: 1000, 5000, 10000
- 测试端点: 所有 API 端点

指标:
- 平均响应时间
- 95% 响应时间
- 错误率
- 吞吐量（RPS）
```

#### 4.2.2 视频生成压力测试
```
测试目标: 视频生成性能和资源占用
测试方法:
- 并发任务: 1, 3, 5, 10
- 场景数: 1, 3, 5
- 持续时间: 30分钟

指标:
- 生成时间
- 内存占用
- GPU 利用率
- 成功率
```

#### 4.2.3 数据库压力测试
```
测试目标: 数据库性能和并发能力
测试方法:
- 并发连接: 10, 50, 100
- 操作类型: 读、写、混合
- 数据量: 1000, 10000, 100000

指标:
- 查询时间
- 事务吞吐量
- 连接池使用率
- 死锁次数
```

#### 4.2.4 内存压力测试
```
测试目标: 内存泄漏和稳定性
测试方法:
- 持续运行: 24小时
- 周期性任务: 每分钟1个
- 监控内存: 每秒采样

指标:
- 内存增长率
- 内存峰值
- GC 频率
- 内存泄漏
```

### 4.3 性能监控

#### 4.3.1 系统监控
```python
# 使用 psutil 监控系统资源
import psutil

def get_system_stats():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "network_io": psutil.net_io_counters()
    }
```

#### 4.3.2 应用监控
```python
# 使用装饰器监控函数性能
import time
import functools

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

#### 4.3.3 数据库监控
```python
# 监控数据库查询
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, 
                                  parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement,
                                 parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop()
    logger.debug(f"Query took {total:.3f}s: {statement}")
```

---

## 5. 实施方案

### 5.1 目录结构

```
backend/
├── utils/
│   ├── performance.py          # 性能监控工具（新增）⭐
│   ├── cache.py                # 缓存工具（新增）⭐
│   └── async_utils.py          # 异步工具（新增）⭐
├── middleware/
│   └── performance_middleware.py  # 性能中间件（新增）⭐
├── config.py                   # 配置文件（更新）
└── tests/
    ├── test_performance.py     # 性能测试（新增）⭐
    ├── test_stress.py          # 压力测试（新增）⭐
    └── locustfile.py           # Locust 测试（新增）⭐
```

### 5.2 核心实现

#### 5.2.1 性能监控工具
```python
class PerformanceMonitor:
    """性能监控类"""
    
    @staticmethod
    def monitor_function(func):
        """监控函数性能"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            # 记录性能数据
            return result
        return wrapper
    
    @staticmethod
    def get_system_metrics():
        """获取系统指标"""
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }
```

#### 5.2.2 缓存工具
```python
class CacheManager:
    """缓存管理类"""
    
    def __init__(self):
        self.memory_cache = {}
        self.redis_client = redis.Redis()
    
    def get(self, key):
        """获取缓存"""
        # 先查内存缓存
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 再查 Redis
        value = self.redis_client.get(key)
        if value:
            return json.loads(value)
        
        return None
    
    def set(self, key, value, expire=3600):
        """设置缓存"""
        self.memory_cache[key] = value
        self.redis_client.setex(key, expire, json.dumps(value))
```

#### 5.2.3 异步工具
```python
class AsyncTaskManager:
    """异步任务管理"""
    
    @staticmethod
    async def run_parallel(tasks):
        """并行运行任务"""
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def run_in_thread_pool(func, *args, max_workers=4):
        """在线程池中运行"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return executor.submit(func, *args).result()
```

### 5.3 配置更新

```python
# 性能优化配置
PERFORMANCE_CONFIG = {
    # 并发配置
    "max_workers": 4,
    "thread_pool_size": 10,
    "async_enabled": True,
    
    # 缓存配置
    "cache_enabled": True,
    "cache_ttl": 3600,
    "redis_url": "redis://localhost:6379",
    
    # 数据库配置
    "db_pool_size": 10,
    "db_max_overflow": 20,
    "db_pool_timeout": 30,
    
    # 监控配置
    "monitoring_enabled": True,
    "metrics_interval": 60,
    
    # 资源限制
    "max_memory_percent": 80,
    "max_cpu_percent": 80,
}
```

---

## 6. 实施步骤

### 步骤 1: 性能监控工具（30分钟）
- 创建 `utils/performance.py`
- 实现性能监控装饰器
- 实现系统指标采集

### 步骤 2: 缓存系统（40分钟）
- 创建 `utils/cache.py`
- 实现内存缓存
- 实现 Redis 缓存（可选）

### 步骤 3: 异步工具（30分钟）
- 创建 `utils/async_utils.py`
- 实现异步任务管理
- 实现线程池管理

### 步骤 4: 性能中间件（20分钟）
- 创建 `middleware/performance_middleware.py`
- 实现请求性能监控
- 集成到应用

### 步骤 5: 数据库优化（30分钟）
- 更新数据库配置
- 添加连接池
- 优化查询

### 步骤 6: 性能测试（40分钟）
- 创建 `tests/test_performance.py`
- 实现性能基准测试
- 实现性能回归测试

### 步骤 7: 压力测试（40分钟）
- 创建 `tests/test_stress.py`
- 实现并发测试
- 实现负载测试

### 步骤 8: Locust 测试（30分钟）
- 创建 `tests/locustfile.py`
- 实现用户行为模拟
- 配置测试场景

### 步骤 9: 优化应用（40分钟）
- 应用性能优化
- 添加缓存
- 优化并发

### 步骤 10: 文档和报告（30分钟）
- 编写优化文档
- 生成性能报告
- 更新 README

**总时间**: 约 5-6 小时

---

## 7. 验收标准

### 性能指标
- ✅ API 响应时间 < 100ms
- ✅ 视频生成时间 < 2分钟
- ✅ 并发支持 10+ 用户
- ✅ 内存占用 < 16GB
- ✅ CPU 利用率 < 80%

### 压力测试
- ✅ 1000 并发请求无错误
- ✅ 24小时稳定运行
- ✅ 无内存泄漏
- ✅ 无性能退化

### 监控指标
- ✅ 实时性能监控
- ✅ 资源使用监控
- ✅ 错误率监控
- ✅ 性能报告生成

---

## 8. 总结

### 优化收益
- ✅ **性能提升**: 响应时间减少 50%
- ✅ **并发能力**: 支持 10+ 并发用户
- ✅ **资源优化**: 内存占用减少 20%
- ✅ **稳定性**: 24小时稳定运行

### 技术亮点
1. **全面监控**: 系统、应用、数据库
2. **多级缓存**: 内存 + Redis
3. **并发优化**: 异步 + 线程池
4. **压力测试**: Locust + pytest

---

**准备开始实施！** 🚀

