# 测试用例增强方案

## 📋 目录
1. [当前测试现状](#当前测试现状)
2. [测试覆盖分析](#测试覆盖分析)
3. [新增测试计划](#新增测试计划)
4. [实施步骤](#实施步骤)

---

## 1. 当前测试现状

### 1.1 已有测试

| 测试文件 | 测试内容 | 覆盖率 |
|---------|---------|--------|
| `test_llm_service.py` | LLM 服务 | 60% |
| `test_video_service.py` | 视频服务 | 60% |
| `test_model_loading.py` | 模型加载 | 80% |
| `test_script_generation.py` | 脚本生成 | 70% |
| `test_single_scene.py` | 单场景生成 | 70% |
| `test_end_to_end.py` | 端到端 | 70% |
| `test_benchmark.py` | 性能测试 | 60% |
| `test_memory_optimization.py` | 显存优化 | 70% |
| `test_database.py` | 数据库 | 80% |

**总体覆盖率**: ~70%

### 1.2 测试缺口

#### 缺少的测试类型
- ❌ 边界测试（边界值、极端情况）
- ❌ 异常测试（错误处理、异常恢复）
- ❌ 并发测试（多线程、竞态条件）
- ❌ 集成测试（模块间协作）
- ❌ 回归测试（防止功能退化）
- ❌ 压力测试（高负载场景）

#### 未覆盖的功能
- ❌ API 接口测试
- ❌ 配置验证测试
- ❌ 文件操作测试
- ❌ 错误恢复测试
- ❌ 数据验证测试

---

## 2. 测试覆盖分析

### 2.1 按模块分析

#### LLM 服务
- ✅ 基础脚本生成
- ✅ 备用方案
- ❌ 提示词验证
- ❌ JSON 解析错误处理
- ❌ 超长提示词处理
- ❌ 特殊字符处理

#### 视频服务
- ✅ 基础视频生成
- ✅ 备用方案
- ❌ 文件权限错误
- ❌ 磁盘空间不足
- ❌ 视频编码错误
- ❌ 并发生成

#### 数据库
- ✅ CRUD 操作
- ✅ 关联查询
- ❌ 事务回滚
- ❌ 并发写入
- ❌ 数据完整性约束
- ❌ 索引性能

#### API
- ❌ 请求验证
- ❌ 错误响应
- ❌ 认证授权
- ❌ 限流
- ❌ CORS

### 2.2 按测试类型分析

#### 单元测试
- **覆盖率**: 70%
- **缺口**: 边界测试、异常测试

#### 集成测试
- **覆盖率**: 50%
- **缺口**: 模块间协作、数据流

#### 端到端测试
- **覆盖率**: 60%
- **缺口**: 完整业务流程

#### 性能测试
- **覆盖率**: 40%
- **缺口**: 压力测试、并发测试

---

## 3. 新增测试计划

### 3.1 边界测试

#### 测试目标
验证系统在边界条件下的行为

#### 测试用例

**LLM 服务边界测试**
- 空提示词
- 超长提示词（10000字）
- 特殊字符（emoji、中文、符号）
- 极短提示词（1字）
- 重复提示词

**视频服务边界测试**
- 最小时长（1秒）
- 最大时长（60秒）
- 最小分辨率（256x256）
- 最大分辨率（2048x2048）
- 极多场景（100个）

**数据库边界测试**
- 最大字段长度
- 空值处理
- 重复数据
- 外键约束

### 3.2 异常测试

#### 测试目标
验证系统的错误处理和恢复能力

#### 测试用例

**模型加载异常**
- 模型文件不存在
- 模型文件损坏
- 显存不足
- CUDA 不可用

**生成异常**
- 生成超时
- 生成中断
- 输出格式错误
- 文件写入失败

**数据库异常**
- 连接失败
- 事务冲突
- 约束违反
- 磁盘满

### 3.3 并发测试

#### 测试目标
验证系统的并发处理能力

#### 测试用例

**并发任务创建**
- 10个并发请求
- 100个并发请求
- 竞态条件测试

**并发数据库操作**
- 并发读写
- 事务隔离
- 死锁检测

**并发视频生成**
- 多任务并行
- 资源竞争
- 队列管理

### 3.4 集成测试

#### 测试目标
验证模块间的协作

#### 测试用例

**完整业务流程**
- 用户注册 → 创建任务 → 生成视频 → 查询结果
- 配额管理 → 任务限制 → 配额恢复
- 错误处理 → 重试 → 成功

**数据流测试**
- 提示词 → 脚本 → 视频 → 存储
- 任务状态 → 进度更新 → 完成通知

### 3.5 API 测试

#### 测试目标
验证 API 接口的正确性

#### 测试用例

**任务 API**
- POST /api/tasks - 创建任务
- GET /api/tasks/{id} - 获取任务
- GET /api/tasks - 任务列表
- DELETE /api/tasks/{id} - 删除任务

**用户 API**
- POST /api/users - 创建用户
- GET /api/users/{id} - 获取用户
- PUT /api/users/{id} - 更新用户

**统计 API**
- GET /api/statistics - 获取统计

### 3.6 性能测试

#### 测试目标
验证系统的性能指标

#### 测试用例

**响应时间测试**
- API 响应时间 < 100ms
- 数据库查询 < 50ms
- 脚本生成 < 10s

**吞吐量测试**
- 每秒处理请求数
- 并发用户数
- 任务处理速度

**资源使用测试**
- CPU 使用率
- 内存使用率
- 显存使用率
- 磁盘 I/O

### 3.7 回归测试

#### 测试目标
防止功能退化

#### 测试用例

**核心功能回归**
- 脚本生成质量
- 视频生成质量
- 数据持久化
- 状态管理

**性能回归**
- 生成速度
- 显存占用
- 响应时间

---

## 4. 实施步骤

### 步骤 1: 边界测试（2小时）

**文件**: `tests/test_boundary.py`

**测试内容**:
- LLM 边界测试（10个用例）
- 视频边界测试（10个用例）
- 数据库边界测试（10个用例）

### 步骤 2: 异常测试（2小时）

**文件**: `tests/test_exceptions.py`

**测试内容**:
- 模型加载异常（5个用例）
- 生成异常（10个用例）
- 数据库异常（5个用例）

### 步骤 3: 并发测试（2小时）

**文件**: `tests/test_concurrency.py`

**测试内容**:
- 并发任务创建（3个用例）
- 并发数据库操作（5个用例）
- 资源竞争测试（3个用例）

### 步骤 4: 集成测试（2小时）

**文件**: `tests/test_integration.py`

**测试内容**:
- 完整业务流程（5个用例）
- 数据流测试（5个用例）
- 错误恢复测试（5个用例）

### 步骤 5: API 测试（2小时）

**文件**: `tests/test_api.py`（增强）

**测试内容**:
- 任务 API（10个用例）
- 用户 API（5个用例）
- 统计 API（3个用例）
- 错误响应（5个用例）

### 步骤 6: 性能测试（1小时）

**文件**: `tests/test_performance.py`

**测试内容**:
- 响应时间测试（5个用例）
- 吞吐量测试（3个用例）
- 资源使用测试（5个用例）

### 步骤 7: 回归测试（1小时）

**文件**: `tests/test_regression.py`

**测试内容**:
- 核心功能回归（10个用例）
- 性能回归（5个用例）

### 步骤 8: 测试工具（1小时）

**文件**: `tests/test_utils.py`

**工具函数**:
- 测试数据生成器
- Mock 对象工厂
- 断言辅助函数
- 性能计时器

### 步骤 9: 测试文档（1小时）

**文件**: `docs/TESTING_GUIDE.md`

**内容**:
- 测试策略
- 测试用例说明
- 运行指南
- 最佳实践

---

## 5. 测试框架增强

### 5.1 测试工具

#### Pytest 插件
```bash
pip install pytest-cov        # 覆盖率
pip install pytest-xdist      # 并行测试
pip install pytest-timeout    # 超时控制
pip install pytest-mock       # Mock 支持
pip install pytest-benchmark  # 性能测试
```

#### 测试配置
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --cov=backend
    --cov-report=html
    --cov-report=term
    --tb=short
```

### 5.2 Mock 和 Fixture

#### 通用 Fixture
```python
# conftest.py
import pytest
from models import get_db_context, init_db

@pytest.fixture(scope="session")
def test_db():
    """测试数据库"""
    init_db()
    yield
    # 清理

@pytest.fixture
def db_session():
    """数据库会话"""
    with get_db_context() as db:
        yield db

@pytest.fixture
def test_user(db_session):
    """测试用户"""
    from repositories import UserRepository
    repo = UserRepository(db_session)
    user = repo.create(username="test", quota=100)
    yield user
```

### 5.3 测试数据生成

#### Factory Pattern
```python
# test_factories.py
import factory
from models import User, Task

class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda o: f"{o.username}@test.com")
    quota = 100

class TaskFactory(factory.Factory):
    class Meta:
        model = Task
    
    task_id = factory.Faker('uuid4')
    prompt = factory.Faker('sentence')
    status = "pending"
```

---

## 6. 测试指标

### 6.1 覆盖率目标

| 模块 | 当前 | 目标 | 优先级 |
|------|------|------|--------|
| LLM 服务 | 60% | 85% | P0 |
| 视频服务 | 60% | 85% | P0 |
| 数据库 | 80% | 95% | P0 |
| API | 40% | 80% | P1 |
| 工具类 | 50% | 75% | P2 |
| **总体** | **70%** | **85%** | - |

### 6.2 质量指标

- **测试通过率**: > 95%
- **测试执行时间**: < 5分钟
- **代码覆盖率**: > 85%
- **Bug 检出率**: > 90%

---

## 7. 持续集成

### 7.1 CI 配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r backend/requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=backend --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### 7.2 Pre-commit Hook

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

---

## 8. 验收标准

### 功能验收
- ✅ 新增 100+ 测试用例
- ✅ 覆盖率提升到 85%+
- ✅ 所有测试通过
- ✅ 文档完整

### 质量验收
- ✅ 边界测试覆盖
- ✅ 异常测试覆盖
- ✅ 并发测试覆盖
- ✅ 集成测试覆盖

### 性能验收
- ✅ 测试执行时间 < 5分钟
- ✅ 并行测试支持
- ✅ 性能基准建立

---

## 9. 总结

### 实施收益

- ✅ **覆盖率提升**: 70% → 85%
- ✅ **测试用例**: 50+ → 150+
- ✅ **质量保障**: 更全面的测试
- ✅ **持续集成**: 自动化测试

### 实施计划

- **总时间**: 12-14 小时
- **优先级**: 高
- **风险**: 低
- **收益**: 高

### 下一步

1. 实施边界测试
2. 实施异常测试
3. 实施并发测试
4. 实施集成测试
5. 实施 API 测试
6. 实施性能测试
7. 实施回归测试
8. 编写测试文档

---

**准备开始实施！** 🚀
