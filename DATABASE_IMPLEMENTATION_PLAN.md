# 数据库持久化实现方案

## 📋 目录
1. [需求分析](#需求分析)
2. [技术选型](#技术选型)
3. [数据模型设计](#数据模型设计)
4. [实现方案](#实现方案)
5. [实施步骤](#实施步骤)

---

## 1. 需求分析

### 1.1 当前问题

- **无持久化**: 任务数据只在内存中，重启丢失
- **无历史记录**: 无法查询历史生成记录
- **无状态管理**: 任务状态无法追踪
- **无用户管理**: 无法区分不同用户的任务
- **无统计分析**: 无法统计使用情况

### 1.2 功能需求

#### 核心功能
- ✅ 任务持久化（创建、更新、查询）
- ✅ 用户管理（基础用户信息）
- ✅ 历史记录（生成历史查询）
- ✅ 状态追踪（任务状态管理）
- ✅ 文件关联（视频文件路径存储）

#### 扩展功能
- ⏳ 统计分析（使用统计）
- ⏳ 错误日志（失败记录）
- ⏳ 性能监控（生成时间统计）
- ⏳ 配额管理（用户配额限制）

### 1.3 非功能需求

- **性能**: 查询响应 < 100ms
- **可靠性**: 数据不丢失
- **可扩展性**: 支持未来功能扩展
- **易维护性**: 代码清晰，文档完善

---

## 2. 技术选型

### 2.1 数据库选择

#### 方案对比

| 数据库 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| **SQLite** | 轻量、无需配置、文件存储 | 并发性能有限 | 单机部署、中小规模 |
| PostgreSQL | 功能强大、高并发 | 需要独立部署 | 大规模生产环境 |
| MySQL | 成熟稳定、生态好 | 需要独立部署 | 传统 Web 应用 |
| MongoDB | 灵活 Schema、高性能 | 学习成本高 | 文档型数据 |

#### 选择：SQLite（推荐）

**理由**:
1. ✅ 零配置，开箱即用
2. ✅ 文件存储，易于备份
3. ✅ 轻量级，适合本地部署
4. ✅ Python 原生支持
5. ✅ 满足中小规模需求

**升级路径**:
- 未来可轻松迁移到 PostgreSQL/MySQL

### 2.2 ORM 选择

#### 方案对比

| ORM | 优点 | 缺点 |
|-----|------|------|
| **SQLAlchemy** | 功能强大、灵活、生态好 | 学习曲线陡 |
| Peewee | 轻量、简单 | 功能有限 |
| Tortoise ORM | 异步支持好 | 生态较小 |

#### 选择：SQLAlchemy

**理由**:
1. ✅ 功能最强大
2. ✅ 社区活跃
3. ✅ 文档完善
4. ✅ 支持多种数据库
5. ✅ FastAPI 集成好

---

## 3. 数据模型设计

### 3.1 ER 图

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│    User     │1      * │    Task     │1      * │   Video     │
│─────────────│◄────────│─────────────│◄────────│─────────────│
│ id          │         │ id          │         │ id          │
│ username    │         │ user_id     │         │ task_id     │
│ email       │         │ prompt      │         │ scene_num   │
│ created_at  │         │ status      │         │ file_path   │
│ ...         │         │ created_at  │         │ duration    │
└─────────────┘         │ ...         │         │ ...         │
                        └─────────────┘         └─────────────┘
                               │
                               │1
                               │
                               │*
                        ┌─────────────┐
                        │   Script    │
                        │─────────────│
                        │ id          │
                        │ task_id     │
                        │ scene_num   │
                        │ description │
                        │ duration    │
                        │ ...         │
                        └─────────────┘
```

### 3.2 数据表设计

#### 表 1: users（用户表）

| 字段 | 类型 | 说明 | 约束 |
|------|------|------|------|
| id | Integer | 用户ID | PK, Auto |
| username | String(50) | 用户名 | Unique, Not Null |
| email | String(100) | 邮箱 | Unique |
| api_key | String(64) | API密钥 | Unique |
| quota | Integer | 配额（次数） | Default: 100 |
| used_quota | Integer | 已使用配额 | Default: 0 |
| is_active | Boolean | 是否激活 | Default: True |
| created_at | DateTime | 创建时间 | Default: Now |
| updated_at | DateTime | 更新时间 | OnUpdate: Now |

**索引**:
- `idx_username` (username)
- `idx_email` (email)
- `idx_api_key` (api_key)

#### 表 2: tasks（任务表）

| 字段 | 类型 | 说明 | 约束 |
|------|------|------|------|
| id | Integer | 任务ID | PK, Auto |
| task_id | String(36) | 任务UUID | Unique, Not Null |
| user_id | Integer | 用户ID | FK(users.id) |
| prompt | Text | 用户提示词 | Not Null |
| status | String(20) | 任务状态 | Not Null |
| progress | Integer | 进度(0-100) | Default: 0 |
| total_scenes | Integer | 总场景数 | Default: 0 |
| completed_scenes | Integer | 完成场景数 | Default: 0 |
| final_video_path | String(500) | 最终视频路径 | Nullable |
| error_message | Text | 错误信息 | Nullable |
| created_at | DateTime | 创建时间 | Default: Now |
| started_at | DateTime | 开始时间 | Nullable |
| completed_at | DateTime | 完成时间 | Nullable |
| duration | Float | 耗时(秒) | Nullable |

**状态枚举**:
- `pending`: 等待中
- `processing`: 处理中
- `completed`: 已完成
- `failed`: 失败
- `cancelled`: 已取消

**索引**:
- `idx_task_id` (task_id)
- `idx_user_id` (user_id)
- `idx_status` (status)
- `idx_created_at` (created_at)

#### 表 3: scripts（脚本表）

| 字段 | 类型 | 说明 | 约束 |
|------|------|------|------|
| id | Integer | 脚本ID | PK, Auto |
| task_id | Integer | 任务ID | FK(tasks.id) |
| scene_number | Integer | 场景编号 | Not Null |
| description | Text | 场景描述 | Not Null |
| duration | Integer | 时长(秒) | Not Null |
| camera_movement | String(50) | 镜头运动 | Nullable |
| lighting | String(50) | 光照 | Nullable |
| created_at | DateTime | 创建时间 | Default: Now |

**索引**:
- `idx_task_id` (task_id)
- `idx_scene_number` (task_id, scene_number)

#### 表 4: videos（视频表）

| 字段 | 类型 | 说明 | 约束 |
|------|------|------|------|
| id | Integer | 视频ID | PK, Auto |
| task_id | Integer | 任务ID | FK(tasks.id) |
| scene_number | Integer | 场景编号 | Not Null |
| file_path | String(500) | 文件路径 | Not Null |
| file_size | Integer | 文件大小(bytes) | Nullable |
| duration | Float | 时长(秒) | Nullable |
| width | Integer | 宽度 | Nullable |
| height | Integer | 高度 | Nullable |
| fps | Integer | 帧率 | Nullable |
| status | String(20) | 状态 | Default: 'completed' |
| created_at | DateTime | 创建时间 | Default: Now |

**索引**:
- `idx_task_id` (task_id)
- `idx_scene_number` (task_id, scene_number)

#### 表 5: statistics（统计表）

| 字段 | 类型 | 说明 | 约束 |
|------|------|------|------|
| id | Integer | 统计ID | PK, Auto |
| date | Date | 日期 | Unique, Not Null |
| total_tasks | Integer | 总任务数 | Default: 0 |
| completed_tasks | Integer | 完成任务数 | Default: 0 |
| failed_tasks | Integer | 失败任务数 | Default: 0 |
| total_videos | Integer | 总视频数 | Default: 0 |
| total_duration | Float | 总时长(秒) | Default: 0 |
| avg_generation_time | Float | 平均生成时间 | Default: 0 |
| created_at | DateTime | 创建时间 | Default: Now |

**索引**:
- `idx_date` (date)

---

## 4. 实现方案

### 4.1 技术架构

```
┌─────────────────────────────────────────┐
│           API Layer (FastAPI)           │
│  ┌─────────────────────────────────┐   │
│  │  Task API  │  User API  │ Stats │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Service Layer (Business)         │
│  ┌─────────────────────────────────┐   │
│  │ TaskService │ UserService │ ... │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Repository Layer (DAO)          │
│  ┌─────────────────────────────────┐   │
│  │ TaskRepo │ UserRepo │ VideoRepo │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       ORM Layer (SQLAlchemy)            │
│  ┌─────────────────────────────────┐   │
│  │  Models  │  Session  │  Engine  │   │
│  └─────────────────────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Database (SQLite)              │
│         video_platform.db               │
└─────────────────────────────────────────┘
```

### 4.2 目录结构

```
backend/
├── models/
│   ├── __init__.py
│   ├── database.py          # 数据库配置和会话
│   ├── user.py              # User 模型
│   ├── task.py              # Task 模型
│   ├── script.py            # Script 模型
│   ├── video.py             # Video 模型
│   └── statistics.py        # Statistics 模型
├── repositories/            # 新增
│   ├── __init__.py
│   ├── base.py              # 基础仓储
│   ├── user_repository.py   # 用户仓储
│   ├── task_repository.py   # 任务仓储
│   └── video_repository.py  # 视频仓储
├── services/
│   ├── task_service.py      # 任务服务（更新）
│   └── ...
└── api/
    ├── tasks.py             # 任务API（更新）
    └── ...
```

### 4.3 核心功能

#### 4.3.1 任务生命周期管理

```python
# 创建任务
task = create_task(user_id, prompt)
# status: pending

# 开始处理
start_task(task_id)
# status: processing, started_at: now

# 更新进度
update_progress(task_id, progress=50, completed_scenes=2)

# 完成任务
complete_task(task_id, video_path)
# status: completed, completed_at: now, duration: calculated

# 失败处理
fail_task(task_id, error_message)
# status: failed, error_message: saved
```

#### 4.3.2 查询功能

```python
# 获取任务详情
task = get_task(task_id)

# 获取用户任务列表
tasks = get_user_tasks(user_id, status=None, limit=10)

# 获取任务脚本
scripts = get_task_scripts(task_id)

# 获取任务视频
videos = get_task_videos(task_id)

# 统计查询
stats = get_statistics(date_from, date_to)
```

#### 4.3.3 事务管理

```python
# 使用上下文管理器
with get_db() as db:
    task = create_task(db, ...)
    script = create_script(db, task.id, ...)
    db.commit()

# 自动回滚
try:
    with get_db() as db:
        # 操作...
        db.commit()
except Exception as e:
    # 自动回滚
    logger.error(f"Transaction failed: {e}")
```

---

## 5. 实施步骤

### 步骤 1: 数据库基础设施（1小时）

**文件**: `backend/models/database.py`

**任务**:
- ✅ 配置 SQLAlchemy
- ✅ 创建数据库引擎
- ✅ 配置会话管理
- ✅ 创建基础模型类

**代码**:
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
```

### 步骤 2: 数据模型定义（1-2小时）

**文件**: 
- `backend/models/user.py`
- `backend/models/task.py`
- `backend/models/script.py`
- `backend/models/video.py`
- `backend/models/statistics.py`

**任务**:
- ✅ 定义所有模型类
- ✅ 配置关系映射
- ✅ 添加索引和约束
- ✅ 添加辅助方法

### 步骤 3: 仓储层实现（1-2小时）

**文件**: `backend/repositories/*.py`

**任务**:
- ✅ 实现基础仓储类
- ✅ 实现任务仓储
- ✅ 实现用户仓储
- ✅ 实现视频仓储
- ✅ 添加常用查询方法

### 步骤 4: 服务层集成（1小时）

**文件**: `backend/services/task_service.py`

**任务**:
- ✅ 集成数据库操作
- ✅ 更新任务创建逻辑
- ✅ 添加状态更新逻辑
- ✅ 添加查询方法

### 步骤 5: API 层更新（1小时）

**文件**: `backend/api/tasks.py`

**任务**:
- ✅ 更新任务创建接口
- ✅ 添加任务查询接口
- ✅ 添加历史记录接口
- ✅ 添加统计接口

### 步骤 6: 数据库迁移（30分钟）

**文件**: `backend/migrations/init_db.py`

**任务**:
- ✅ 创建初始化脚本
- ✅ 创建所有表
- ✅ 添加初始数据
- ✅ 测试迁移

### 步骤 7: 测试（1小时）

**文件**: `tests/test_database.py`

**任务**:
- ✅ 单元测试（模型、仓储）
- ✅ 集成测试（服务、API）
- ✅ 性能测试
- ✅ 边界测试

### 步骤 8: 文档（30分钟）

**文件**: `docs/DATABASE_GUIDE.md`

**任务**:
- ✅ 数据库设计文档
- ✅ API 使用文档
- ✅ 迁移指南
- ✅ 常见问题

---

## 6. 数据库操作示例

### 6.1 创建任务

```python
from repositories.task_repository import TaskRepository
from models.database import get_db

# 创建任务
with get_db() as db:
    repo = TaskRepository(db)
    task = repo.create(
        task_id="uuid-xxx",
        user_id=1,
        prompt="制作森林探险视频",
        status="pending"
    )
    db.commit()
```

### 6.2 更新任务状态

```python
# 开始任务
repo.update_status(task_id, "processing")
repo.update_started_at(task_id, datetime.now())

# 更新进度
repo.update_progress(task_id, progress=50, completed_scenes=2)

# 完成任务
repo.update_status(task_id, "completed")
repo.update_completed_at(task_id, datetime.now())
repo.update_video_path(task_id, "/path/to/video.mp4")
```

### 6.3 查询任务

```python
# 获取单个任务
task = repo.get_by_task_id(task_id)

# 获取用户任务列表
tasks = repo.get_by_user(user_id, status="completed", limit=10)

# 获取最近任务
recent_tasks = repo.get_recent(limit=20)

# 统计查询
total = repo.count_by_status("completed")
```

### 6.4 关联查询

```python
# 获取任务及其脚本
task = repo.get_with_scripts(task_id)
for script in task.scripts:
    print(script.description)

# 获取任务及其视频
task = repo.get_with_videos(task_id)
for video in task.videos:
    print(video.file_path)
```

---

## 7. 性能优化

### 7.1 索引优化

```python
# 常用查询字段添加索引
Index('idx_task_user_status', Task.user_id, Task.status)
Index('idx_task_created', Task.created_at.desc())
```

### 7.2 查询优化

```python
# 使用 joinedload 避免 N+1 查询
from sqlalchemy.orm import joinedload

task = db.query(Task)\
    .options(joinedload(Task.scripts))\
    .options(joinedload(Task.videos))\
    .filter(Task.task_id == task_id)\
    .first()
```

### 7.3 批量操作

```python
# 批量插入
scripts = [Script(...) for scene in scenes]
db.bulk_save_objects(scripts)
db.commit()
```

### 7.4 连接池配置

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # 连接池大小
    max_overflow=20,       # 最大溢出连接
    pool_timeout=30,       # 连接超时
    pool_recycle=3600      # 连接回收时间
)
```

---

## 8. 数据备份和恢复

### 8.1 备份策略

```bash
# 定期备份（每天）
cp video_platform.db backups/video_platform_$(date +%Y%m%d).db

# 压缩备份
tar -czf backups/db_$(date +%Y%m%d).tar.gz video_platform.db
```

### 8.2 恢复

```bash
# 从备份恢复
cp backups/video_platform_20240101.db video_platform.db
```

### 8.3 导出数据

```python
# 导出为 JSON
import json
tasks = db.query(Task).all()
with open('tasks.json', 'w') as f:
    json.dump([task.to_dict() for task in tasks], f)
```

---

## 9. 安全考虑

### 9.1 SQL 注入防护

```python
# ✅ 使用参数化查询
task = db.query(Task).filter(Task.task_id == task_id).first()

# ❌ 避免字符串拼接
# task = db.execute(f"SELECT * FROM tasks WHERE task_id = '{task_id}'")
```

### 9.2 敏感数据加密

```python
# API Key 加密存储
import hashlib
api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
```

### 9.3 访问控制

```python
# 验证用户权限
def get_task(task_id, user_id):
    task = repo.get_by_task_id(task_id)
    if task.user_id != user_id:
        raise PermissionError("Access denied")
    return task
```

---

## 10. 监控和维护

### 10.1 数据库监控

```python
# 监控数据库大小
import os
db_size = os.path.getsize('video_platform.db') / 1024 / 1024
logger.info(f"Database size: {db_size:.2f} MB")

# 监控表大小
SELECT name, SUM(pgsize) as size 
FROM dbstat 
GROUP BY name 
ORDER BY size DESC;
```

### 10.2 性能监控

```python
# 慢查询日志
import time

start = time.time()
result = db.query(Task).all()
duration = time.time() - start

if duration > 1.0:
    logger.warning(f"Slow query: {duration:.2f}s")
```

### 10.3 定期维护

```sql
-- SQLite 优化
VACUUM;  -- 清理碎片
ANALYZE; -- 更新统计信息
```

---

## 11. 迁移到生产数据库

### 11.1 迁移到 PostgreSQL

```python
# 更新配置
DATABASE_URL = "postgresql://user:pass@localhost/dbname"

# 代码无需修改（SQLAlchemy 抽象）
engine = create_engine(DATABASE_URL)
```

### 11.2 数据迁移

```bash
# 使用 pgloader
pgloader video_platform.db postgresql://localhost/dbname
```

---

## 12. 验收标准

### 功能验收
- ✅ 任务可以持久化存储
- ✅ 任务状态可以追踪
- ✅ 历史记录可以查询
- ✅ 关联数据完整
- ✅ 事务正确处理

### 性能验收
- ✅ 查询响应 < 100ms
- ✅ 插入响应 < 50ms
- ✅ 支持并发操作
- ✅ 数据库大小合理

### 质量验收
- ✅ 代码覆盖率 > 80%
- ✅ 无 SQL 注入风险
- ✅ 数据完整性约束
- ✅ 文档完整

---

## 13. 总结

### 实施收益

- ✅ **数据持久化**: 任务数据永久保存
- ✅ **状态管理**: 完整的任务生命周期追踪
- ✅ **历史查询**: 支持历史记录查询
- ✅ **统计分析**: 支持使用统计
- ✅ **可扩展性**: 易于添加新功能

### 技术亮点

1. **分层架构**: Model → Repository → Service → API
2. **ORM 抽象**: 易于切换数据库
3. **事务管理**: 保证数据一致性
4. **性能优化**: 索引、连接池、查询优化
5. **安全设计**: 防 SQL 注入、权限控制

### 实施计划

- **总时间**: 6-8 小时
- **优先级**: 高
- **风险**: 低
- **收益**: 高

---

**准备开始实施！** 🚀
