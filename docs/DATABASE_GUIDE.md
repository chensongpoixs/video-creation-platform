# 数据库使用指南

本文档介绍数据库的设计、使用方法和最佳实践。

---

## 📊 数据库设计

### 数据表

系统包含 5 个核心数据表：

1. **users** - 用户表
2. **tasks** - 任务表
3. **scripts** - 脚本表（分镜）
4. **videos** - 视频表
5. **statistics** - 统计表

### ER 关系

```
User (1) ──< (N) Task (1) ──< (N) Script
                    │
                    └──< (N) Video
```

---

## 🚀 快速开始

### 1. 初始化数据库

```bash
# 创建数据库表和默认用户
python scripts/init_database.py
```

**输出**:
```
✅ 数据库初始化完成
✅ 创建默认用户成功
   用户名: admin
   API Key: xxx...
```

### 2. 测试数据库

```bash
# 运行数据库测试
python tests/test_database.py
```

---

## 💻 使用方法

### 基础操作

#### 1. 获取数据库会话

```python
from models import get_db_context

# 方式 1: 上下文管理器（推荐）
with get_db_context() as db:
    # 数据库操作
    user = User(username="test")
    db.add(user)
    # 自动提交

# 方式 2: FastAPI 依赖注入
from fastapi import Depends
from models import get_db

@app.get("/tasks")
def get_tasks(db: Session = Depends(get_db)):
    return db.query(Task).all()
```

#### 2. 使用仓储层

```python
from repositories import TaskRepository, UserRepository

with get_db_context() as db:
    # 创建仓储实例
    task_repo = TaskRepository(db)
    user_repo = UserRepository(db)
    
    # 使用仓储方法
    task = task_repo.get_by_task_id("uuid-xxx")
    user = user_repo.get_by_username("admin")
```

### 任务操作

#### 创建任务

```python
from repositories import TaskRepository
from models import get_db_context, TaskStatus
import uuid

with get_db_context() as db:
    repo = TaskRepository(db)
    
    task = repo.create(
        task_id=str(uuid.uuid4()),
        user_id=1,
        prompt="制作森林探险视频",
        status=TaskStatus.PENDING.value,
        total_scenes=5
    )
    
    print(f"任务创建成功: {task.task_id}")
```

#### 更新任务状态

```python
# 开始任务
repo.start_task(task_id)

# 更新进度
repo.update_progress(task_id, progress=50, completed_scenes=2)

# 完成任务
repo.complete_task(task_id, video_path="/path/to/video.mp4")

# 任务失败
repo.fail_task(task_id, error_message="生成失败")
```

#### 查询任务

```python
# 根据任务ID查询
task = repo.get_by_task_id(task_id)

# 查询用户任务
tasks = repo.get_by_user(user_id, status="completed", limit=10)

# 查询最近任务
recent_tasks = repo.get_recent(limit=20)

# 获取任务及关联数据
task = repo.get_with_relations(task_id)
print(f"脚本数: {len(task.scripts)}")
print(f"视频数: {len(task.videos)}")
```

### 用户操作

#### 创建用户

```python
from repositories import UserRepository
import secrets

with get_db_context() as db:
    repo = UserRepository(db)
    
    user = repo.create(
        username="newuser",
        email="user@example.com",
        api_key=secrets.token_urlsafe(32),
        quota=100
    )
```

#### 配额管理

```python
# 使用配额
repo.use_quota(user_id, amount=1)

# 重置配额
repo.reset_quota(user_id, quota=100)

# 检查配额
user = repo.get(user_id)
if user.has_quota():
    print("配额充足")
```

### 脚本和视频操作

#### 创建脚本

```python
from models import Script

with get_db_context() as db:
    script = Script(
        task_id=task.id,
        scene_number=1,
        description="森林场景",
        duration=3
    )
    db.add(script)
```

#### 创建视频

```python
from repositories import VideoRepository

with get_db_context() as db:
    repo = VideoRepository(db)
    
    video = repo.create(
        task_id=task.id,
        scene_number=1,
        file_path="/path/to/scene_1.mp4",
        file_size=1024*1024*5,  # 5MB
        duration=3.0,
        width=1024,
        height=576,
        fps=6
    )
```

---

## 📈 统计查询

### 任务统计

```python
from repositories import TaskRepository
from datetime import datetime, timedelta

with get_db_context() as db:
    repo = TaskRepository(db)
    
    # 获取统计
    stats = repo.get_statistics(
        date_from=datetime.now() - timedelta(days=7),
        date_to=datetime.now()
    )
    
    print(f"总任务数: {stats['total']}")
    print(f"完成数: {stats['completed']}")
    print(f"成功率: {stats['success_rate']}%")
    print(f"平均耗时: {stats['avg_duration']}秒")
```

### 用户统计

```python
# 统计用户任务数
total = repo.count_by_user(user_id)
completed = repo.count_by_user(user_id, status="completed")

print(f"总任务: {total}, 已完成: {completed}")
```

---

## 🔧 高级功能

### 关联查询

```python
from sqlalchemy.orm import joinedload

# 预加载关联数据（避免 N+1 查询）
task = db.query(Task)\
    .options(joinedload(Task.scripts))\
    .options(joinedload(Task.videos))\
    .filter(Task.task_id == task_id)\
    .first()

# 访问关联数据（不会触发额外查询）
for script in task.scripts:
    print(script.description)
```

### 批量操作

```python
# 批量插入
scripts = [
    Script(task_id=task.id, scene_number=i, description=f"场景{i}")
    for i in range(1, 6)
]
db.bulk_save_objects(scripts)
db.commit()
```

### 事务管理

```python
from models import get_db_context

try:
    with get_db_context() as db:
        # 操作 1
        task = Task(...)
        db.add(task)
        
        # 操作 2
        script = Script(...)
        db.add(script)
        
        # 自动提交
except Exception as e:
    # 自动回滚
    print(f"事务失败: {e}")
```

---

## 🎯 最佳实践

### 1. 使用仓储层

✅ **推荐**:
```python
repo = TaskRepository(db)
task = repo.get_by_task_id(task_id)
```

❌ **不推荐**:
```python
task = db.query(Task).filter(Task.task_id == task_id).first()
```

### 2. 使用上下文管理器

✅ **推荐**:
```python
with get_db_context() as db:
    # 操作
    pass  # 自动提交/回滚
```

❌ **不推荐**:
```python
db = SessionLocal()
try:
    # 操作
    db.commit()
except:
    db.rollback()
finally:
    db.close()
```

### 3. 避免 N+1 查询

✅ **推荐**:
```python
# 使用 joinedload 预加载
task = repo.get_with_relations(task_id)
```

❌ **不推荐**:
```python
task = repo.get_by_task_id(task_id)
for script in task.scripts:  # 每次循环都查询数据库
    print(script.description)
```

### 4. 使用模型方法

✅ **推荐**:
```python
task.start()  # 使用模型方法
task.complete(video_path)
```

❌ **不推荐**:
```python
task.status = "processing"
task.started_at = datetime.now()
```

---

## 🔍 常见问题

### Q1: 如何重置数据库？

```bash
# 删除数据库文件
rm backend/video_platform.db

# 重新初始化
python scripts/init_database.py
```

### Q2: 如何备份数据库？

```bash
# SQLite 备份
cp backend/video_platform.db backups/db_$(date +%Y%m%d).db
```

### Q3: 如何查看数据库内容？

```bash
# 使用 SQLite 命令行
sqlite3 backend/video_platform.db

# 查看表
.tables

# 查询数据
SELECT * FROM tasks LIMIT 10;
```

### Q4: 如何迁移到 PostgreSQL？

```python
# 1. 更新配置
DATABASE_URL = "postgresql://user:pass@localhost/dbname"

# 2. 重新初始化
python scripts/init_database.py

# 3. 代码无需修改（SQLAlchemy 抽象）
```

### Q5: 如何处理并发？

```python
# SQLite 默认支持并发读，但写入需要锁
# 对于高并发场景，建议使用 PostgreSQL

# 配置连接池
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20
)
```

---

## 📚 API 参考

### TaskRepository

| 方法 | 说明 |
|------|------|
| `get_by_task_id(task_id)` | 根据任务ID获取 |
| `get_with_relations(task_id)` | 获取任务及关联数据 |
| `get_by_user(user_id, status, limit)` | 获取用户任务列表 |
| `get_recent(limit)` | 获取最近任务 |
| `start_task(task_id)` | 开始任务 |
| `complete_task(task_id, video_path)` | 完成任务 |
| `fail_task(task_id, error_message)` | 任务失败 |
| `update_progress(task_id, progress, completed_scenes)` | 更新进度 |
| `get_statistics(date_from, date_to)` | 获取统计 |

### UserRepository

| 方法 | 说明 |
|------|------|
| `get_by_username(username)` | 根据用户名获取 |
| `get_by_email(email)` | 根据邮箱获取 |
| `get_by_api_key(api_key)` | 根据API密钥获取 |
| `use_quota(user_id, amount)` | 使用配额 |
| `reset_quota(user_id, quota)` | 重置配额 |

### VideoRepository

| 方法 | 说明 |
|------|------|
| `get_by_task(task_id)` | 获取任务的所有视频 |
| `get_by_scene(task_id, scene_number)` | 获取指定场景视频 |
| `count_by_task(task_id)` | 统计任务视频数 |

---

## 🎉 总结

### 核心特性

- ✅ **完整的数据模型**: 用户、任务、脚本、视频、统计
- ✅ **仓储层抽象**: 简化数据库操作
- ✅ **关系映射**: 自动处理表关联
- ✅ **事务管理**: 自动提交/回滚
- ✅ **性能优化**: 索引、连接池、预加载

### 使用建议

1. 使用仓储层进行数据操作
2. 使用上下文管理器管理会话
3. 使用模型方法更新状态
4. 避免 N+1 查询问题
5. 定期备份数据库

---

**享受数据持久化带来的便利！** 🚀
