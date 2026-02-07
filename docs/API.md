# API 文档

## 基础信息

- 基础URL: `http://localhost:8000`
- 所有请求和响应均为JSON格式

## 接口列表

### 1. 健康检查

**接口**: `GET /health`

**描述**: 检查服务运行状态

**响应**:
```json
{
  "status": "ok",
  "message": "服务运行正常"
}
```

---

### 2. 创建视频生成任务

**接口**: `POST /api/tasks/`

**描述**: 提交视频创作指令，创建新任务

**请求体**:
```json
{
  "prompt": "制作一段关于森林探险的短视频，包含河流和小动物"
}
```

**响应**:
```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "prompt": "制作一段关于森林探险的短视频，包含河流和小动物",
  "result": null,
  "created_at": "2024-01-01T12:00:00",
  "error": null
}
```

**状态说明**:
- `pending`: 等待处理
- `processing`: 生成中
- `completed`: 已完成
- `failed`: 失败

---

### 3. 查询任务状态

**接口**: `GET /api/tasks/{task_id}`

**描述**: 获取指定任务的详细信息

**路径参数**:
- `task_id`: 任务ID

**响应**:
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "prompt": "制作一段关于森林探险的短视频",
  "result": "videos/uuid_final.mp4",
  "created_at": "2024-01-01T12:00:00",
  "error": null
}
```

---

### 4. 获取任务列表

**接口**: `GET /api/tasks/`

**描述**: 获取所有任务列表（分页）

**查询参数**:
- `skip`: 跳过数量（默认0）
- `limit`: 返回数量（默认10）

**响应**:
```json
{
  "tasks": [
    {
      "task_id": "uuid-string",
      "status": "completed",
      "prompt": "...",
      "result": "videos/...",
      "created_at": "2024-01-01T12:00:00"
    }
  ],
  "total": 100
}
```

---

### 5. 删除任务

**接口**: `DELETE /api/tasks/{task_id}`

**描述**: 删除指定任务

**路径参数**:
- `task_id`: 任务ID

**响应**:
```json
{
  "message": "任务已删除"
}
```

---

## 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 404 | 资源不存在 |
| 422 | 请求参数错误 |
| 500 | 服务器内部错误 |

## 使用示例

### Python
```python
import requests

# 创建任务
response = requests.post(
    "http://localhost:8000/api/tasks/",
    json={"prompt": "制作一段关于森林探险的短视频"}
)
task_id = response.json()["task_id"]

# 查询任务状态
response = requests.get(f"http://localhost:8000/api/tasks/{task_id}")
print(response.json())
```

### JavaScript
```javascript
// 创建任务
const response = await fetch('http://localhost:8000/api/tasks/', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({prompt: '制作一段关于森林探险的短视频'})
});
const data = await response.json();
console.log(data.task_id);
```
