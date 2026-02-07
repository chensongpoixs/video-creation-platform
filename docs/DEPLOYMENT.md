# 部署指南

## 环境要求

### 硬件要求
- CPU: Intel Core i7 或更高
- GPU: NVIDIA RTX 3090/4090 (推荐，24GB显存)
- 内存: 32GB+ RAM
- 存储: 100GB+ SSD

### 软件要求
- 操作系统: Ubuntu 20.04+ / Windows 10+
- Python: 3.10+
- CUDA: 11.7+
- Docker: 20.10+ (可选)

---

## 本地部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd video-creation-platform
```

### 2. 配置环境
```bash
# Linux/Mac
chmod +x scripts/setup.sh
./scripts/setup.sh

# Windows
python -m venv venv
venv\Scripts\activate
cd backend
pip install -r requirements.txt
```

### 3. 配置模型
下载模型文件并放置到指定目录：
```
backend/
├── models/
│   ├── llm/          # 大语言模型
│   └── video/        # 视频生成模型
```

### 4. 启动服务
```bash
# Linux/Mac
./scripts/run.sh

# Windows
cd backend
python main.py
```

### 5. 访问系统
打开浏览器访问: `http://localhost:8000`

---

## Docker部署

### 1. 构建镜像
```bash
docker-compose build
```

### 2. 启动服务
```bash
docker-compose up -d
```

### 3. 查看日志
```bash
docker-compose logs -f
```

### 4. 停止服务
```bash
docker-compose down
```

---

## 生产环境部署

### 1. 使用Nginx反向代理
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /videos/ {
        alias /path/to/videos/;
    }
}
```

### 2. 使用Supervisor管理进程
```ini
[program:video-platform]
command=/path/to/venv/bin/python main.py
directory=/path/to/backend
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/video-platform.err.log
stdout_logfile=/var/log/video-platform.out.log
```

### 3. 配置HTTPS
```bash
certbot --nginx -d your-domain.com
```

---

## 性能优化

### 1. GPU优化
- 使用FP16半精度推理
- 启用CUDA内存优化
- 配置显存分配策略

### 2. 并发优化
- 调整`MAX_CONCURRENT_TASKS`参数
- 使用任务队列管理
- 配置合理的超时时间

### 3. 缓存优化
- 启用模型缓存
- 使用Redis缓存任务状态
- 配置CDN加速视频访问

---

## 故障排查

### 问题1: CUDA不可用
```bash
# 检查CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题2: 显存不足
- 降低视频分辨率
- 减少并发任务数
- 使用模型量化

### 问题3: 服务无法启动
```bash
# 查看日志
tail -f backend/logs/app.log

# 检查端口占用
netstat -tuln | grep 8000
```
