# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Backend: install dependencies (China mirror)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Backend: start development server (from backend/)
uvicorn main:app --host 0.0.0.0 --port 8010 --reload

# Backend: initialize/reset the SQLite database
python scripts/init_database.py

# Frontend: start Vue dev server (from frontend/)
npm run dev            # hot-reload, port auto-detected, proxies API to 8010
npm run build          # production build to frontend/dist/ (includes public/config.js)
npm run preview        # preview production build

# Download models (China mirror default: hf-mirror.com)
python scripts/download_model.py --source hf
HF_MIRROR="https://hf-mirror.com" python scripts/download_model.py  # explicit mirror

# Custom backend URL for dev proxy
VITE_BACKEND_URL=http://192.168.1.100:8010 npm run dev

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_auth.py -v
pytest tests/test_video_post_processing.py -v

# Run tests with coverage
pytest tests/ --cov=backend -v

# Run performance benchmarks
pytest tests/test_performance.py -v --benchmark-only

# Load testing
locust -f tests/locustfile.py --host=http://localhost:8010

# Lint / type check (backend)
flake8 backend/
mypy backend/
black --check backend/

# Frontend type check
cd frontend && npx vue-tsc --noEmit
```

## Architecture

This is a **multimodal video creation platform** — natural language input → LLM script generation → diffusion-based video generation → post-processing → final video output.

- **Backend**: Python 3.10+ / FastAPI, served via Uvicorn on port 8000
- **Frontend**: Vue 3 + TypeScript + Vite + Element Plus (white-background enterprise design), served from `frontend/`
- **Production serving**: Backend serves the Vue SPA from `frontend/dist/` at `/`; all non-API routes return `index.html`

### Layered design (backend/)

```
api/          → FastAPI routers (auth.py, tasks.py)
middleware/   → JWT auth dependency injection, performance monitoring
services/     → Business logic (LLM, video generation, post-processing, model loading)
repositories/ → SQLAlchemy data access layer (generic BaseRepository + typed repos)
models/       → SQLAlchemy ORM models (User, Task, TaskStatus, Video, Script)
schemas/      → Pydantic request/response models
utils/        → Cross-cutting: JWT, logging, cache, memory monitor, async helpers
```

- **Auth**: JWT-based (HS256). `auth_middleware.py` provides `get_current_user()`, `get_current_active_user()`, `get_optional_user()` as FastAPI dependencies. Tokens: 60-min access, 7-day refresh.
- **Task processing**: `POST /api/tasks` (requires auth) creates a DB-persisted task and runs `process_video_task()` via FastAPI `BackgroundTasks`. Pipeline: `generate_script()` (LLM → parsed scene list) → `generate_video_from_script()` (per-scene video generation → stitch → post-process). All task state is stored in SQLite via `TaskRepository`.
- **Video generation**: Uses Stable Video Diffusion (SVD-XT) via `diffusers`. Falls back to OpenCV-generated colored frames with text overlays when the model is unavailable.
- **LLM**: ChatGLM3-6B via `transformers`. Falls back to `generate_fallback_script()` (simple sentence splitting) when the model is unavailable.
- **Post-processing pipeline**: filters → transitions → subtitles → audio → optimization → compression. Each step is a separate service class.
- **Model loading**: Singleton `llm_loader` and `video_loader` in `services/model_loader.py`. Models are loaded at app startup (lifespan) and unloaded at shutdown. FP16 + xFormers + attention/VAE slicing for memory optimization.

### Frontend (Vue 3 SPA)

```
frontend/src/
├── router/index.ts       → Vue Router with auth guards
├── stores/auth.ts        → Pinia auth store (JWT, user info, login/register/logout)
├── stores/tasks.ts       → Pinia tasks store (CRUD, polling)
├── api/index.ts          → Axios instance with JWT interceptor + auto-refresh
├── api/auth.ts           → Auth API calls
├── api/tasks.ts          → Task API calls
├── views/                → Page-level components (Home, Login, Register, Tasks, TaskDetail)
├── components/
│   ├── layout/           → AppHeader (white navbar), AppFooter
│   ├── auth/             → LoginForm, RegisterForm
│   ├── tasks/            → TaskCreate, TaskStatus, TaskList, TaskCard
│   ├── video/            → VideoPlayer (custom controls + metadata)
│   └── common/           → LoadingSkeleton
└── styles/               → Design tokens (variables.css) + global overrides
```

- **State management**: Pinia stores with localStorage token persistence
- **Auth flow**: Axios interceptor auto-attaches Bearer token; 401 triggers refresh token attempt, then redirect to `/login`
- **API proxy**: Vite dev server proxies `/api`, `/videos`, `/health` to `localhost:8000`
- **Design**: White background (`#ffffff`), Element Plus with custom theme variables, Google Material-like color system
- **Routes**: `/` (home/creation), `/login`, `/register`, `/tasks` (history, requires auth), `/tasks/:id` (detail + video playback, requires auth)

### Database (SQLite via SQLAlchemy 2.0)

Connection URL: `sqlite:///./video_platform.db` (configured in `config.py`). ORM models in `backend/models/`:
- `models/database.py` — `Base`, engine, `SessionLocal`, `get_db()`, `get_db_context()`, `init_db()`, `get_db_info()`
- `models/user.py` — `User` (username, email, password_hash, api_key, is_active, quota, used_quota)
- `models/task.py` — `Task`, `TaskStatus` enum (PENDING/PROCESSING/COMPLETED/FAILED)
- `models/video.py` — `Video` (task_id FK, scene_number, file_path, file_size, duration)
- `models/script.py` — `Script` (task_id FK, scene_number, description, duration, camera_movement, lighting)

### Configuration

All config in `backend/config.py` as module-level dicts: `LLM_CONFIG`, `VIDEO_CONFIG`, `VIDEO_PROCESSING_CONFIG`, `MEMORY_CONFIG`, `JWT_CONFIG`, `VIDEO_POST_PROCESSING_CONFIG`, `PERFORMANCE_CONFIG`. JWT secret key from `JWT_SECRET_KEY` env var. HF mirror defaults to `https://hf-mirror.com`, override via `HF_MIRROR` env var.

### Key dependencies

`torch`, `transformers`, `diffusers`, `opencv-python`, `moviepy`, `pillow`, `xformers`, `python-jose` (JWT), `bcrypt`+`passlib` (passwords), `sqlalchemy`, `ffmpeg-python`+`pydub` (audio). Frontend: `vue`, `vue-router`, `pinia`, `axios`, `element-plus`.
