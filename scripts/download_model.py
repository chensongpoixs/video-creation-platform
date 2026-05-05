#!/usr/bin/env python3
"""
项目模型下载脚本

下载本项目实际使用的两个模型（仅下载权重/配置/分词器，不下载整个仓库）：
  1. ChatGLM3-6B（LLM 剧本生成）→ models/chatglm3-6b
  2. Stable Video Diffusion XT（视频生成）→ models/svd-xt

来源选择：
  --source hf    Hugging Face（默认，自动使用 HF_MIRROR 镜像）
  --source ms    ModelScope（国内更快，自动过滤非模型文件）

用法：
  python scripts/download_model.py                  # 下载全部两个模型（HF 镜像）
  python scripts/download_model.py --model llm      # 只下载 ChatGLM3-6B
  python scripts/download_model.py --model video    # 只下载 SVD-XT
  python scripts/download_model.py --source ms      # 从 ModelScope 下载全部
  HF_MIRROR="https://hf-mirror.com" python scripts/download_model.py
"""

import os
import sys
import argparse
import time
from pathlib import Path

# ============================================================
# 镜像配置
# ============================================================
HF_MIRROR = os.getenv("HF_MIRROR", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)

# ============================================================
# 项目使用的两个模型（与 backend/config.py 保持一致）
# ============================================================
PROJECT_MODELS = {
    "llm": {
        "name": "ChatGLM3-6B（LLM 剧本生成）",
        "repo_id": "THUDM/chatglm3-6b",
        "modelscope_id": "ZhipuAI/chatglm3-6b",
        "output_dir": str(Path(__file__).resolve().parent.parent / "backend" / "models" / "chatglm3-6b"),
        "loader": "transformers",  # 使用 AutoModel + AutoTokenizer
    },
    "video": {
        "name": "Stable Video Diffusion XT（视频生成）",
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "modelscope_id": "AI-ModelScope/stable-video-diffusion-img2vid-xt",
        "output_dir": str(Path(__file__).resolve().parent.parent / "backend" / "models" / "svd-xt"),
        "loader": "diffusers",  # 使用 StableVideoDiffusionPipeline
    },
}

# ModelScope snapshot_download 排除的非模型文件（避免下载整个仓库）
MODELSCOPE_IGNORE_PATTERNS = [
    "*.md", "*.txt", "*.py", "*.ipynb", "*.png", "*.jpg", "*.gif",
    ".gitattributes", ".gitignore", "LICENSE*", "assets/*", "examples/*",
    "docs/*", "tests/*", "*.safetensors.md", "*.msgpack",
]


def download_from_huggingface(model_key: str) -> bool:
    """从 Hugging Face 下载指定模型（仅下载权重/配置/分词器，不下载整个仓库）

    HuggingFace 的 from_pretrained() 本身只下载模型必需的权重、配置和分词器文件，
    不会下载 README、示例脚本等非必需文件，因此无需额外过滤。
    """
    cfg = PROJECT_MODELS[model_key]
    repo_id = cfg["repo_id"]
    output_dir = cfg["output_dir"]
    loader = cfg["loader"]

    print(f"\n{'='*60}")
    print(f"下载: {cfg['name']}")
    print(f"仓库: {repo_id}")
    print(f"保存: {output_dir}")
    print(f"镜像: {HF_MIRROR}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        if loader == "transformers":
            from transformers import AutoModel, AutoTokenizer

            print("\n[1/2] 下载 tokenizer...")
            t0 = time.time()
            AutoTokenizer.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=output_dir,
            )
            print(f"      ✅ Tokenizer 下载完成 ({time.time() - t0:.0f}s)")

            print("\n[2/2] 下载模型权重 + 配置...")
            t0 = time.time()
            AutoModel.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=output_dir,
            )
            print(f"      ✅ 模型下载完成 ({time.time() - t0:.0f}s)")

        elif loader == "diffusers":
            from diffusers import StableVideoDiffusionPipeline
            import torch

            print("\n下载 Stable Video Diffusion Pipeline（权重 + 配置 + scheduler + VAE）...")
            t0 = time.time()
            StableVideoDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float32,
                cache_dir=output_dir,
            )
            print(f"      ✅ 视频模型下载完成 ({time.time() - t0:.0f}s)")

        print(f"\n✅ {cfg['name']} 下载成功")
        return True

    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        if loader == "transformers":
            print("   安装: pip install transformers")
        elif loader == "diffusers":
            print("   安装: pip install diffusers")
        return False
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def download_from_modelscope(model_key: str) -> bool:
    """从 ModelScope 下载指定模型（过滤非模型文件，避免下载整个仓库）"""
    cfg = PROJECT_MODELS[model_key]
    ms_id = cfg["modelscope_id"]
    output_dir = cfg["output_dir"]

    print(f"\n{'='*60}")
    print(f"下载: {cfg['name']}")
    print(f"ModelScope ID: {ms_id}")
    print(f"保存: {output_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        from modelscope import snapshot_download

        print("\n从 ModelScope 下载（过滤 README/示例/测试等非模型文件）...")
        t0 = time.time()

        snapshot_download(
            ms_id,
            cache_dir=output_dir,
            ignore_file_pattern=MODELSCOPE_IGNORE_PATTERNS,
        )

        elapsed = time.time() - t0
        print(f"      ✅ 下载完成 ({elapsed:.0f}s)")
        print(f"\n✅ {cfg['name']} 下载成功")
        return True

    except ImportError:
        print("\n❌ modelscope 未安装，请运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="下载视频创作平台使用的模型（仅下载权重/配置，不下载整个仓库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_model.py                  # 下载全部两个模型
  python scripts/download_model.py --model llm      # 只下载 ChatGLM3-6B
  python scripts/download_model.py --model video    # 只下载 SVD-XT
  python scripts/download_model.py --source ms      # 从 ModelScope 下载全部
  python scripts/download_model.py --source ms --model llm
        """,
    )
    parser.add_argument(
        "--source", choices=["hf", "ms"], default="hf",
        help="下载源: hf=HuggingFace(默认, 自动镜像), ms=ModelScope",
    )
    parser.add_argument(
        "--model", choices=["all", "llm", "video"], default="all",
        help="下载的模型: all(全部), llm(ChatGLM3-6B), video(SVD-XT)",
    )

    args = parser.parse_args()

    # 确定要下载的模型列表
    if args.model == "all":
        keys = ["llm", "video"]
    else:
        keys = [args.model]

    # 选择下载函数
    download_fn = download_from_huggingface if args.source == "hf" else download_from_modelscope

    # 逐个下载
    success = 0
    failed = 0

    for key in keys:
        if download_fn(key):
            success += 1
        else:
            failed += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"下载汇总: 成功 {success}, 失败 {failed}")
    if failed > 0:
        print("提示: 失败的模型可切换下载源重试，如 --source ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
