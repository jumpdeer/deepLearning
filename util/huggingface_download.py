import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfFolder

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# 数据集 ID
DATASET_ID = "svjack/pokemon-blip-captions-en-zh"

# 2. 你想把文件保存在本地的哪个主文件夹下
SAVE_DIRECTORY = "../Items/DiffusionInWritting/offline_assets"

# 3. 使用国内镜像节点
# HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


## 使用这种方法需要开启clash代理
os.environ["https_proxy"] = 'http://localhost:7890'
os.environ["http_proxy"] = 'http://localhost:7890'
os.environ["all_proxy"] = 'socks5://localhost:7890'

# --- 执行下载 ---

def download_asset(repo_id, repo_type, local_dir_name):
    """下载一个模型或数据集的通用函数"""
    local_path = os.path.join(SAVE_DIRECTORY, local_dir_name)
    print(f"--- 开始下载 {repo_id} ---")
    print(f"保存至: {local_path}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # 设置为 False，会直接复制文件，方便打包上传
            resume_download=True,  # 开启断点续传
            # endpoint=HF_MIRROR_ENDPOINT,  # 使用镜像
            # 如果你有 Hugging Face 的 token，可以取消注释下面这行
            # token="hf_YOUR_TOKEN_HERE"
        )
        print(f"--- ✅ {repo_id} 下载成功 ---\n")
    except Exception as e:
        print(f"--- ❌ {repo_id} 下载失败 ---")
        print(f"错误信息: {e}\n")


if __name__ == "__main__":
    print("开始下载离线资源...")
    # 创建主保存目录
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    # 下载 CLIP 模型
    download_asset(repo_id=CLIP_MODEL_ID, repo_type="model", local_dir_name="openai-clip-vit-base-patch32")

    # 下载宝可梦数据集
    download_asset(repo_id=DATASET_ID, repo_type="dataset", local_dir_name="pokemon-blip-captions-en-zh")

    print("所有任务已完成。")
