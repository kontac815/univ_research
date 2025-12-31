import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import wandb

# ====== .env の読み込み ======
# プロジェクト直下にある .env を安全に読み込む
load_dotenv()

# ====== Hugging Face ======
hf_token = os.getenv("HF_TOKENS")
if hf_token:
    login(token=hf_token)
    print("[HF] Logged in")
else:
    print("[HF] No token found in .env")

# # ====== Weights & Biases ======
# wandb_key = os.getenv("WANDB_API_KEY")
# if wandb_key:
#     wandb.login(key=wandb_key)
#     print("[W&B] Logged in")
# else:
#     print("[W&B] No API key found in .env")

def wandb_init_safe(**kwargs):
    """
    - すでに `wandb login` 済み前提で、単に init する
    - 失敗したら W&B を丸ごと無効化して None を返す
    """
    try:
        run = wandb.init(**kwargs)
        print("[W&B] run started:", run.name)
        return run
    except Exception as e:
        print("[WARN] wandb.init failed, disabling W&B:", e)
        os.environ["WANDB_DISABLED"] = "true"
        return None