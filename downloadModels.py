# Add this to your script before loading models
from huggingface_hub import snapshot_download
import os

if not os.path.exists("./yisol/IDM-VTON") or not os.path.exists("./models/IDM-VTON/unet/diffusion_pytorch_model.safetensors"):
    print("Downloading model from Hugging Face Hub...")
    snapshot_download(repo_id="yisol/IDM-VTON", local_dir="yisol/IDM-VTON")

