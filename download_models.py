import os


hubert_path = "./assets/rvc/hubert/hubert_base.pt"
rmvpe_path = "./assets/rvc/rmvpe/rmvpe.pt"

os.system(f"wget https://huggingface.co/theNeofr/rvc-base/resolve/main/hubert_base.pt -O {hubert_path}")
os.system(f"wget https://huggingface.co/theNeofr/rvc-base/resolve/main/rmvpe.pt -O {rmvpe_path}")
print("success installing hubert and rmvpe...")
