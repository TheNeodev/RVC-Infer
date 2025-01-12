import os
from pathlib import Path
import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

BASE_DIR = Path(__file__).resolve().parent.parent


def dl_model(link, model_name, dir_path):
    url = f"{link}{model_name}"
    destination = dir_path / model_name
    dir_path.mkdir(parents=True, exist_ok=True)  # Correct way to ensure directories exist
    
    with requests.get(url, stream=True) as r:  # stream=True for large files
        try:
            r.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {model_name} to {destination}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {model_name}: {e}")
    

if __name__ == "__main__":
    print("Downloading hubert_base.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets/rvc/hubert")
    print("Downloading rmvpe.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets/rvc/rmvpe")
    print("All models downloaded!")
