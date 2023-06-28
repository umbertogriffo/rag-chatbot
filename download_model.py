from pathlib import Path

import requests
from tqdm import tqdm

local_path = "./models/gpt4all-lora-quantized-ggml.bin"
Path(local_path).parent.mkdir(parents=True, exist_ok=True)

url = "https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin"

# send a GET request to the URL to download the file.
response = requests.get(url, stream=True)

# open the file in binary mode and write the contents of the response
# to it in chunks.
with open(local_path, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size=8192)):
        if chunk:
            f.write(chunk)
