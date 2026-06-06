## Live Model Switching

`llama.cpp server` now ships with router mode, which lets you dynamically load, unload, and switch between multiple models without restarting.

Run the server:
```shell
./llama.cpp/llama-server \
    --models-dir ./models \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072
```

Run the server on Docker with CUDA:
```shell
# Assuming one has the nvidia-container-toolkit (https://github.com/NVIDIA/nvidia-container-toolkit) properly installed on Linux.
# Download the model from https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF to the /model folder.
docker run --gpus all \
  -v $(pwd)/models:/models \
  -p 8000:8000 \
  local/llama.cpp:server-cuda \
  --models-dir ./models \
  --port 8000 \
  --host 0.0.0.0 \
  --ctx_size 4096 \
  --predict 2048 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --batch-size 1024 \
  --ubatch-size 512 \
  --threads 16 \
  --temp 0.6
```

To manually load a model:

```shell
curl -X POST http://localhost:8080/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.2-1B-Instruct-Q5_K_M"}'
```

To manually unload a model to free VRAM:

```shell
curl -X POST http://localhost:8080/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.2-1B-Instruct-Q5_K_M"}'
```
