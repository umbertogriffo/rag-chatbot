# llama.cpp Server with Docker and CUDA

Build the CUDA image specifying the version:

```shell
git clone git@github.com:ggml-org/llama.cpp.git
cd llama.cpp
docker build -t local/llama.cpp:server-cuda --target server --build-arg CUDA_VERSION=12.1.0 -f .devops/cuda.Dockerfile .
```

Run the server:
```shell
# Assuming one has the nvidia-container-toolkit (https://github.com/NVIDIA/nvidia-container-toolkit) properly installed on Linux.
# Download the model from https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF to the /model folder.
docker run --gpus all \
  -v $(pwd)/models:/models \
  -p 8000:8000 \
  local/llama.cpp:server-cuda \
  -m /models/DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf \
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

The server will be available at `http://localhost:8000`.

## References
- [Docker With CUDA](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md#docker-with-cuda)
- [LLaMA.cpp HTTP Server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Self-host LLMs in production with llama.cpp llama-server](https://docs.servicestack.net/ai-server/llama-server#enter-llama-server-the-production-workhorse)
