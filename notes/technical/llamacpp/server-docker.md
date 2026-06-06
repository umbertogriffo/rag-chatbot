# llama.cpp Server with Docker and CUDA

## Prerequisites

- Docker installed on your machine.
- NVIDIA GPU with CUDA support (if you want to use the CUDA version).
- [NVIDIA Container Toolkit installed](https://github.com/NVIDIA/nvidia-container-toolkit) (for CUDA support).

## Installing NVIDIA Container Toolkit

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

Once configured, test GPU access:

```shell
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

## Building and Running the Server

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
