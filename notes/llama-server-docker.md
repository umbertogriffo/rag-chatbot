# llama.cpp Server Setup Guide

This guide covers different ways to run llama.cpp server for the RAG Chatbot.

## Quick Start with Docker Compose (Recommended)

The easiest way to get started is using docker-compose:

```bash
# Start both llama.cpp server and chatbot API
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This automatically:
- Starts llama.cpp server on port 8080
- Starts chatbot API on port 8000
- Mounts `./models` directory for model files
- Configures health checks and dependencies

## Docker with CUDA (GPU Support)

### Prerequisites

- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed
- Docker with GPU support enabled

### Using Pre-built Image

The default `docker-compose.yml` uses CUDA-enabled image:

```yaml
services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    # ... configuration
```

Place your model in `./models/` and update the command:

```yaml
command: >
  --model /models/your-model.gguf
  --host 0.0.0.0
  --port 8080
  --ctx-size 4096
  --n-gpu-layers 99
  --flash-attn
```

### Building Custom CUDA Image

If you need a specific CUDA version:

```bash
# Clone llama.cpp repository
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with specific CUDA version
docker build \
  -t local/llama.cpp:server-cuda \
  --target server \
  --build-arg CUDA_VERSION=12.6.0 \
  -f .devops/cuda.Dockerfile .
```

Then update `docker-compose.yml` to use your local image:

```yaml
services:
  llama-server:
    image: local/llama.cpp:server-cuda
    # ... rest of configuration
```

### Run Standalone CUDA Container

```bash
# Download model first
mkdir -p models
cd models
# Download your model (example)
wget https://huggingface.co/.../model.gguf

# Run server
docker run --gpus all \
  -v $(pwd)/models:/models \
  -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --flash-attn
```

## CPU-Only Setup

### Using Docker Compose Override

For CPU-only deployment:

```bash
# Use the CPU override file
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

The override file (`docker-compose.override.yml`) configures CPU-only settings.

### Standalone CPU Container

```bash
docker run \
  -v $(pwd)/models:/models \
  -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server \
  --model /models/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --threads 4
```

## Native Installation (No Docker)

### Prerequisites

- C++ compiler (GCC, Clang, or MSVC)
- CMake 3.14+
- CUDA Toolkit (optional, for GPU)
- Metal (macOS, automatic)

### Build from Source

```bash
# Clone repository
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build (CPU)
cmake -B build
cmake --build build --config Release

# Build (CUDA)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Build (Metal - macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

### Run Server

```bash
# Run the server
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
```

## Server Configuration

### Essential Parameters

- `--model`: Path to GGUF model file (required)
- `--host`: Bind address (default: 127.0.0.1, use 0.0.0.0 for docker)
- `--port`: Server port (default: 8080)
- `--ctx-size`: Context window size (default: 2048)
- `--n-gpu-layers`: Number of layers to offload to GPU (default: 0, use 99 for all)

### Performance Parameters

- `--threads`: Number of CPU threads (default: auto)
- `--batch-size`: Batch size for prompt processing (default: 2048)
- `--ubatch-size`: Physical batch size (default: 512)
- `--flash-attn`: Enable flash attention (faster, GPU only)
- `--cont-batching`: Enable continuous batching (better throughput)

### Memory Management

- `--cache-type-k`: KV cache type for keys (f16, q8_0, q4_0)
- `--cache-type-v`: KV cache type for values (f16, q8_0, q4_0)
- `--no-mmap`: Disable memory mapping (slower but uses less RAM)

### Example Configurations

#### High Performance (GPU, Large Model)

```bash
llama-server \
  --model ./models/llama-3.1-70b-q4_k_m.gguf \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --flash-attn \
  --cont-batching \
  --batch-size 2048 \
  --ubatch-size 512 \
  --threads 8
```

#### Balanced (GPU, Medium Model)

```bash
llama-server \
  --model ./models/llama-3.1-8b-q5_k_m.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --flash-attn \
  --batch-size 1024 \
  --ubatch-size 256
```

#### Memory Constrained (CPU, Small Model)

```bash
llama-server \
  --model ./models/llama-3.2-1b-q4_k_m.gguf \
  --port 8080 \
  --ctx-size 2048 \
  --threads 4 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --batch-size 512
```

## Model Selection

### Recommended Models

For RAG applications, consider:

1. **Small/Fast** (1-3B parameters):
   - Llama 3.2 1B/3B
   - Phi-3 Mini
   - Gemma 2B

2. **Balanced** (7-8B parameters):
   - Llama 3.1 8B
   - Mistral 7B v0.3
   - Qwen 2.5 7B

3. **High Quality** (70B+ parameters):
   - Llama 3.1 70B
   - Qwen 2.5 72B

### Quantization Guide

- **Q4_K_M**: Good balance, recommended for most use cases
- **Q5_K_M**: Better quality, slightly larger
- **Q8_0**: High quality, large file size
- **Q4_0**: Smallest, fastest, lower quality
- **Q2_K**: Very small, experimental, significant quality loss

Example: `llama-3.1-8b-instruct-q4_k_m.gguf` (4.9GB) vs `llama-3.1-8b-instruct-q8_0.gguf` (8.5GB)

## Health Checks and Monitoring

### Check Server Status

```bash
# Health check endpoint
curl http://localhost:8080/health

# Server info
curl http://localhost:8080/v1/models

# Metrics (if enabled)
curl http://localhost:8080/metrics
```

### Docker Logs

```bash
# View server logs
docker-compose logs -f llama-server

# View API logs
docker-compose logs -f chatbot-api
```

## Troubleshooting

### Server won't start

**Check port availability:**
```bash
lsof -i :8080  # or: netstat -an | grep 8080
```

**Check model file:**
```bash
ls -lh models/
file models/your-model.gguf
```

### Out of Memory

**Solutions:**
- Use smaller model or lower quantization (Q4_K_M instead of Q8_0)
- Reduce context size: `--ctx-size 2048`
- Reduce GPU layers: `--n-gpu-layers 20` (instead of 99)
- Use quantized KV cache: `--cache-type-k q4_0 --cache-type-v q4_0`

### Slow Performance

**GPU not being used:**
```bash
# Check GPU utilization
nvidia-smi

# Ensure n-gpu-layers is set
# Check server logs for "offloaded" message
```

**Optimize parameters:**
- Enable flash attention: `--flash-attn`
- Increase batch size: `--batch-size 2048`
- Enable continuous batching: `--cont-batching`

### Connection Refused

**Docker networking:**
```bash
# Check containers are running
docker-compose ps

# Check container can reach server
docker-compose exec chatbot-api curl http://llama-server:8080/health
```

**Firewall:**
```bash
# Allow port through firewall (Linux)
sudo ufw allow 8080

# Check if port is listening
netstat -tlnp | grep 8080
```

## Advanced Configuration

### Multiple Models (Load Balancing)

Run multiple server instances:

```yaml
# docker-compose.yml
services:
  llama-server-1:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    ports:
      - "8081:8080"
    # ... configuration

  llama-server-2:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    ports:
      - "8082:8080"
    # ... configuration
```

Use a load balancer (nginx, HAProxy) to distribute requests.

### Environment Variables

```bash
# Set via environment variables
export LLAMA_ARG_MODEL=/models/model.gguf
export LLAMA_ARG_CTX_SIZE=4096
export LLAMA_ARG_N_GPU_LAYERS=99

llama-server  # Reads from environment
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp Server
After=network.target

[Service]
Type=simple
User=llama
WorkingDirectory=/opt/llama.cpp
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  --model /opt/models/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable llama-server
sudo systemctl start llama-server
sudo systemctl status llama-server
```

## Production Deployment

### Recommendations

1. **Use Docker**: Easier management and isolation
2. **Health Checks**: Configure proper health check intervals
3. **Resource Limits**: Set CPU/memory limits in docker-compose
4. **Monitoring**: Use Prometheus/Grafana for metrics
5. **Backup**: Keep model files backed up
6. **Updates**: Regularly update llama.cpp server image

### Security

1. **Don't expose server publicly**: Use reverse proxy
2. **Network isolation**: Use Docker networks
3. **Authentication**: Add API key authentication if needed
4. **Rate limiting**: Implement in reverse proxy
5. **HTTPS**: Use TLS for external access

## Resources

- [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Server Documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [OpenAI API Compatibility](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#api-endpoints)
- [GGUF Model Hub](https://huggingface.co/models?library=gguf)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

