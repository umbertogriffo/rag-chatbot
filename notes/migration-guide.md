# Migration Guide: From llama-cpp-python to OpenAI SDK

## Overview

This guide helps you migrate from the previous llama-cpp-python integration to the new OpenAI SDK-based architecture that communicates with llama.cpp's OpenAI-compatible server.

## What Changed

### Architecture

**Before:**
- Direct integration with llama-cpp-python library
- Model loaded in the same process as the API
- Required hardware-specific compilation (CUDA/Metal)
- Threading locks for thread safety
- Single-process architecture

**After:**
- HTTP communication with llama.cpp server via OpenAI SDK
- Model runs in separate llama.cpp server process
- No compilation needed in API service
- Native async/await support
- Scalable multi-process architecture

### Benefits

1. **Simplified Installation**: No CMAKE_ARGS or hardware-specific builds
2. **Better Scalability**: Multiple API workers can share one model server
3. **Easier Deployment**: Docker-based, platform-independent
4. **Faster Development**: No native extension compilation
5. **Server Independence**: Can upgrade llama.cpp server without changing Python code
6. **Better Concurrency**: True async HTTP calls, no GIL limitations
7. **Operational Flexibility**: Server and API can be scaled independently

## Migration Steps

### 1. Update Dependencies

```bash
# Pull latest code
git pull origin main

# Clean old environment
make clean

# Install new dependencies (includes OpenAI SDK)
make setup
```

The `openai` package is now a dependency instead of `llama-cpp-python`.

### 2. Set Up llama.cpp Server

You have two options:

#### Option A: Using Docker Compose (Recommended)

```bash
# Start both llama.cpp server and API
docker-compose up -d
```

This starts:
- `llama-server` on port 8080
- `chatbot-api` on port 8000

#### Option B: Manual Server Setup

Download and run llama.cpp server separately:

```bash
# Download llama.cpp server binary or build from source
# See: https://github.com/ggml-org/llama.cpp

# Start the server with your model
llama-server \
  --model ./models/your-model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
```

For detailed instructions, see `notes/llama-server-docker.md`.

### 3. Update Configuration

Update your `.env` file with new server settings:

```env
# New: llama.cpp server URL
LLAMA_SERVER_BASE_URL=http://localhost:8080

# Changed: MODEL now refers to model loaded on server
MODEL=llama-3.1

# New: Timeout for server requests
LLAMA_SERVER_TIMEOUT=300

# Unchanged: Other settings remain the same
MAX_NEW_TOKENS=512
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Model Management

**Before:**
- Models auto-downloaded by Python client
- Models loaded in Python process

**After:**
- Models managed server-side
- Download models manually to `./models/` directory
- Server loads model at startup
- API connects to server via HTTP

Example workflow:

```bash
# 1. Download your model (example)
wget https://huggingface.co/.../model.gguf -O models/model.gguf

# 2. Start server with model (if not using docker-compose)
llama-server --model ./models/model.gguf --port 8080

# 3. Start API (it will connect to server)
sh start.sh
```

### 5. Code Changes (If You Extended the Client)

If you have custom code that uses the LLM client:

**Before:**
```python
from bot.client.lama_cpp_client import LamaCppClient

# Client was initialized with model_folder
client = LamaCppClient(model_folder=Path("./models"), model_settings=settings)

# Methods returned llama-cpp-python types
stream = client.start_answer_iterator_streamer(prompt)
for output in stream:
    token = output["choices"][0]["delta"].get("content", "")
```

**After:**
```python
from bot.client.openai_client import OpenAIClient

# Client connects to server via URL
client = OpenAIClient(
    base_url="http://localhost:8080",
    model_name="llama-3.1",
    model_settings=settings,
    timeout=300
)

# Methods return OpenAI SDK types
stream = client.start_answer_iterator_streamer(prompt)
for output in stream:
    token = client.parse_token(output)  # Use parse_token helper
```

**Key Changes:**
- `LamaCppClient` → `OpenAIClient`
- `model_folder` → `base_url`
- Response types are from OpenAI SDK
- Use `parse_token()` method for streaming tokens
- No `close()` method needed (HTTP client handles cleanup)

### 6. Testing

Update your tests to connect to a running server:

```python
# conftest.py
@pytest.fixture
def openai_client():
    return OpenAIClient(
        base_url=os.getenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080"),
        model_name="llama-3.1",
        model_settings=model_settings,
        timeout=300,
    )
```

Run tests (requires llama.cpp server running):

```bash
# Start server first
docker-compose up -d llama-server

# Run tests
make test
```

## Breaking Changes

### Removed

1. **`version/llama_cpp` file**: No longer needed
2. **`make install_cuda` / `make install_metal`**: Use `make setup` instead
3. **`LamaCppClient.close()` method**: HTTP client manages connections
4. **`LamaCppClient._lock` threading lock**: Server handles concurrency
5. **Model auto-download**: Models are server-side managed

### Changed

1. **Client initialization**: Now requires `base_url` instead of `model_folder`
2. **Streaming response types**: OpenAI SDK types instead of llama-cpp-python types
3. **Configuration**: New `LLAMA_SERVER_BASE_URL` and `LLAMA_SERVER_TIMEOUT` settings
4. **Dependency injection**: `LamaCppClientDep` → `LLMClientDep`

### Added

1. **Docker support**: `docker-compose.yml` for integrated deployment
2. **Server health checks**: Connection validation on client init
3. **Better async support**: Native async/await throughout
4. **OpenAI SDK dependency**: Replaces llama-cpp-python

## Troubleshooting

### "Cannot connect to llama.cpp server"

**Cause**: Server not running or wrong URL

**Solution**:
```bash
# Check if server is running
curl http://localhost:8080/health

# Start server if not running
docker-compose up -d llama-server

# Or start manually
llama-server --model ./models/model.gguf --port 8080
```

### "Model not found" error

**Cause**: Model file doesn't exist on server

**Solution**:
```bash
# Ensure model exists in models directory
ls -lh models/

# Download model if missing
# (Download from HuggingFace or other source)

# Restart server with correct model path
```

### Tests failing with connection errors

**Cause**: llama.cpp server not running during tests

**Solution**:
```bash
# Start server before running tests
docker-compose up -d llama-server

# Set correct server URL for tests
export LLAMA_SERVER_BASE_URL=http://localhost:8080

# Run tests
make test
```

### Slow first request

**Cause**: Server warming up / loading model

**Solution**: This is expected. First request may take longer as the model initializes. Subsequent requests will be faster.

### Out of memory errors

**Cause**: Model too large for available GPU/RAM

**Solution**:
- Use a smaller quantized model (e.g., Q4_K_M instead of Q8_0)
- Reduce `--ctx-size` parameter
- Reduce `--n-gpu-layers` if using GPU

## Rollback Plan

If you need to rollback to the old implementation:

```bash
# Checkout previous version
git checkout <previous-commit>

# Reinstall with hardware-specific build
make install_cuda  # or make install_metal

# Restart services
sh start.sh
```

## Support

For issues or questions:
- Check existing GitHub issues
- Review `notes/llama-server-docker.md` for server setup
- Consult llama.cpp documentation: https://github.com/ggml-org/llama.cpp

## Timeline

This migration is a **breaking change** with no backward compatibility. All users must migrate to the new architecture.

Recommended timeline:
1. Week 1: Test in development environment
2. Week 2: Update staging/QA environment
3. Week 3: Deploy to production

## Checklist

- [ ] Pull latest code
- [ ] Run `make clean && make setup`
- [ ] Update `.env` with server URL
- [ ] Download model to `./models/` directory
- [ ] Start llama.cpp server (or use docker-compose)
- [ ] Update custom code if any
- [ ] Run tests to verify
- [ ] Deploy to production

## Additional Resources

- [llama.cpp Server Documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [OpenAI Python SDK Documentation](https://github.com/openai/openai-python)
- [Docker Installation Guide](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) (for GPU support)
