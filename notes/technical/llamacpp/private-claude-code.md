# Private Claude Code

## Pre-requisites

You need to install NVIDIA driver, CUDA and cuDNN to be able to run the models on the GPU.

## Install llama.cpp on Linux

```shell
sudo apt-get update
sudo apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev git-all -y
# git clone https://github.com/ggml-org/llama.cpp
git clone --branch b9503 --depth 1 https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
# CUDA compilation is very memory heavy. If you want to reduce the strain, reduce the number of parallel jobs by changing the `-j` flag. For example, `-j 8` will use 8 parallel jobs.
cmake --build llama.cpp/build --config Release -j 8 --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

For more info read here: https://unsloth.ai/docs/basics/claude-code#install-llama.cpp

## Install llama.cpp on Apple Silicon

```shell
brew install cmake
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

## Download the models

```shell
pip install huggingface_hub hf_transfer
hf download unsloth/Qwen3.6-27B-GGUF \
    --local-dir unsloth/Qwen3.6-27B-GGUF \
    --include "*UD-Q2_K_XL*" # Use "*UD-Q2_K_XL*" for Dynamic 2bit
hf download unsloth/Qwen3.6-27B-GGUF \
    --local-dir unsloth/Qwen3.6-27B-GGUF \
    --include "*mmproj*" # Use "*mmproj*" for Vision models
```

For more info on the model read here: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF

```shell
hf download unsloth/Qwen3.5-9B-GGUF \
    --local-dir unsloth/Qwen3.5-9B-GGUF \
    --include "*UD-Q8_K_XL*" # Use "*UD-Q8_K_XL*" for Dynamic 8bit

hf download unsloth/Qwen3.5-9B-GGUF \
    --local-dir unsloth/Qwen3.5-9B-GGUF \
    --include "*mmproj*" # Use "*mmproj*" for Vision models
```

For more info on the model read here: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF

## Start the Llama-server

```shell
./llama.cpp/llama-server \
    --model unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q2_K_XL.gguf \
    --alias "unsloth/Qwen3.6-27B-GGUF" \
    --mmproj unsloth/Qwen3.6-27B-GGUF/mmproj-BF16.gguf \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072 # change as required
    # --chat-template-kwargs "{\"enable_thinking\": false}"
```

```shell
./llama.cpp/llama-server \
    --model unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q8_K_XL.gguf \
    --alias "unsloth/Qwen3.5-9B-GGUF" \
    --mmproj unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --port 8001 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --ctx-size 131072 # change as required
    # --chat-template-kwargs "{\"enable_thinking\": false}"
```

For more info read here: https://unsloth.ai/docs/basics/claude-code#start-the-llama-server

Tunnel port `8001` over ssh:
```shell
ssh -i ~/.ssh/umberto_dev_bot -L 8001:localhost:8001 umberto@<DEV_MACHINE_IP>
```

Open a browser and access the localhost at port number 8001. You should see the Llama-server UI.

## Install Claude Code and run it locally

```shell
curl -fsSL https://claude.ai/install.sh | bash
```

```
  Version: 2.1.121
  Location: ~/.local/bin/claude
```

Add this to your `.bashrc`/`.zshrc`:

```shell
nano ~/.bashrc
```

```shell
export ANTHROPIC_BASE_URL="http://localhost:8001"
export ANTHROPIC_API_KEY='sk-no-key-required' ## or 'sk-1234'
```

```shell
source ~/.bashrc
```

Navigate to your project folder (mkdir project ; cd project) and run:

```shell
claude --model unsloth/Qwen3.5-9B-GGUF
```

For more info read here:
- https://unsloth.ai/docs/basics/claude-code#claude-code-tutorial
- https://www.reddit.com/r/LocalLLaMA/comments/1s8l1ef/how_to_connect_claude_code_cli_to_a_local/
