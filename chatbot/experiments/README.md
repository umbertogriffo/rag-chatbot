# Experiments

This directory contains experimental code and proof-of-concept scripts.

## Note on Legacy Code

The `exp_lama_cpp` directory contains experiments that were written for the old `llama-cpp-python` direct integration.

**Migration Status:**
- These experiments use the deprecated `LamaCppClient` class
- They are kept for reference purposes
- To use these with the new architecture, you would need to:
  1. Replace `LamaCppClient` with `OpenAIClient`
  2. Update initialization to use server URL instead of model folder
  3. Adjust response parsing as needed

**Recommendation:**
- For new experiments, use `OpenAIClient` from `chatbot/bot/client/openai_client.py`
- See `tests/bot/client/test_openai_client.py` for examples of the new API

## Active Experiments

- `check_embedders_and_ranking.py`: Embedding model comparisons
- `explore_memory.py`: Memory/vector store experiments
- `exp_pdf_parsing/`: PDF parsing with Docling

## Running Experiments

Most experiments are standalone scripts that can be run directly:

```bash
cd chatbot/experiments
python explore_memory.py
```

Ensure you have:
1. Set up your environment (`make setup`)
2. Started the llama.cpp server (if needed for LLM experiments)
3. Configured `.env` file properly
