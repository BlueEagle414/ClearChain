# ClearChain Installation & Requirements

ClearChain relies on a modern Python stack to handle asynchronous operations, terminal UI rendering, and local machine learning models. 

## System Requirements
* **Python:** Version 3.10 or higher.
* **OS:** Windows, macOS, or Linux (Includes headless server support).
* **Memory:** At least 8GB RAM recommended if running local routing and security models.

## Core Dependencies

To install the required packages, simply run:
`pip install -r requirements.txt`

ClearChain leverages the following major libraries:
* **UI & Core:** `textual` for the terminal interface, `appdirs` for secure config management.
* **Database:** `lancedb` and `pyarrow` for asynchronous, high-performance vector storage.
* **Local Machine Learning:** `transformers` and `torch` to run the local DistilRoBERTa/DeBERTa and MiniLM models.
* **API Providers:** `openai`, `google-genai`, and `httpx` (for Ollama) to communicate with your chosen LLM.

## Setting Up Local Models (Optional but Recommended)

By default, ClearChain enforces local security and routing to prevent sensitive data leakage. 
1. When you first run `main.py`, the `transformers` library will automatically download the DeBERTa and MiniLM models from HuggingFace. 
2. This may take a few minutes depending on your internet connection. 
3. If you wish to disable this and rely entirely on API models, change `"use_local_security": false` and `"use_local_routing": false` in your `config.json`.

## Using Ollama (Fully Local LLM)
If you prefer a 100% offline CoVe pipeline:
1. Install [Ollama](https://ollama.com/).
2. Pull your desired models (e.g., `ollama pull llama3` and `ollama pull nomic-embed-text`).
3. Update your ClearChain configuration to use `ollama` as the `llm_provider`.