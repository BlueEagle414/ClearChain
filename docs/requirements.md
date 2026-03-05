# ClearChain Installation & Requirements

ClearChain relies on a modern Python stack to handle asynchronous operations, terminal UI rendering, and local machine learning models. While it is designed to be a more streamlined, secure alternative to sprawling frameworks like LangChain, it still requires a few robust libraries to run its verification pipeline effectively.

## System Requirements
* **Python:** Version 3.10 or higher.
* **OS:** Windows, macOS, or Linux (Includes headless server support).
* **Memory:** At least 8GB RAM recommended if running local routing and security models.

## Core Dependencies

To install the required packages, simply run:
`pip install -r requirements.txt`

ClearChain leverages the following major libraries:

* **UI & Core:** `textual` for the terminal interface, `appdirs` for secure config management, and `keyring` for safely storing API credentials in your OS manager.
* **Database:** `lancedb` and `pyarrow` for asynchronous, high-performance vector storage.
* **Data Processing:** `langchain-text-splitters` for robust document chunking, giving you the integrity of LangChain without the heavy framework bloat.
* **Local Machine Learning:** `transformers` and `torch` to run the local **deberta-v3-base-prompt-injection-v2** and **MiniLM** models.
* **API Providers:** `openai`, `google-genai`, and `httpx` (for Ollama) to communicate with your chosen LLM.

## ⚠️ Important Note on PyTorch Installation

If you are running ClearChain on a machine with a dedicated NVIDIA or AMD GPU, do not simply rely on `requirements.txt` to install `PyTorch`. Default pip installations will frequently default to the CPU-only version, which will cause your local security and routing models to run sluggishly.

Instead, please visit the official `PyTorch` website to generate the correct installation command for your specific CUDA or ROCm version before installing the rest of the requirements.

## Setting Up Local Models (Optional but Recommended)

By default, ClearChain enforces local security and routing to prevent sensitive data leakage.

1. When you first run **main.py**, the `transformers` library will automatically download the **deberta-v3-base-prompt-injection-v2** and **MiniLM** models from HuggingFace.
2. This may take a few minutes depending on your internet connection.
3. If you wish to disable this and rely entirely on API models, change "use_local_security": false and "use_local_routing": false in your **config.json**.

## Using Ollama (Fully Local LLM)

If you prefer a 100% offline CoVe pipeline:

1. Install Ollama.
2. Pull your desired models (e.g., ollama pull llama3 and ollama pull nomic-embed-text).
3. Update your ClearChain configuration to use ollama as the llm_provider.
