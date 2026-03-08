# 📘 ClearChain Model Configuration Guide

With the new Hybrid LLM architecture, ClearChain allows you to mix and match cloud and local models seamlessly. You can use Gemini for text generation while using a local Ollama model for embeddings, or run everything 100% locally.

## Part 1: Connecting Ollama (Mix-and-Match)

To connect Ollama, you just need to modify your `config.json` file (located in your user data directory, usually `~/.local/share/ClearChain/config.json` on Linux/Mac or `%LOCALAPPDATA%\cove_verifier\ClearChain\config.json` on Windows).

### Scenario A: 100% Local (Ollama for Everything)
If you want to run completely offline without needing an API key, set both providers to `ollama`. 

```json
{
    "text_provider": "ollama",
    "embedding_provider": "ollama",
    "primary_model": "llama3:latest",
    "json_model": "llama3:latest",
    "embedding_model": "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434/api"
}
```
*Note: Make sure you have pulled these models in your terminal first using `ollama pull llama3` and `ollama pull nomic-embed-text`.*

### Scenario B: Mix-and-Match (e.g., Gemini for Text, Ollama for Embeddings)
If you want the reasoning power of Gemini but want to keep your vector database embeddings local (to save costs or ensure privacy of the database structure):

```json
{
    "text_provider": "gemini",
    "embedding_provider": "ollama",
    "primary_model": "gemini-3.1-pro-preview",
    "json_model": "gemini-3.1-pro-preview",
    "embedding_model": "nomic-embed-text"
}
```
*Because `text_provider` is a cloud model, ClearChain will still prompt you for your Gemini API key, but it will route all embedding requests to your local Ollama instance.*

---

## Part 2: Connecting Your *Own* Custom Model

If you have trained or fine-tuned your own model, there are two main ways to connect it to ClearChain:

### Method 1: Using Ollama (For `.gguf` or `.safetensors` files)
If you have a local model file, the easiest way is to wrap it in an Ollama Modelfile.

1. Create a file named `Modelfile` in the same folder as your model:
   ```dockerfile
   FROM ./my-custom-model-v1.gguf
   # You can also add custom system prompts or parameters here
   PARAMETER temperature 0.3
   ```
2. Build it into Ollama via your terminal:
   ```bash
   ollama create my-clearchain-model -f Modelfile
   ```
3. Update your `config.json` to use your new model:
   ```json
   "text_provider": "ollama",
   "primary_model": "my-clearchain-model"
   ```

### Method 2: Using a Custom API (LM Studio, vLLM, or Custom Python Backend)
If you are hosting your model using a tool like **LM Studio**, **Oobabooga**, or **vLLM**, these tools provide an "OpenAI-Compatible API". 

With the updated files below, you can simply set your provider to `openai` and point the URL to your local server!

```json
{
    "text_provider": "openai",
    "primary_model": "my-custom-model-name",
    "openai_base_url": "http://localhost:1234/v1" 
}
```
*(Port `1234` is the default for LM Studio. Port `8000` is default for vLLM).*