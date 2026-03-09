# 🗜️ Quantizing ClearChain's BERT Models to INT8

ClearChain uses local BERT models to handle **Security** (detecting prompt injections) and **Routing** (figuring out what your query is about). By default, these models load in full precision (FP32), which can eat up a good chunk of your RAM or VRAM.

**Do note that if you quantize these models, that you're still technically trading off accuracy and/or security for more wiggle room (RAM/VRAM). I haven't
test extensively yet so keep that in mind.**

**Quantizing to INT8** compresses these models so they use about **75% less memory** and run faster, with almost zero loss in accuracy. 

Here is the easiest way to enable INT8 quantization in ClearChain using the industry-standard `bitsandbytes` library.

## Step 1: Install the Prerequisites
To load Hugging Face models in 8-bit, you need two libraries: `accelerate` and `bitsandbytes`. 

Open your terminal and run:
```bash
pip install accelerate bitsandbytes
```
*(Note: `bitsandbytes` works best if you have an NVIDIA GPU. If you are on a CPU-only machine, see the CPU note at the bottom).*

## Step 2: Update `local_models.py`
We need to tell ClearChain's `transformers` pipeline to load the models in 8-bit mode. 

Open the `core/local_models.py` file in your code editor. Look for the `initialize_models` function (around line 20). We are going to add `model_kwargs={"load_in_8bit": True}` to the pipelines.

Update the function so it looks like this:

```python
    def initialize_models(self):
        if config.get("use_local_security") and self.security_pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Local security enforced but transformers library is missing. Stopping to prevent data leakage to API.")
            else:
                logging.info("Loading DistilRoBERTa Security Model in INT8...")
                self.security_pipeline = pipeline(
                    "text-classification", 
                    model=config["local_security_model"], 
                    device_map="auto",
                    model_kwargs={"load_in_8bit": True} # <-- ADDED THIS
                )

        if config.get("use_local_routing") and self.routing_pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Local routing enforced but transformers library is missing. Halting to prevent data leakage to API.")
            else:
                logging.info("Loading MiniLM Routing Model in INT8...")
                self.routing_pipeline = pipeline(
                    "zero-shot-classification", 
                    model=config["local_routing_model"], 
                    device_map="auto",
                    model_kwargs={"load_in_8bit": True} # <-- ADDED THIS
                )
```

## Step 3: Test it out!
Run ClearChain as you normally would:
```bash
python main.py
```
When ClearChain starts up, it will download the models (if it hasn't already) and use `bitsandbytes` to compress them into 8-bit integers on the fly before loading them into memory. You should notice a significantly smaller memory footprint!

---

### 💡 Alternative: CPU-Only Users
If you don't have a GPU, `bitsandbytes` might not work for you. Instead, you can use PyTorch's built-in dynamic quantization. 

Create a new file called `quantize_cpu.py` in your project folder, paste this code, and run it:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Choose the model you want to quantize
model_id = "protectai/deberta-v3-base-prompt-injection-v2"
save_path = "./quantized-security-model"

print(f"Downloading and quantizing {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# 2. Apply PyTorch Dynamic INT8 Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 3. Save the new tiny model locally
tokenizer.save_pretrained(save_path)
quantized_model.save_pretrained(save_path)
print(f"Done! Model saved to {save_path}")
```

Once the script finishes, just open your `config.json` and change `"local_security_model"` to point to your new local folder:
`"local_security_model": "./quantized-security-model"`
