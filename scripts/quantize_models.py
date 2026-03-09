import os
import sys
import subprocess  # nosec B404
import platform
import shutil

# Ensure the library is actually installed in this Conda environment (or using Python .env)
try:
    import optimum
except ImportError:
    print("Error: The 'optimum' library is not installed in this environment.")
    print("Please run: pip install optimum[onnxruntime]")
    sys.exit(1)

def get_optimum_cli_path():
    """Finds the absolute path to optimum-cli inside the current Python environment."""
    bin_dir = os.path.dirname(sys.executable)

    if os.name == 'nt':  # Windows
        cli_path = os.path.join(bin_dir, "Scripts", "optimum-cli.exe")
        if not os.path.exists(cli_path):
            cli_path = os.path.join(bin_dir, "optimum-cli.exe")
    else:  # Linux / Mac (Conda/venv)
        cli_path = os.path.join(bin_dir, "optimum-cli")

    if os.path.exists(cli_path):
        return cli_path

    return "optimum-cli"  # Fallback to PATH if not found locally

def run_quantization():
    # Define the models from your config
    models = {
        "security": {
            "hf_id": "protectai/deberta-v3-base-prompt-injection-v2",
            "output_dir": "local_onnx_models/security_model_int8"
        },
        "routing": {
            "hf_id": "cross-encoder/nli-MiniLM2-L6-H768",
            "output_dir": "local_onnx_models/routing_model_int8"
        }
    }

    os.makedirs("local_onnx_models", exist_ok=True)
    cli_command = get_optimum_cli_path()

    arch = platform.machine().lower()
    if 'arm' in arch or 'aarch64' in arch:
        quant_flag = "--arm64"
    else:
        quant_flag = "--avx2"

    for name, data in models.items():
        print(f"\n{'='*50}")
        print(f"[{name.upper()}] Starting ONNX Export and INT8 Quantization")
        print(f"{'='*50}")

        temp_dir = data["output_dir"] + "_temp"

        # Step 1: Export (Removed --optimize O2 to prevent shape inference crashes)
        print(f"\n>>> Step 1/2: Exporting Graph...")
        export_cmd = [
            cli_command, "export", "onnx",
            "--model", data["hf_id"],
            "--task", "text-classification",
            temp_dir
        ]

        # Step 2: Quantize
        print(f"\n>>> Step 2/2: Applying INT8 Quantization ({quant_flag})...")
        quantize_cmd = [
            cli_command, "onnxruntime", "quantize",
            "--onnx_model", temp_dir,
            quant_flag,
            "--output", data["output_dir"]
        ]

        try:
            subprocess.run(export_cmd, check=True)  # nosec B603
            subprocess.run(quantize_cmd, check=True)  # nosec B603
            print(f"\n[{name.upper()}] ✅ Successfully saved to {data['output_dir']}")

            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        except subprocess.CalledProcessError as e:
            print(f"\n[{name.upper()}] ❌ Failed to quantize model. Error: {e}")
        except FileNotFoundError:
            print(f"\n[{name.upper()}] ❌ Error: Could not find '{cli_command}'. Make sure optimum is installed.")

if __name__ == "__main__":
    print("Initializing ClearChain Model Quantizer...")
    run_quantization()
    print("\nAll models quantized! You can now update your config.json to point to these local directories.")
