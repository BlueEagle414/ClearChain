import json
import os
import appdirs

# Permission errors
USER_DATA_DIR = appdirs.user_data_dir("cove_verifier", "ClearChain")
os.makedirs(USER_DATA_DIR, mode=0o700, exist_ok=True)

CONFIG_PATH = os.path.join(USER_DATA_DIR, "config.json")

DEFAULT_CONFIG = {
    "llm_provider": "gemini",
    "similarity_threshold": 0.75,
    "confidence_threshold": 0.8,
    "primary_model": "gemini-3.1-pro-preview",
    "json_model": "gemini-3.1-pro-preview",
    "embedding_model": "models/gemini-embedding-001",
    "last_embedding_model": "", 
    "use_local_security": True,
    "use_local_routing": True,
    "local_security_model": "protectai/deberta-v3-base-prompt-injection-v2",
    "local_routing_model": "cross-encoder/nli-MiniLM2-L6-H768",
    "domain_labels": ["technology", "cybersecurity", "networking", "software development", "hardware", "general inquiry"],
    "tag_labels": ["configuration", "troubleshooting", "architecture", "performance", "security policy", "legacy system"]
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        os.chmod(CONFIG_PATH, 0o600)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, "r") as f:
            loaded = json.load(f)
            # Merge with defaults in case new keys were added
            for k, v in DEFAULT_CONFIG.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
    except json.JSONDecodeError:
        return DEFAULT_CONFIG

def save_config(new_config):
    global config
    config.update(new_config)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    os.chmod(CONFIG_PATH, 0o600)

config = load_config()
