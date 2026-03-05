import os
import keyring
from config import USER_DATA_DIR

APP_NAME = "ClearChain"
KEY_NAME = "cove_llm_api_key"
FALLBACK_KEY_PATH = os.path.join(USER_DATA_DIR, ".api_key")

_session_api_key = None

def is_headless_linux():
    if os.name != "posix":
        return False
    return not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")

def get_api_key() -> str:
    global _session_api_key
    if _session_api_key:
        return _session_api_key
        
    # Fallback 1: Environment variable
    env_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
        
    # Fallback 2: Secure hidden file
    if os.path.exists(FALLBACK_KEY_PATH):
        try:
            with open(FALLBACK_KEY_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            pass

    # Fallback 3: OS Keyring
    if not is_headless_linux():
        try:
            api_key = keyring.get_password(APP_NAME, KEY_NAME)
            if api_key:
                return api_key
        except Exception:
            pass 

    raise ValueError("API Key not found in Environment, Secure File, or OS Keyring.")

def set_api_key(api_key: str):
    global _session_api_key
    _session_api_key = api_key
    
    if not is_headless_linux():
        try:
            keyring.set_password(APP_NAME, KEY_NAME, api_key)
            return
        except Exception:
            pass
            
    # Fallback to secure file
    with open(FALLBACK_KEY_PATH, "w") as f:
        f.write(api_key)
    os.chmod(FALLBACK_KEY_PATH, 0o600)

def delete_api_key():
    global _session_api_key
    try:
        keyring.delete_password(APP_NAME, KEY_NAME)
    except Exception:
        pass
        
    if "LLM_API_KEY" in os.environ:
        del os.environ["LLM_API_KEY"]
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
        
    if os.path.exists(FALLBACK_KEY_PATH):
        os.remove(FALLBACK_KEY_PATH)
        
    _session_api_key = None