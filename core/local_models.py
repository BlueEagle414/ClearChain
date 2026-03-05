import logging
import threading
from config import config

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LocalModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LocalModelManager, cls).__new__(cls)
                cls._instance.security_pipeline = None
                cls._instance.routing_pipeline = None
        return cls._instance

    def initialize_models(self):
        if config.get("use_local_security") and self.security_pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Local security enforced but transformers library is missing. Stopping to prevent data leakage to API.")
            else:
                logging.info("Loading DistilRoBERTa Security Model...")
                self.security_pipeline = pipeline(
                    "text-classification", 
                    model=config["local_security_model"], 
                    device_map="auto"
                )

        if config.get("use_local_routing") and self.routing_pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Local routing enforced but transformers library is missing. Halting to prevent data leakage to API.")
            else:
                logging.info("Loading MiniLM Routing & Classification Model...")
                self.routing_pipeline = pipeline(
                    "zero-shot-classification", 
                    model=config["local_routing_model"], 
                    device_map="auto"
                )

    def detect_prompt_injection(self, text: str, threshold: float = 0.85) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Local security enforced but transformers library is missing. Stopping to prevent data leakage to API.")

        if not self.security_pipeline:
            self.initialize_models()
            if not self.security_pipeline:
                raise RuntimeError("Local security model failed to initialize.")
            
        result = self.security_pipeline(text, truncation=True, max_length=512)
        
        label = result[0]['label'].upper()
        score = result[0]['score']
        
        is_threat_label = label in ["INJECTION", "MALICIOUS", "LABEL_1"]
        is_confident = score >= threshold
        
        if is_threat_label and is_confident:
            logging.warning(f"High-confidence prompt injection blocked: Label '{label}' with score {score:.4f}")
            return True
        elif is_threat_label:
            # This is crucial for tuning! It lets you see when DistilRoBERTa is being dramatic.
            logging.info(f"Low-confidence injection label ignored: '{label}' with score {score:.4f}")
            
        return False

    def classify_text(self, text: str, candidate_labels: list[str], multi_label: bool = False) -> dict:
        """Uses MiniLM to classify text into domains, tags, or intents."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Local routing enforced but transformers library is missing. Halting to prevent data leakage to API.")

        if not self.routing_pipeline:
            self.initialize_models()
            if not self.routing_pipeline:
                raise RuntimeError("Local routing model failed to initialize.")
            
        result = self.routing_pipeline(text, candidate_labels, multi_label=multi_label, truncation=True, max_length=512)
        return {
            "labels": result["labels"],
            "scores": result["scores"]
        }

    def detect_needs_verification(self, draft: str, confidence_threshold: float = 0.60) -> bool:
        candidate_labels = [
            "factual claim or technical explanation", 
            "conversational greeting", 
            "refusal to answer or missing data",
            "clarifying question"
        ]
        try:
            result = self.classify_text(draft, candidate_labels, multi_label=False)
            
            top_intent = result['labels'][0]
            top_score = result['scores'][0]
            
            logging.debug(f"Draft routing intent: '{top_intent}' (Score: {top_score:.4f})")
            
            safe_to_skip_labels = ["conversational greeting", "refusal to answer or missing data", "clarifying question"]
            
            if top_intent in safe_to_skip_labels and top_score >= confidence_threshold:
                logging.info(f"Fast-path triggered: Draft classified as '{top_intent}' ({top_score:.4f}). Bypassing verification.")
                return False
                
            if top_score < confidence_threshold:
                logging.warning(f"Low confidence intent routing ({top_score:.4f}). Defaulting to verification for safety.")
                
            return True
            
        except Exception as e:
            logging.error(f"Local routing failed during fast-path check: {e}. Defaulting to verification to prevent data leaks.")
            return True

local_models = LocalModelManager()
