import logging
import threading
import numpy as np
from config import config

try:
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

class LocalModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LocalModelManager, cls).__new__(cls)
                cls._instance.security_model = None
                cls._instance.security_tokenizer = None
                cls._instance.routing_model = None
                cls._instance.routing_tokenizer = None
        return cls._instance

    def initialize_models(self):
        if config.get("use_local_security") and self.security_model is None:
            if not OPTIMUM_AVAILABLE:
                raise RuntimeError("Optimum/ONNX library is missing. Stopping to prevent data leakage.")
            else:
                logging.info("Loading INT8 ONNX Security Model (PyTorch-Free)...")
                model_path = config["local_security_model"]

                self.security_tokenizer = AutoTokenizer.from_pretrained(model_path)  # nosec B615
                self.security_model = ORTModelForSequenceClassification.from_pretrained(model_path)  # nosec B615

        if config.get("use_local_routing") and self.routing_model is None:
            if not OPTIMUM_AVAILABLE:
                raise RuntimeError("Optimum/ONNX library is missing. Halting to prevent data leakage.")
            else:
                logging.info("Loading INT8 ONNX Routing Model (PyTorch-Free)...")
                model_path = config["local_routing_model"]

                self.routing_tokenizer = AutoTokenizer.from_pretrained(model_path)  # nosec B615
                self.routing_model = ORTModelForSequenceClassification.from_pretrained(model_path)  # nosec B615

    def detect_prompt_injection(self, text: str, threshold: float = 0.85) -> bool:
        if not OPTIMUM_AVAILABLE:
            raise RuntimeError("Optimum library is missing.")

        if not self.security_model:
            self.initialize_models()

        # 1. Tokenize and return NumPy arrays instead of PyTorch tensors
        inputs = self.security_tokenizer(text, return_tensors="np", truncation=True, max_length=512)

        # 2. Run through ONNX Runtime
        outputs = self.security_model(**inputs)
        logits = outputs.logits[0]

        # 3. Manual Softmax using pure NumPy
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # 4. Extract results
        pred_id = int(np.argmax(probs))
        label = self.security_model.config.id2label[pred_id].upper()
        score = float(probs[pred_id])

        is_threat_label = label in ["INJECTION", "MALICIOUS", "LABEL_1"]
        is_confident = score >= threshold

        if is_threat_label and is_confident:
            logging.warning(f"High-confidence prompt injection blocked: Label '{label}' with score {score:.4f}")
            return True
        elif is_threat_label:
            logging.info(f"Low-confidence injection label ignored: '{label}' with score {score:.4f}")

        return False

    def classify_text(self, text: str, candidate_labels: list[str], multi_label: bool = False) -> dict:
        """Manual Zero-Shot Classification using pure NumPy and ONNX."""
        if not OPTIMUM_AVAILABLE:
            raise RuntimeError("Optimum library is missing.")

        if not self.routing_model:
            self.initialize_models()

        # Find the Entailment and Contradiction IDs from the model config
        entailment_id, contradiction_id = 1, 0 # Defaults
        for k, v in self.routing_model.config.label2id.items():
            if k.lower().startswith("entail"): entailment_id = v
            if k.lower().startswith("contradict"): contradiction_id = v

        # Create pairs: [Text, "This example is {label}."]
        texts = [text] * len(candidate_labels)
        text_pairs = [f"This example is {label}." for label in candidate_labels]

        # Tokenize all pairs into NumPy arrays
        inputs = self.routing_tokenizer(texts, text_pairs, return_tensors="np", padding=True, truncation=True, max_length=512)

        # Run through ONNX Runtime
        outputs = self.routing_model(**inputs)
        logits = outputs.logits # Shape: (num_labels, 3)

        if multi_label:
            # Independent softmax for each label (Entailment vs Contradiction)
            entail_contra_logits = logits[:, [contradiction_id, entailment_id]]
            exp_logits = np.exp(entail_contra_logits - np.max(entail_contra_logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            scores = probs[:, 1] # Take entailment probability
        else:
            # Softmax across all labels' entailment scores
            entail_logits = logits[:, entailment_id]
            exp_logits = np.exp(entail_logits - np.max(entail_logits))
            scores = exp_logits / exp_logits.sum()

        # Sort results highest to lowest
        sorted_indices = np.argsort(scores)[::-1]

        return {
            "labels": [candidate_labels[i] for i in sorted_indices],
            "scores": [float(scores[i]) for i in sorted_indices]
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

    def check_query_entailment(self, query_a: str, query_b: str, threshold: float = 0.80) -> bool:
        """Uses the local routing model to check if two queries mean the same thing."""
        if not config.get("use_local_routing"):
            return True

        if not OPTIMUM_AVAILABLE:
            return True

        if not self.routing_model:
            try:
                self.initialize_models()
            except Exception as e:
                logging.error(f"Failed to initialize models for entailment check: {e}")
                return True

        if not self.routing_model or not self.routing_tokenizer:
            return True

        try:
            # Tokenize the pair together for cross-attention
            inputs = self.routing_tokenizer(
                query_a, query_b, 
                return_tensors="np", truncation=True, max_length=512
            )
            
            outputs = self.routing_model(**inputs)
            logits = outputs.logits[0]
            
            # Manual Softmax using pure NumPy
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            # Find the Entailment ID from the model config
            entailment_id = 1 # Default fallback
            for k, v in self.routing_model.config.label2id.items():
                if k.lower().startswith("entail"): 
                    entailment_id = v
                    break
                    
            entailment_score = float(probs[entailment_id])
            
            return entailment_score >= threshold
        except Exception as e:
            logging.error(f"Entailment check failed: {e}")
            return True

local_models = LocalModelManager()
