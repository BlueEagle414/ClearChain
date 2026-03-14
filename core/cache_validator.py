import re
import logging
import asyncio
from core.local_models import local_models

class CacheValidator:
    @staticmethod
    def extract_constraints(text: str) -> set:
        """Extracts numbers, IPs, versions, and specific technical constraints."""
        # Matches IPs, versions (v1.2), and standalone numbers/ports
        pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\bv?\d+(?:\.\d+)+\b|\b\d+\b'
        return set(re.findall(pattern, text.lower()))

    @staticmethod
    async def validate_candidate(new_query: str, cached_query: str) -> bool:
        # 1. Constraint Check (Fast Fail)
        new_constraints = CacheValidator.extract_constraints(new_query)
        cached_constraints = CacheValidator.extract_constraints(cached_query)
        
        if new_constraints != cached_constraints:
            logging.debug(f"Cache rejected due to constraint mismatch: {new_constraints} vs {cached_constraints}")
            return False

        # 2. Intent Check via Local NLI (Entailment)
        # Reusing the local routing model to check if the intents match
        is_match = await asyncio.to_thread(local_models.check_query_entailment, new_query, cached_query)
        if not is_match:
            logging.debug("Cache rejected by local NLI intent check.")
            return False

        return True