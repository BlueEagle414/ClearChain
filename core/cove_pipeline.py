import json
import asyncio
import yaml
import os
from typing import Callable, Dict, Any, Awaitable
from core.llm.base import LLMProvider
from core.llm_service import (
    detect_prompt_injection, detect_needs_verification, classify_content
)
from core.stream_parser import clean_and_parse_json_stream
from db.database import get_context, get_cached_result, save_cached_result
from config import config

# Load externalized prompts
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

# Semaphore to limit concurrent API calls during Phase 0 to prevent rate limits
api_semaphore = asyncio.Semaphore(2)

async def bounded_call(coro):
    async with api_semaphore:
        return await coro

async def execute_cove(
    provider: LLMProvider, 
    query: str, 
    log_callback: Callable[[str], Awaitable[None]], 
    stream_callback: Callable[[str], Awaitable[None]],
    final_answer_callback: Callable[[str], Awaitable[None]]
) -> Dict[str, Any]:
    try:
        normalized_query = query.strip().lower()
        
        # Step -1: Semantic Cache Check
        cached_result = await get_cached_result(provider, normalized_query)
        if cached_result:
            await log_callback("[*] CACHE HIT: Semantic match found in local cache. Serving immediately.\n")
            await final_answer_callback(cached_result.get("final_answer", ""))
            return cached_result

        await log_callback("[*] Running parallel Security Check, Context Retrieval, and Classification...")
        
        # Step 0: Parallelize Security, Context, and Classification with Rate Limiting
        results = await asyncio.gather(
            bounded_call(detect_prompt_injection(provider, query)),
            bounded_call(get_context(provider, query)),
            bounded_call(classify_content(provider, query, config.get("domain_labels", ["technology", "general"]), False)),
            bounded_call(classify_content(provider, query, config.get("tag_labels", ["configuration", "troubleshooting"]), True)),
            return_exceptions=True
        )
        
        for res in results:
            if isinstance(res, Exception):
                raise res
                
        is_malicious = results[0]
        context, max_similarity = results[1]
        domain_res = results[2]
        tags_res = results[3]
        
        if is_malicious:
            await log_callback("[!] SECURITY ALERT: Malicious prompt injection detected. Aborting.")
            ans = "Security Error: Prompt injection or jailbreak attempt detected."
            await final_answer_callback(ans)
            return {"final_answer": ans, "hallucinations_caught": False, "confidence_score": 0.0}
            
        top_domain = domain_res["labels"][0] if domain_res.get("labels") else "unknown"
        top_tags = [label for label, score in zip(tags_res.get("labels", []), tags_res.get("scores", [])) if score > 0.4]
        if not top_tags and tags_res.get("labels"):
            top_tags = [tags_res["labels"][0]]
            
        await log_callback(f"[*] Classification -> Domain: [{top_domain.upper()}], Tags: {top_tags}")
        await log_callback(f"[*] Highest Vector Similarity Score: {max_similarity:.4f}")
        
        threshold = config["similarity_threshold"]
        if max_similarity < threshold:
            abort_msg = "No securely vetted information exists in the local database for this query."
            await log_callback(f"[!] THRESHOLD FAILED (< {threshold}). Aborting pipeline.\n")
            await final_answer_callback(abort_msg)
            return {"final_answer": abort_msg, "hallucinations_caught": False, "confidence_score": 0.0}

        await log_callback(f"--- Context Retrieved ---\n{context}\n")

        # Phase 1: The Draft (Streaming)
        await log_callback("[Phase 1] Generating Initial Draft (API Call 1)...\n--- Draft Stream ---")
        
        system_prompt = prompts["system_prompts"]["phase1_draft"].format(
            domain=top_domain.title(),
            tags=', '.join(top_tags),
            context=context
        )
        
        safe_query = query.replace("<query>", "").replace("</query>", "")
        user_prompt = f"<query>{safe_query}</query>"
        
        draft_chunks = []
        async for chunk in provider.generate_text_stream(user_prompt, system_instruction=system_prompt):
            draft_chunks.append(chunk)
            await stream_callback(chunk)
            
        draft = "".join(draft_chunks)
        await stream_callback("\n\n") 
        await log_callback(f"--- Draft Completed ---\n")

        trigger_phrases = prompts["triggers"]["abstain_phrases"]
        
        is_short = len(draft.split()) < 50
        has_abstain = any(phrase in draft.lower() for phrase in trigger_phrases)
        
        if has_abstain:
            await log_callback("[*] ABSTAIN DETECTED: The model correctly identified missing information. Bypassing verification.")
            result_dict = {"final_answer": draft.strip(), "hallucinations_caught": False, "confidence_score": 1.0}
            await final_answer_callback(result_dict["final_answer"])
            await save_cached_result(provider, normalized_query, result_dict)
            return result_dict
            
        if is_short:
            await log_callback("[*] FAST-PATH CHECK: Draft is short. Checking if verification is needed...")
            needs_verification = await detect_needs_verification(provider, draft)
            if not needs_verification:
                await log_callback("[*] FAST-PATH ROUTER: No complex claims detected. Skipping Phases 2 & 3 to save latency.")
                result_dict = {"final_answer": draft.strip(), "hallucinations_caught": False, "confidence_score": 0.95}
                await final_answer_callback(result_dict["final_answer"])
                await save_cached_result(provider, normalized_query, result_dict)
                return result_dict

        # Phase 2: The Interrogation
        await log_callback("[Phase 2] Generating Interrogation Questions (API Call 2)...")
        prompt2 = prompts["system_prompts"]["phase2_interrogation"].format(
            context=context,
            draft=draft
        )
        
        questions = await provider.generate_text(prompt2)
        await log_callback(f"--- Interrogation Questions ---\n{questions.strip()}\n")

        # Phase 3: The Revision (Streaming JSON)
        await log_callback("[Phase 3] Finalizing JSON and removing hallucinations (API Call 3)...\n--- JSON Stream ---")
        
        prompt3 = prompts["system_prompts"]["phase3_revision"].format(
            context=context,
            draft=draft,
            questions=questions
        )
        
        json_generator = provider.generate_json_stream(prompt3)
        result_dict = await clean_and_parse_json_stream(json_generator, log_callback, final_answer_callback)
        
        await log_callback(f"\n--- Structured JSON Response ---\n{json.dumps(result_dict, indent=2)}\n")
        
        await save_cached_result(provider, normalized_query, result_dict)
        
        return result_dict

    except asyncio.CancelledError:
        await log_callback("\n[!] Operation cancelled by user.\n")
        raise
    except Exception as e:
        await log_callback(f"[!] Pipeline Error: {str(e)}")
        err_msg = f"Error during verification: {str(e)}"
        await final_answer_callback(err_msg)
        return {
            "final_answer": err_msg,
            "hallucinations_caught": False,
            "confidence_score": 0.0
        }
