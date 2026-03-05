import json
import logging
import partialjson
from typing import AsyncIterator, Callable, Dict, Any, Awaitable

async def clean_and_parse_json_stream(
    json_generator: AsyncIterator[str], 
    stream_callback: Callable[[str], Awaitable[None]],
    final_answer_callback: Callable[[str], Awaitable[None]]
) -> Dict[str, Any]:
    json_buffer = ""
    parser = partialjson.JSONParser()
    displayed_steps = 0
    
    async for chunk in json_generator:
        json_buffer += chunk
        
        try:
            parsed = parser.parse(json_buffer)
            if isinstance(parsed, dict):
                # Extract reasoning steps
                steps = parsed.get("reasoning_steps", [])
                if isinstance(steps, list):
                    while displayed_steps < len(steps):
                        step = steps[displayed_steps]
                        if step: # Only yield if the step is not empty
                            await stream_callback(f"- {step}\n")
                        displayed_steps += 1
                
                # Extract final answer as it streams
                final_ans = parsed.get("final_answer", "")
                if final_ans:
                    await final_answer_callback(final_ans)
                    
        except Exception as e:
            logging.debug(f"Partial parse error: {e}")

    await stream_callback("\n\n")
    
    try:
        return json.loads(json_buffer.strip())
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse final JSON: {e}\nRaw Buffer: {json_buffer}")
        
        try:
            fallback = parser.parse(json_buffer)
            if isinstance(fallback, dict):
                return fallback
        except Exception:
            pass
            
        return {
            "reasoning_steps": ["Error: Could not parse reasoning steps from the stream."],
            "final_answer": "Error: The model returned malformed JSON that could not be parsed.",
            "hallucinations_caught": False,
            "confidence_score": 0.0
        }
