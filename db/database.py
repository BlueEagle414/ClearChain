import os
import json
import lancedb
import pyarrow as pa
import hashlib
import logging
import asyncio
from db.chunker import chunk_text
from config import USER_DATA_DIR

DB_PATH = os.path.join(USER_DATA_DIR, "lancedb_data")

_db_connection = None
_db_conn_lock = asyncio.Lock()
db_write_lock = asyncio.Lock()

async def get_db_connection():
    global _db_connection
    if _db_connection is None:
        async with _db_conn_lock:
            if _db_connection is None:
                _db_connection = await lancedb.connect_async(DB_PATH)
    return _db_connection

async def init_db(provider):
    db = await get_db_connection()
    table_names = await db.table_names()
    
    if "tech_specs" in table_names and "cove_cache" in table_names:
        return
        
    test_embedding = await provider.generate_embedding("test_dimension")
    dim = len(test_embedding)
    logging.info(f"Initialized Database with dynamic embedding dimension: {dim}")
    
    tech_specs_schema = pa.schema([
        pa.field("entity", pa.string()),
        pa.field("entity_hash", pa.string()),
        pa.field("details", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim))
    ])
    
    cache_schema = pa.schema([
        pa.field("query_text", pa.string()),
        pa.field("query_hash", pa.string()),
        pa.field("response_json", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim))
    ])
    
    async with db_write_lock:
        if "tech_specs" not in table_names:
            await db.create_table("tech_specs", schema=tech_specs_schema, exist_ok=True)
            
        if "cove_cache" not in table_names:
            await db.create_table("cove_cache", schema=cache_schema, exist_ok=True)

async def flush_all_data(provider) -> bool:
    logging.warning("Initiating complete database flush...")
    try:
        db = await get_db_connection()
        table_names = await db.table_names()
        
        async with db_write_lock:
            if "tech_specs" in table_names:
                await db.drop_table("tech_specs")
                logging.info("Successfully dropped table: tech_specs")
                
            if "cove_cache" in table_names:
                await db.drop_table("cove_cache")
                logging.info("Successfully dropped table: cove_cache")
                
        logging.info("Database flush complete. Re-initializing empty schema...")
        
        await init_db(provider)
        
        return True
        
    except Exception as e:
        logging.error(f"Critical failure during database flush: {e}")
        return False

async def add_tech_spec(provider, entity: str, details: str):
    db = await get_db_connection()
    tbl = await db.open_table("tech_specs")
    e_hash = hashlib.sha256(entity.encode()).hexdigest()
    
    try:
        query_builder = tbl.search()
        if asyncio.iscoroutine(query_builder):
            query_builder = await query_builder
        existing = await query_builder.where(f"entity_hash = '{e_hash}'").to_list()
        if existing:
            raise ValueError(f"Entity '{entity}' already exists in the Knowledge Base.")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        logging.warning(f"Could not check for existing entity: {e}")
        
    chunks = chunk_text(details)
    data_to_insert = []
    
    for chunk in chunks:
        text_to_embed = f"{entity}: {chunk}"
        embedding = await provider.generate_embedding(text_to_embed)
        data_to_insert.append({
            "entity": entity,
            "entity_hash": e_hash,
            "details": chunk,
            "vector": embedding
        })
        
    async with db_write_lock:
        await tbl.add(data_to_insert)
        
    # Cache Invalidation Strategy
    logging.info("Invalidating cove_cache due to new tech_specs...")
    try:
        cache_tbl = await db.open_table("cove_cache")
        cache_schema = cache_tbl.schema
        await db.drop_table("cove_cache", ignore_missing=True)
        await db.create_table("cove_cache", schema=cache_schema)
    except Exception as e:
        logging.error(f"Failed to invalidate cache: {e}")

async def get_context(provider, query: str) -> tuple[str, float]:
    query_embedding = await provider.generate_embedding(query)
    db = await get_db_connection()
    tbl = await db.open_table("tech_specs")
    
    query_builder = tbl.search(query_embedding)
    if asyncio.iscoroutine(query_builder):
        query_builder = await query_builder
        
    # Version-safe metric/distance call
    try:
        results = await query_builder.distance_type("cosine").limit(5).to_list()
    except AttributeError:
        results = await query_builder.metric("cosine").limit(5).to_list()
        
    if not results:
        return "", 0.0
        
    highest_score = 1.0 - results[0]["_distance"]
    context_lines = [f"- {row['entity']}: {row['details']}" for row in results]
    return "\n".join(context_lines), highest_score

async def get_cached_result(provider, query: str, threshold: float = 0.95) -> dict | None:
    try:
        db = await get_db_connection()
        tbl = await db.open_table("cove_cache")
        query_embedding = await provider.generate_embedding(query)
        
        query_builder = tbl.search(query_embedding)
        if asyncio.iscoroutine(query_builder):
            query_builder = await query_builder
            
        try:
            results = await query_builder.distance_type("cosine").limit(1).to_list()
        except AttributeError:
            results = await query_builder.metric("cosine").limit(1).to_list()
            
        if results:
            similarity = 1.0 - results[0]["_distance"]
            logging.debug(f"Cache similarity score: {similarity:.4f}")
            if similarity >= threshold:
                logging.info("Semantic cache match threshold met!")
                return json.loads(results[0]["response_json"])
    except Exception as e:
        logging.error(f"Failed to retrieve cached result: {e}")
    return None

async def save_cached_result(provider, query: str, result: dict):
    try:
        db = await get_db_connection()
        tbl = await db.open_table("cove_cache")
        query_embedding = await provider.generate_embedding(query)
        q_hash = hashlib.sha256(query.encode()).hexdigest()
        
        async with db_write_lock:
            try:
                await tbl.delete(f"query_hash = '{q_hash}'")
            except Exception as e:
                logging.warning(f"Failed to delete old cache entry: {e}")
                
            await tbl.add([{
                "query_text": query,
                "query_hash": q_hash,
                "response_json": json.dumps(result),
                "vector": query_embedding
            }])
    except Exception as e:
        logging.error(f"Failed to save cached result: {e}")
