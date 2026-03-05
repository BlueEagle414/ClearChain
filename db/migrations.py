import logging
from db.database import get_db_connection, init_db
from config import config, save_config

async def rebuild_db(provider):
    logging.info("Rebuilding database due to schema or model changes...")
    db = await get_db_connection()
    
    table_names = await db.table_names()
    
    if "tech_specs" in table_names:
        await db.drop_table("tech_specs")
    if "cove_cache" in table_names:
        await db.drop_table("cove_cache")
        
    await init_db(provider)
    
    # Update config to track the new model
    save_config({"last_embedding_model": config["embedding_model"]})
    logging.info("Database rebuild complete.")

async def check_and_migrate(provider):
    current_model = config.get("embedding_model")
    last_model = config.get("last_embedding_model")
    
    if current_model != last_model:
        await rebuild_db(provider)
