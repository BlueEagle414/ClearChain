import os
import sys
import getpass
import asyncio
import logging
import argparse

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Markdown, Log, Label, Button, TextArea
from textual.screen import ModalScreen
from textual import work

from core.security import get_api_key, set_api_key
from core.llm import get_llm_provider
from core.cove_pipeline import execute_cove
from db.migrations import check_and_migrate
from db.database import init_db, flush_all_data, add_tech_spec
from core.local_models import local_models
from config import config, save_config, USER_DATA_DIR

from cli import run_cli

log_file_path = os.path.join(USER_DATA_DIR, "daemon.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if os.path.exists(log_file_path):
    os.chmod(log_file_path, 0o600)

def get_api_key_sync():
    text_prov = config.get("text_provider", "gemini").lower()
    embed_prov = config.get("embedding_provider", "gemini").lower()
    
    def is_local(prov):
        if prov == "ollama":
            return True
        if prov == "openai" and config.get("openai_base_url"):
            return True
        return False
        
    if is_local(text_prov) and is_local(embed_prov):
        return None
        
    print("\nInitializing ClearChain Daemon...")
    try:
        return get_api_key()
    except ValueError as e:
        print(f"\n{e}")
        key = getpass.getpass("Enter API Key: ")
        if key.strip():
            set_api_key(key.strip())
            return key.strip()
            
    print("API Key is strictly required for cloud providers. Exiting.")
    sys.exit(1)

class WipeDbScreen(ModalScreen[bool]):
    CSS = """
    WipeDbScreen { align: center middle; }
    #wipe_dialog { padding: 1 2; width: 50; height: 15; border: thick $error 80%; background: $surface; }
    #wipe_buttons { height: 3; align: center middle; margin-top: 2; }
    Button { margin: 0 1; }
    """
    def compose(self) -> ComposeResult:
        with Vertical(id="wipe_dialog"):
            yield Label("⚠️ WARNING: This will permanently delete all data in the knowledge base and cache. Are you sure?", id="wipe_label")
            with Horizontal(id="wipe_buttons"):
                yield Button("Yes, Wipe DB", variant="error", id="confirm_wipe_btn")
                yield Button("Cancel", variant="primary", id="cancel_wipe_btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_wipe_btn":
            self.dismiss(True)
        else:
            self.dismiss(False)

class AddDataScreen(ModalScreen[tuple[str, str]]):
    CSS = """
    AddDataScreen { align: center middle; }
    #dialog { padding: 1 2; width: 60; height: 20; border: thick $background 80%; background: $surface; }
    .input_field { margin-bottom: 1; }
    #details_input { height: 1fr; margin-bottom: 1; }
    #buttons { height: 3; align: center middle; }
    Button { margin: 0 1; }
    """
    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Add New Tech Spec to Knowledge Base")
            yield Input(placeholder="Entity Name (e.g., Omega-77)", id="entity_input", classes="input_field")
            yield TextArea(id="details_input", classes="input_field")
            with Horizontal(id="buttons"):
                yield Button("Save", variant="success", id="save_btn")
                yield Button("Cancel", variant="error", id="cancel_btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_btn":
            entity = self.query_one("#entity_input", Input).value.strip()
            details = self.query_one("#details_input", TextArea).text.strip()
            if entity and details:
                self.dismiss((entity, details))
            else:
                self.app.notify("Both Entity and Details are required.", severity="error")
        else:
            self.dismiss(None)

class CoVeApp(App):
    CSS = """
    #main_container { height: 100%; }
    #left_pane { width: 65%; height: 100%; border-right: solid green; }
    #right_pane { width: 35%; height: 100%; }
    #markdown_viewer { height: 1fr; padding: 1; overflow-y: auto; }
    #query_input { dock: bottom; margin: 1; }
    #log_viewer { height: 1fr; padding: 1; background: $surface; }
    #settings_bar { height: 3; dock: top; padding: 1; background: $boost; }
    .setting_input { width: 10; margin-left: 1; margin-right: 2; }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "clear", "Clear Screen"),
        ("w", "wipe_db", "Wipe DB"),
        ("a", "add_data", "Add Data"),
        ("escape", "abort", "Abort Query")
    ]

    def __init__(self, api_key: str):
        super().__init__()
        self.provider = get_llm_provider(config, api_key)
        self.current_draft = ""
        self.fatal_error = None
        self.cove_worker = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_container"):
            with Vertical(id="left_pane"):
                yield Markdown("# ClearChain\nType a query below to begin.", id="markdown_viewer")
                yield Input(placeholder="Enter your query here...", id="query_input")
            with Vertical(id="right_pane"):
                with Horizontal(id="settings_bar"):
                    yield Label("Sim Thresh:")
                    yield Input(value=str(config["similarity_threshold"]), id="sim_thresh", classes="setting_input")
                    yield Label("Conf Thresh:")
                    yield Input(value=str(config["confidence_threshold"]), id="conf_thresh", classes="setting_input")
                yield Log(id="log_viewer", highlight=True)
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#query_input").focus()
        await self.log_msg("Initializing database and models...")
        
        try:
            await check_and_migrate(self.provider)
            await init_db(self.provider)
        except Exception as e:
            self.fatal_error = f"Database Initialization Failed: {e}"
            self.exit(1)
            return
            
        if config.get("use_local_security") or config.get("use_local_routing"):
            await self.log_msg("Loading local BERT models into memory...")
            try:
                await asyncio.to_thread(local_models.initialize_models)
            except Exception as e:
                self.fatal_error = f"Local Model Initialization Failed: {e}"
                self.exit(1)
                return
                
        await self.log_msg("System initialized. Ready for queries.")

    async def action_clear(self) -> None:
        self.query_one("#markdown_viewer", Markdown).update("# ClearChain\nType a query below to begin.")
        self.query_one("#log_viewer", Log).clear()
        await self.log_msg("Screen cleared.")

    async def action_wipe_db(self) -> None:
        def check_reply(confirm: bool | None) -> None:
            if confirm:
                self.run_wipe_worker()
            else:
                self.notify("Database wipe cancelled.", severity="information")
        self.push_screen(WipeDbScreen(), check_reply)

    async def action_abort(self) -> None:
        if self.cove_worker and not self.cove_worker.is_finished:
            self.cove_worker.cancel()
            await self.log_msg("[!] Abort signal sent. Cancelling query...")
            self.query_one("#markdown_viewer", Markdown).update("## Query Aborted\n\nThe query was cancelled by the user.")
        else:
            self.notify("No active query to abort.", severity="information")

    @work(exclusive=True)
    async def run_wipe_worker(self) -> None:
        await self.log_msg("Initiating database wipe...")
        success = await flush_all_data(self.provider)
        if success:
            await self.log_msg("Database wiped and re-initialized successfully.")
            self.notify("Database wiped successfully.", severity="information")
        else:
            await self.log_msg("Failed to wipe database. Check logs.")
            self.notify("Failed to wipe database.", severity="error")

    async def action_add_data(self) -> None:
        def check_reply(data: tuple[str, str] | None) -> None:
            if data is not None:
                entity, details = data
                self.run_add_data_worker(entity, details)
        self.push_screen(AddDataScreen(), check_reply)

    @work(exclusive=True)
    async def run_add_data_worker(self, entity: str, details: str) -> None:
        if len(details) > 50000:
            self.notify("Details text is too large. Please limit to 50,000 characters.", severity="error")
            return
            
        await self.log_msg(f"Adding data for entity: '{entity}'...")
        try:
            await add_tech_spec(self.provider, entity, details)
            await self.log_msg(f"Successfully added '{entity}' to the database.")
            self.notify(f"Added '{entity}' to database.", severity="information")
        except ValueError as e:
            await self.log_msg(f"Notice: {e}")
            self.notify(str(e), severity="warning")
        except Exception as e:
            await self.log_msg(f"Error adding data: {e}")
            self.notify("Error adding data.", severity="error")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "sim_thresh":
            try:
                val = float(event.value)
                config["similarity_threshold"] = val
                save_config({"similarity_threshold": val})
                await self.log_msg(f"Similarity threshold updated to {val}")
            except ValueError:
                await self.log_msg("Invalid similarity threshold value.")
            return
            
        if event.input.id == "conf_thresh":
            try:
                val = float(event.value)
                config["confidence_threshold"] = val
                save_config({"confidence_threshold": val})
                await self.log_msg(f"Confidence threshold updated to {val}")
            except ValueError:
                await self.log_msg("Invalid confidence threshold value.")
            return
            
        if event.input.id == "query_input":
            query = event.value.strip()
            if not query:
                return
            if len(query) > 2000:
                self.notify("Query is too long. Please limit to 2000 characters.", severity="error")
                return
                
            event.input.value = ""
            self.query_one("#markdown_viewer", Markdown).update(f"## Query: {query}\n\n*Processing...*")
            self.query_one("#log_viewer", Log).clear()
            await self.log_msg(f"Received query: {query}")
            self.cove_worker = self.run_cove_pipeline(query)

    @work(exclusive=True)
    async def run_cove_pipeline(self, query: str) -> None:
        self.current_draft = ""
        
        async def log_cb(msg: str):
            await self.log_msg(msg)
            
        async def stream_cb(chunk: str):
            self.current_draft += chunk
            md = self.query_one("#markdown_viewer", Markdown)
            await md.update(f"## Query: {query}\n\n### Draft\n{self.current_draft}")
            
        async def final_answer_cb(answer: str):
            md = self.query_one("#markdown_viewer", Markdown)
            await md.update(f"## Query: {query}\n\n### Final Verified Answer\n{answer}")
            
        try:
            result = await execute_cove(self.provider, query, log_cb, stream_cb, final_answer_cb)
            
            score = result.get("confidence_score", 0.0)
            threshold = config["confidence_threshold"]
            
            status = "VERIFIED"
            if score < threshold:
                status = "REJECTED"
            elif result.get("hallucinations_caught"):
                status = "VERIFIED & CORRECTED"
                
            await self.log_msg(f"\nPipeline Complete. Status: {status} (Score: {score})")
        except asyncio.CancelledError:
            await self.log_msg("[!] Pipeline execution was successfully aborted.")
            raise

    async def log_msg(self, msg: str) -> None:
        log_widget = self.query_one("#log_viewer", Log)
        log_widget.write_line(msg.strip())
        await asyncio.to_thread(logging.info, msg.strip())

def main():
    parser = argparse.ArgumentParser(description="ClearChain Daemon")
    parser.add_argument("--cli", action="store_true", help="Run in lightweight CLI mode instead of the TUI")
    args = parser.parse_args()

    api_key = get_api_key_sync()
    
    if args.cli:
        asyncio.run(run_cli(api_key))
    else:
        app = CoVeApp(api_key)
        app.run()
        
        if app.fatal_error:
            print(f"\n[!] DAEMON CRASHED DURING STARTUP:\n{app.fatal_error}\n")

if __name__ == "__main__":
    main()
