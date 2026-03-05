import os
import sys
import cmd
import getpass
import logging
from rich.console import Console
from rich.panel import Panel

from google import genai
from db.database import init_db, add_tech_spec
from core.security import get_gemini_api_key, set_gemini_api_key, delete_gemini_api_key
from core.cove_pipeline import execute_cove
from core.llm_service import get_available_models
from core.local_models import local_models
from config import config, save_config, USER_DATA_DIR

log_file_path = os.path.join(USER_DATA_DIR, "daemon.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if os.path.exists(log_file_path):
    os.chmod(log_file_path, 0o600)

console = Console()

class CoVeDaemon(cmd.Cmd):
    intro = "Welcome to the ClearChain Daemon. Type 'help' or '?' to list commands.\n"
    prompt = "(cove) "

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)

    def cmdloop(self, intro=None):
        if intro is not None:
            self.intro = intro
        while True:
            try:
                super().cmdloop(intro="")
                break
            except KeyboardInterrupt:
                console.print("^C\n[yellow]Use 'exit' or 'quit' to close the daemon, or type a new command.[/yellow]")

    def do_verify(self, arg):
        if not arg:
            console.print("[red]Error: Please provide a query.[/red]")
            return

        console.print(f"\n[bold cyan]Running Verification for:[/bold cyan] {arg}\n")
        
        def log_cb(msg):
            console.print(f"[dim]{msg}[/dim]")
            logging.info(msg.strip())
            
        def stream_cb(chunk):
            sys.stdout.write(chunk)
            sys.stdout.flush()

        result = execute_cove(self.client, arg, log_cb, stream_cb)
        
        score = result.get("confidence_score", 0.0)
        threshold = config["confidence_threshold"]
        
        console.print("\n")
        if score < threshold:
            console.print(Panel(result.get("final_answer"), title=f"REJECTED (Score: {score})", border_style="red"))
        elif result.get("hallucinations_caught"):
            console.print(Panel(result.get("final_answer"), title=f"VERIFIED & CORRECTED (Score: {score})", border_style="yellow"))
        else:
            console.print(Panel(result.get("final_answer"), title=f"VERIFIED (Score: {score})", border_style="green"))
        console.print("\n")

    def do_add_kb(self, arg):
        """Add a new entity to the Knowledge Base. Usage: add_kb <Entity Name> | <Technical Details>"""
        try:
            entity, details = [x.strip() for x in arg.split("|", 1)]
            console.print(f"[yellow]Chunking and embedding '{entity}'...[/yellow]")
            add_tech_spec(self.client, entity, details)
            console.print(f"[bold green]Successfully added '{entity}' to LanceDB![/bold green]")
        except ValueError:
            console.print("[red]Error: Format must be 'Entity Name | Technical Details'[/red]")
        except Exception as e:
            console.print(f"[red]Database Error: {e}[/red]")

    def do_models(self, arg):
        console.print("[yellow]Fetching models from Gemini API...[/yellow]")
        try:
            text_models, embed_models = get_available_models(self.client)
            console.print("\n[bold]Text Models:[/bold]")
            for m in text_models: console.print(f" - {m}")
            console.print("\n[bold]Embedding Models:[/bold]")
            for m in embed_models: console.print(f" - {m}")
        except Exception as e:
            console.print(f"[red]Error fetching models: {e}[/red]")

    def do_config(self, arg):
        args = arg.split()
        if len(args) == 0:
            console.print(config)
        elif len(args) == 2:
            key, val = args[0], args[1]
            if key in config:
                # Type casting based on existing config
                val_type = type(config[key])
                try:
                    if val_type is bool:
                        parsed_val = val.lower() in ("true", "1", "yes", "y")
                    else:
                        parsed_val = val_type(val)
                    save_config({key: parsed_val})
                    console.print(f"[green]Updated {key} to {parsed_val}[/green]")
                except ValueError:
                    console.print(f"[red]Invalid type. {key} expects {val_type.__name__}[/red]")
            else:
                console.print(f"[red]Unknown config key: {key}[/red]")
        else:
            console.print("[red]Usage: config [key] [value][/red]")

    def do_clear_key(self, arg):
        delete_gemini_api_key()
        console.print("[green]API Key cleared. Exiting daemon...[/green]")
        return True

    def do_exit(self, arg):
        console.print("Shutting down...")
        return True
        
    def do_quit(self, arg):
        return self.do_exit(arg)

def main():
    console.print("[bold cyan]Initializing ClearChain Verifier Daemon...[/bold cyan]")
    
    # 1. Enforce Security
    try:
        api_key = get_gemini_api_key()
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        key = getpass.getpass("Enter Google Gemini API Key (Stored securely in OS Keyring): ")
        if key.strip():
            set_gemini_api_key(key.strip())
            api_key = key.strip()
        else:
            console.print("[red]API Key is strictly required. Exiting.[/red]")
            sys.exit(1)

    # 2. Initialize Database
    try:
        temp_client = genai.Client(api_key=api_key)
        init_db(temp_client)
    except Exception as e:
        console.print(f"[red]Failed to initialize database: {e}[/red]")
        sys.exit(1)

    # 3. Pre-load local models if enabled (Moved after API key validation)
    if config.get("use_local_security") or config.get("use_local_routing"):
        console.print("[dim]Loading local BERT models into memory...[/dim]")
        local_models.initialize_models()

    # 4. Start Daemon REPL
    CoVeDaemon(api_key).cmdloop()

if __name__ == "__main__":
    main()