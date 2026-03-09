import os
import sys
import asyncio
import logging
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from core.llm import get_llm_provider
from core.cove_pipeline import execute_cove
from db.migrations import check_and_migrate
from db.database import init_db, flush_all_data, add_tech_spec
from core.local_models import local_models
from config import config, save_config

console = Console()

async def run_cli(api_key: str):
    console.print(Panel.fit("[bold green]ClearChain CLI[/bold green]\nInitializing system...", border_style="green"))
    
    provider = get_llm_provider(config, api_key)
    
    try:
        await check_and_migrate(provider)
        await init_db(provider)
    except Exception as e:
        console.print(f"[bold red]Database Initialization Failed:[/bold red] {e}")
        sys.exit(1)
        
    if config.get("use_local_security") or config.get("use_local_routing"):
        console.print("[dim]Loading local BERT models into memory...[/dim]")
        try:
            await asyncio.to_thread(local_models.initialize_models)
        except Exception as e:
            console.print(f"[bold red]Local Model Initialization Failed:[/bold red] {e}")
            sys.exit(1)
            
    console.print("[bold green]System initialized. Ready for queries.[/bold green]")
    console.print("Type [bold cyan]/help[/bold cyan] for a list of commands.\n")

    while True:
        try:
            query = Prompt.ask("\n[bold blue]Query[/bold blue]")
            query = query.strip()
            
            if not query:
                continue
                
            if query.startswith("/"):
                await handle_command(query, provider)
                continue

            if len(query) > 2000:
                console.print("[bold red]Error:[/bold red] Query is too long (limit 2000 chars).")
                continue

            await process_query(query, provider)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting ClearChain CLI...[/dim]")
            if hasattr(provider, 'close'):
                await provider.close()
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")

async def handle_command(command_str: str, provider):
    parts = command_str.split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ["/quit", "/exit", "/q"]:
        console.print("[dim]Exiting ClearChain CLI...[/dim]")
        if hasattr(provider, 'close'):
            await provider.close()
        sys.exit(0)
    elif cmd == "/help":
        console.print(Panel(
            "[cyan]/add[/cyan]   - Add new data to the knowledge base\n"
            "[cyan]/wipe[/cyan]  - Wipe the database and cache\n"
            "[cyan]/sim[/cyan]   - Set similarity threshold (e.g., /sim 0.75)\n"
            "[cyan]/conf[/cyan]  - Set confidence threshold (e.g., /conf 0.8)\n"
            "[cyan]/clear[/cyan] - Clear the terminal screen\n"
            "[cyan]/quit[/cyan]  - Exit the application",
            title="Available Commands", border_style="cyan"
        ))
    elif cmd == "/clear":
        console.clear()
        console.print(Panel.fit("[bold green]ClearChain CLI[/bold green]", border_style="green"))
    elif cmd == "/sim":
        try:
            val = float(arg)
            config["similarity_threshold"] = val
            save_config({"similarity_threshold": val})
            console.print(f"[green]Similarity threshold updated to {val}[/green]")
        except ValueError:
            console.print("[red]Invalid value. Usage: /sim 0.75[/red]")
    elif cmd == "/conf":
        try:
            val = float(arg)
            config["confidence_threshold"] = val
            save_config({"confidence_threshold": val})
            console.print(f"[green]Confidence threshold updated to {val}[/green]")
        except ValueError:
            console.print("[red]Invalid value. Usage: /conf 0.8[/red]")
    elif cmd == "/wipe":
        if Confirm.ask("[bold red]WARNING: Permanently delete all data?[/bold red]"):
            console.print("[dim]Wiping database...[/dim]")
            success = await flush_all_data(provider)
            if success:
                console.print("[green]Database wiped successfully.[/green]")
            else:
                console.print("[red]Failed to wipe database. Check logs.[/red]")
    elif cmd == "/add":
        entity = Prompt.ask("[cyan]Entity Name[/cyan]")
        console.print("[cyan]Enter Details (Type 'END' on a new line to save, or 'CANCEL' to abort):[/cyan]")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                if line.strip() == "CANCEL":
                    console.print("[red]Add data cancelled.[/red]")
                    return
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[red]Add data cancelled.[/red]")
                return
                
        details = "\n".join(lines).strip()

        if entity and details:
            if len(details) > 50000:
                console.print("[red]Details text is too large (limit 50,000 chars).[/red]")
                return
            console.print(f"\n[dim]Adding '{entity}' to database...[/dim]")
            try:
                await add_tech_spec(provider, entity, details)
                console.print(f"[green]Successfully added '{entity}'.[/green]")
            except Exception as e:
                console.print(f"[red]Error adding data:[/red] {e}")
        else:
            console.print("[red]Entity and Details are required. Cancelled.[/red]")
    else:
        console.print(f"[red]Unknown command:[/red] {cmd}. Type /help for a list of commands.")

async def process_query(query: str, provider):
    console.print("\n[dim]Processing...[/dim]")
    
    async def log_cb(msg: str):
        console.print(f"[dim]{msg.strip()}[/dim]")
        logging.info(msg.strip())
        
    async def stream_cb(chunk: str):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        
    async def final_answer_cb(answer: str):
        pass # Handled at the end to render full markdown
        
    try:
        result = await execute_cove(provider, query, log_cb, stream_cb, final_answer_cb)
        
        score = result.get("confidence_score", 0.0)
        threshold = config["confidence_threshold"]
        
        status = "[bold green]VERIFIED[/bold green]"
        if score < threshold:
            status = "[bold red]REJECTED[/bold red]"
        elif result.get("hallucinations_caught"):
            status = "[bold yellow]VERIFIED & CORRECTED[/bold yellow]"
            
        console.print(f"\n\n[bold]Status:[/bold] {status} (Score: {score})")
        console.print(Panel(Markdown(result.get("final_answer", "")), title="Final Verified Answer", border_style="blue"))
        
    except asyncio.CancelledError:
        console.print("\n[yellow]Query aborted.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Error:[/bold red] {e}")
