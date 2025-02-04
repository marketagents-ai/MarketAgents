import logging
from typing import Any, List
import json
import pyfiglet
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich.align import Align
from rich.box import HEAVY

console = Console(force_terminal=True, color_system="auto")

def print_ascii_art():
    ascii_art = pyfiglet.figlet_format("MARKET AGENTS", font="slant")
    console.print(f"[cyan]{ascii_art}[/cyan]")

def setup_logger(name: str = "MarketSimulation", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add RichHandler to log to console with rich formatting and enable markup
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True  # Enable markup interpretation
    )
    logger.addHandler(handler)
    
    # Prevent propagation to avoid double logging
    logger.propagate = False
    
    return logger

# Create a single, centralized logger instance
orchestration_logger = setup_logger()

def json_to_markdown(data: dict) -> str:
    """Convert a JSON object to markdown format"""
    markdown = ""
    for key, value in data.items():
        markdown += f"### {key.title()}\n"
        if isinstance(value, list):
            for item in value:
                markdown += f"- {item}\n"
        else:
            markdown += f"{value}\n"
        markdown += "\n"
    return markdown

def log_persona(logger: logging.Logger, agent_id: int, persona: str):
    header = f"[bold yellow]🎭 Agent {agent_id} persona:[/bold yellow]\n"
    text = Text.from_markup(header)
    text.append(persona)
    panel = Panel(
        Align.left(text),
        border_style="yellow",
        box=HEAVY,
        width=80
    )
    console.print(panel)

def log_perception(logger: logging.Logger, agent_id: int, perception: str):
    try:
        perception_dict = json.loads(perception) if isinstance(perception, str) else perception
        markdown = json_to_markdown(perception_dict)
        header = f"[bold cyan]👁️  Agent {agent_id:02d} perceives:[/bold cyan]\n"
        text = Text.from_markup(header)
        text.append(markdown)
        panel = Panel(
            Align.left(text),
            border_style="cyan",
            box=HEAVY,
            width=80
        )
        console.print(panel)
    except Exception as e:
        console.print(f"[bold blue]👁️ Agent {agent_id} perceives:[/bold blue]\n[cyan]{perception}[/cyan]")

def log_reflection(logger: logging.Logger, agent_id: int, reflection: str):
    try:
        reflection_dict = json.loads(reflection) if isinstance(reflection, str) else reflection
        markdown = json_to_markdown(reflection_dict)
        header = f"[bold magenta]💭 Agent {agent_id:02d} reflects:[/bold magenta]\n"
        text = Text.from_markup(header)
        text.append(markdown)
        panel = Panel(
            Align.left(text),
            border_style="magenta",
            box=HEAVY,
            width=80
        )
        console.print(panel)
    except Exception as e:
        console.print(f"[bold magenta]💭 Agent {agent_id} reflects:[/bold magenta]\n[magenta]{reflection}[/magenta]")

def log_action(logger: logging.Logger, agent_id: int, action: Any):
    """Log an agent's action in a consistent format using rich formatting"""
    try:
        action_dict = json.loads(action) if isinstance(action, str) else action
        markdown = json_to_markdown(action_dict)
        header = f"[bold green]🎯 Agent {agent_id:02d} action:[/bold green]\n"
        text = Text.from_markup(header)
        text.append(markdown)
        panel = Panel(
            Align.left(text),
            border_style="green",
            box=HEAVY,
            width=80
        )
        console.print(panel)
    except Exception as e:
        console.print(f"[bold green]🎯 Agent {agent_id} action:[/bold green]\n[green]{action}[/green]")

def log_section(logger: logging.Logger, message: str):
    border = "=" * 70
    logger.info(f"[magenta]{border}[/magenta]")
    logger.info(f"[yellow]🔥 {message.upper()} 🔥[/yellow]")
    logger.info(f"[magenta]{border}[/magenta]")

def log_round(logger: logging.Logger, round_num: int, environment_name: str):
    logger.info(f"[green]🔔 ROUND {round_num:02d} BEGINS 🔔[/green]")
    logger.info(f"[cyan]🎲 Environment: {environment_name}. Let the market dynamics unfold! 🎲[/cyan]")

def log_agent_init(logger: logging.Logger, agent_id: int, is_buyer: bool, persona):
    agent_type = "🛒 Buyer" if is_buyer else "💼 Seller"
    trader_type = " | ".join(persona.trader_type)
    logger.info(f"[blue]🤖 Agent {agent_id:02d} | {agent_type} | {trader_type} | Initialized[/blue]")

def log_environment_setup(logger: logging.Logger, env_name: str):
    logger.info(f"[green]🏛️ Entering the {env_name.upper()} ECOSYSTEM 🏛️[/green]")
    logger.info(f"[yellow]📈 Where market forces shape destinies 📉[/yellow]")

def log_completion(logger: logging.Logger, message: str):
    logger.info(f"[green]🎉 {message} 🚀[/green]")

def log_skipped(logger: logging.Logger, message: str):
    logger.info(f"[red]⏭️ {message} (Unexpected market shift!)[/red]")

def log_running(logger: logging.Logger, env_name: str):
    logger.info(f"[green]🏁 The {env_name} market is now ACTIVE 🏁[/green]")
    logger.info(f"[yellow]💥 Prepare for economic disruption! 💥[/yellow]")

def log_raw_action(logger: logging.Logger, agent_id: int, action: dict):
    logger.info(f"[black on yellow]🔧 Agent {agent_id} executes: [/black on yellow]")
    logger.info(f"[yellow]{action}[/yellow]")

def log_market_update(logger: logging.Logger, update: str):
    logger.info(f"[black on cyan]📢 MARKET INSIGHT:[/black on cyan]")
    logger.info(f"[cyan]{update}[/cyan]")

def log_trade(logger: logging.Logger, buyer_id: int, seller_id: int, item: str, price: float):
    logger.info(f"[black on green]💰 TRANSACTION ALERT 💰[/black on green]")
    logger.info(f"[green]🤝 Agent {buyer_id:02d} acquires {item} from Agent {seller_id:02d} at ${price:.2f}[/green]")

def log_leaderboard(logger: logging.Logger, rankings: list):
    header = f"[bold black on yellow]🏆 PERFORMANCE RANKINGS 🏆[/bold black on yellow]"
    content = ""
    for rank, (agent_id, score) in enumerate(rankings, 1):
        indicator = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else "  "
        color = ["yellow", "white", "red"][rank-1] if rank <= 3 else "blue"
        content += f"[{color}]{indicator} #{rank}: Agent {agent_id:02d} - ${score:.2f}[/{color}]\n"
    text = Text.from_markup(content)
    panel = Panel(
        Align.left(text),
        title=header,
        title_align="left",
        border_style="yellow",
        box=HEAVY,
        width=80
    )
    console.print(panel)

def log_topic_proposal(logger: logging.Logger, cohort_id: str, proposer_id: int, topic: str):
    header = f"[bold white on blue]📢 TOPIC PROPOSAL - {cohort_id.upper()} 📢[/bold white on blue]"
    proposer_info = f"[bold]🎯 Proposer: Agent {proposer_id:02d}[/bold]"
    topic_info = f"[cyan]💬 Topic: {topic}[/cyan]"
    text = Text.from_markup(f"{proposer_info}\n\n{topic_info}")
    panel = Panel(
        Align.left(text),
        border_style="blue",
        box=HEAVY,
        width=80,
        title=header,
        title_align="left"
    )
    console.print(panel)

def log_group_message(logger: logging.Logger, cohort_id: str, agent_id: int, message: str, sub_round: int):
    agent_colors = [
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "red",
        "white"
    ]
    color = agent_colors[agent_id % len(agent_colors)]
    header = f"[bold black on white]💬 {cohort_id.upper()} - Round {sub_round}[/bold black on white]"
    agent_info = f"[bold {color}]🤖 Agent {agent_id:02d} says:[/bold {color}]"
    text = Text.from_markup(f"{agent_info}\n\n{message}")
    panel = Panel(
        Align.left(text),
        border_style=color,
        box=HEAVY,
        width=80,
        title=header,
        title_align="left"
    )
    console.print(panel)

def log_cohort_formation(logger: logging.Logger, cohort_id: str, agent_indices: List[int]):
    logger.info(f"[black on green]🎯 COHORT FORMATION 🎯[/black on green]")
    logger.info(f"[green]📋 {cohort_id.upper()}: Agents {agent_indices}[/green]")
    logger.info(f"[green]{'─' * 50}[/green]")

def log_sub_round_start(logger: logging.Logger, cohort_id: str, sub_round: int):
    logger.info(f"[black on yellow]🔄 SUB-ROUND {sub_round} - {cohort_id.upper()} 🔄[/black on yellow]")
    logger.info(f"[yellow]{'─' * 50}[/yellow]")

def log_group_chat_summary(logger: logging.Logger, cohort_id: str, messages_count: int, topic: str):
    header = f"[bold white on magenta]📊 GROUP CHAT SUMMARY - {cohort_id.upper()} 📊[/bold white on magenta]"
    content = f"[magenta]📝 Total Messages: {messages_count}\n\n💭 Topic Discussed: {topic}[/magenta]"
    text = Text.from_markup(content)
    panel = Panel(
        Align.left(text),
        border_style="magenta",
        box=HEAVY,
        width=80,
        title=header,
        title_align="left"
    )
    console.print(panel)

# Example usage:

if __name__ == "__main__":
    print_ascii_art()
