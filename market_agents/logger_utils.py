import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_section(logger: logging.Logger, message: str):
    border = "======================================"
    logger.info(f"{border}")
    logger.info(f"🔥 {message.upper()} 🔥")
    logger.info(f"{border}")

def log_round(logger: logging.Logger, round_num: int):
    logger.info(f"🔔 ROUND {round_num:02d} BEGINS 🔔")
    logger.info("🎲 Let the market dynamics unfold! 🎲")

def log_agent_init(logger: logging.Logger, agent_id: int, is_buyer: bool, persona):
    agent_type = "🛒 Buyer" if is_buyer else "💼 Seller"
    trader_type = " | ".join(persona.trader_type)
    
    logger.info(f"🤖 Agent {agent_id:02d} | {agent_type} | {trader_type} | Initialized")

def log_environment_setup(logger: logging.Logger, env_name: str):
    logger.info(f"🏛️ Entering the {env_name.upper()} ECOSYSTEM 🏛️")
    logger.info("📈 Where market forces shape destinies 📉")

def log_completion(logger: logging.Logger, message: str):
    logger.info(f"🎉 {message} 🚀")

def log_skipped(logger: logging.Logger, message: str):
    logger.info(f"⏭️ {message} (Unexpected market shift!)")

def log_running(logger: logging.Logger, env_name: str):
    logger.info(f"🏁 The {env_name} market is now ACTIVE 🏁")
    logger.info("💥 Prepare for economic disruption! 💥")

def log_perception(logger: logging.Logger, agent_id: int, perception: str):
    logger.info(f"👁️ Agent {agent_id} perceives: {perception}")

def log_raw_action(logger: logging.Logger, agent_id: int, action: dict):
    logger.info(f"🔧 Agent {agent_id} executes: {action}")

def log_action(logger: logging.Logger, agent_id: int, action: str):
    if "Bid" in action:
        emoji = "💰"
    elif "Ask" in action:
        emoji = "💵"
    elif "reflects" in action.lower():
        emoji = "💭"
    elif "perceives" in action.lower():
        emoji = "👁️"
    else:
        emoji = "🔧"
    logger.info(f"{emoji} Agent {agent_id:02d} executes: {action}")

def log_market_update(logger: logging.Logger, update: str):
    logger.info(f"📢 MARKET INSIGHT: {update}")

def log_reflection(logger: logging.Logger, agent_id: int, reflection: str):
    logger.info(f"💭 Agent {agent_id:02d} reflects: '{reflection}'")

def log_trade(logger: logging.Logger, buyer_id: int, seller_id: int, item: str, price: float):
    logger.info(f"💰 TRANSACTION ALERT 💰")
    logger.info(f"🤝 Agent {buyer_id:02d} acquires {item} from Agent {seller_id:02d} at ${price:.2f}")

def log_leaderboard(logger: logging.Logger, rankings: list):
    logger.info("🏆 PERFORMANCE RANKINGS 🏆")
    for rank, (agent_id, score) in enumerate(rankings, 1):
        indicator = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else "  "
        logger.info(f"{indicator} #{rank}: Agent {agent_id:02d} - ${score:.2f}")
