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
    logger.info(f"🔍 {message.upper()} 🔍")
    logger.info(f"{border}")

def log_search_start(logger: logging.Logger, query: str):
    logger.info(f"🚀 STARTING SEARCH: '{query}'")
    logger.info("🌐 Initiating web research process...")

def log_query_progress(logger: logging.Logger, current: int, total: int, query: str):
    logger.info(f"📊 Processing Query {current}/{total}: '{query}'")

def log_url_extraction(logger: logging.Logger, url: str, method: str, status: str):
    emoji = "✅" if status == "success" else "❌"
    logger.info(f"{emoji} [{method}] {url}")

def log_content_analysis(logger: logging.Logger, url: str, content_type: str):
    logger.info(f"📑 Analyzing content from: {url}")
    logger.info(f"📌 Content type: {content_type}")

def log_ai_summary(logger: logging.Logger, url: str):
    logger.info(f"🤖 Generating AI summary for: {url}")

def log_extraction_stats(logger: logging.Logger, total: int, successful: int, failed: int):
    logger.info(f"""
    📈 Extraction Statistics:
    - Total URLs: {total}
    - Successful: {successful} ✅
    - Failed: {failed} ❌
    """)

def log_db_operation(logger: logging.Logger, operation: str, status: str, details: str = ""):
    emoji = "✅" if status == "success" else "❌"
    logger.info(f"🗄️ Database {operation}: {emoji} {details}")

def log_completion(logger: logging.Logger, message: str):
    logger.info(f"🎉 {message} ✨")

def log_processing_time(logger: logging.Logger, operation: str, time_taken: float):
    logger.info(f"⏱️ {operation} completed in {time_taken:.2f} seconds")

def log_summary_stats(logger: logging.Logger, stats: dict):
    logger.info("""
    📊 SUMMARY STATISTICS 📊
    """)
    for key, value in stats.items():
        logger.info(f"- {key}: {value}")