import os
import sys
from loguru import logger

def setup_logging() -> None:
    """Configures Loguru logger.
    
    Registers a colored stdout handler for the console and a rotating, 
    compressed file handler for persistent logs.
    """
    # Remove standard default logger
    logger.remove()

    # Console output handler (clean & colored)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        backtrace=True,
        diagnose=True
    )

    # Ensure local log folder is ready
    os.makedirs("logs", exist_ok=True)

    # Persistent file output handler
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=False  # Avoid exposing sensitive variables in production logs
    )

    logger.info("Application logging successfully initialized.")
