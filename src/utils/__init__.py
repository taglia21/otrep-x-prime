"""
OTREP-X PRIME Common Utilities Module
"""

import time
import logging
from functools import wraps
from typing import Callable, Any

def configure_logging(level: int = logging.INFO) -> None:
    """Initialize standardized logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/system.log'),
            logging.StreamHandler()
        ]
    )

def environment_setup() -> None:
    """Validate and prepare runtime environment"""
    # Add environment validation logic here
    pass

def timed_api_operation(func: Callable) -> Callable:
    """Decorator for timing API operations"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.info(
            f"Operation {func.__name__} completed in {end_time - start_time:.4f} seconds"
        )
        return result
    return wrapper
