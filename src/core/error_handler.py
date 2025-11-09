import traceback
from src.core_logger import logger

def safe_run(fn, *args, **kwargs):
    """
    Wrapper for any function that may raise runtime errors.
    Automatically logs tracebacks and returns None instead of crashing.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"Exception in {fn.__name__}: {e}")
        logger.debug(traceback.format_exc())
        return None
