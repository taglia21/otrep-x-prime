"""
Compliance Logging and Audit Trail
===================================
Structured logging for regulatory compliance and debugging.

Features:
- Time-stamped log files with rotation
- Structured JSON audit events
- Separate streams for trading, risk, and system logs
- Configurable log levels

Author: OTREP-X Development Team
Date: December 2025
"""

import os
import json
import logging
import logging.config
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


class AuditEventType(Enum):
    """Types of audit events for compliance tracking."""
    # Trading events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Risk events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    DAILY_LIMIT_REACHED = "daily_limit_reached"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_LOADED = "config_loaded"
    API_ERROR = "api_error"
    DATA_STALE = "data_stale"


@dataclass
class AuditEvent:
    """Structured audit event for compliance logging."""
    event_type: AuditEventType
    timestamp: str
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None
    signal_value: Optional[float] = None
    order_id: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return json.dumps(data)
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data


def get_logging_config(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True
) -> Dict:
    """
    Generate logging configuration dictionary.
    
    Creates separate log files for:
    - main.log: General application logs
    - trading.log: Trading-specific logs
    - audit.log: Compliance audit trail (JSON format)
    - errors.log: Error-only logs
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        enable_console: Enable console output
        enable_file: Enable file output
        enable_audit: Enable audit trail
        
    Returns:
        logging.config.dictConfig compatible dictionary
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for log files
    date_str = datetime.now().strftime("%Y%m%d")
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": "%(message)s"  # JSON events are pre-formatted
            }
        },
        "handlers": {},
        "loggers": {
            "": {  # Root logger
                "handlers": [],
                "level": log_level,
                "propagate": True
            },
            "otrep.trading": {
                "handlers": [],
                "level": log_level,
                "propagate": False
            },
            "otrep.audit": {
                "handlers": [],
                "level": "INFO",
                "propagate": False
            },
            "otrep.risk": {
                "handlers": [],
                "level": log_level,
                "propagate": False
            }
        }
    }
    
    # Console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
        config["loggers"][""]["handlers"].append("console")
        config["loggers"]["otrep.trading"]["handlers"].append("console")
        config["loggers"]["otrep.risk"]["handlers"].append("console")
    
    # File handlers
    if enable_file:
        # Main log
        config["handlers"]["main_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": os.path.join(log_dir, f"main_{date_str}.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "encoding": "utf-8"
        }
        config["loggers"][""]["handlers"].append("main_file")
        
        # Trading log
        config["handlers"]["trading_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": os.path.join(log_dir, f"trading_{date_str}.log"),
            "maxBytes": 10485760,
            "backupCount": 10,
            "encoding": "utf-8"
        }
        config["loggers"]["otrep.trading"]["handlers"].append("trading_file")
        
        # Error log
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": os.path.join(log_dir, f"errors_{date_str}.log"),
            "maxBytes": 10485760,
            "backupCount": 10,
            "encoding": "utf-8"
        }
        config["loggers"][""]["handlers"].append("error_file")
    
    # Audit trail handler (JSON format)
    if enable_audit:
        config["handlers"]["audit_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": os.path.join(log_dir, f"audit_{date_str}.jsonl"),
            "maxBytes": 52428800,  # 50MB
            "backupCount": 30,  # Keep 30 days
            "encoding": "utf-8"
        }
        config["loggers"]["otrep.audit"]["handlers"].append("audit_file")
    
    return config


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True
) -> None:
    """
    Configure logging for the application.
    
    Call this once at application startup.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        enable_console: Enable console output
        enable_file: Enable file output  
        enable_audit: Enable audit trail
    """
    config = get_logging_config(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_audit=enable_audit
    )
    logging.config.dictConfig(config)


def get_audit_logger() -> logging.Logger:
    """Get the audit trail logger."""
    return logging.getLogger("otrep.audit")


def get_trading_logger() -> logging.Logger:
    """Get the trading logger."""
    return logging.getLogger("otrep.trading")


def get_risk_logger() -> logging.Logger:
    """Get the risk management logger."""
    return logging.getLogger("otrep.risk")


def log_audit_event(event: AuditEvent) -> None:
    """
    Log a structured audit event.
    
    Args:
        event: AuditEvent to log
    """
    audit_logger = get_audit_logger()
    audit_logger.info(event.to_json())


def create_audit_event(
    event_type: AuditEventType,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
    signal_value: Optional[float] = None,
    order_id: Optional[str] = None,
    message: Optional[str] = None,
    **kwargs
) -> AuditEvent:
    """
    Create and log an audit event.
    
    Convenience function that creates the event and logs it.
    
    Args:
        event_type: Type of audit event
        symbol: Stock symbol (optional)
        side: Order side (optional)
        quantity: Order quantity (optional)
        price: Execution price (optional)
        signal_value: Signal value (optional)
        order_id: Order ID (optional)
        message: Additional message (optional)
        **kwargs: Additional metadata
        
    Returns:
        The created AuditEvent
    """
    event = AuditEvent(
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        signal_value=signal_value,
        order_id=order_id,
        message=message,
        metadata=kwargs if kwargs else None
    )
    log_audit_event(event)
    return event
