"""
Logging Configuration
=====================
Structured logging setup for the fraud platform.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, 'transaction_id'):
            log_data['transaction_id'] = record.transaction_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'latency_ms'):
            log_data['latency_ms'] = record.latency_ms
        if hasattr(record, 'fraud_score'):
            log_data['fraud_score'] = record.fraud_score
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} | {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


def setup_logging(level: str = "INFO", format: str = "console") -> None:
    """
    Setup application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ('json' or 'console')
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())
    
    root_logger.addHandler(handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Default logger instance
logger = logging.getLogger("fraud_platform")


class TransactionLogger:
    """Specialized logger for transaction scoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("fraud_platform.scoring")
    
    def log_score(
        self,
        transaction_id: str,
        user_id: str,
        fraud_score: float,
        decision: str,
        latency_ms: float,
    ):
        """Log a transaction score."""
        extra = {
            'transaction_id': transaction_id,
            'user_id': user_id,
            'fraud_score': fraud_score,
            'latency_ms': latency_ms,
        }
        
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"Transaction {transaction_id}: score={fraud_score:.3f}, decision={decision}, latency={latency_ms:.1f}ms",
            args=(),
            exc_info=None,
        )
        
        for key, value in extra.items():
            setattr(record, key, value)
        
        self.logger.handle(record)
    
    def log_error(
        self,
        transaction_id: str,
        error: str,
    ):
        """Log a scoring error."""
        self.logger.error(f"Scoring error for {transaction_id}: {error}")


# Global transaction logger
tx_logger = TransactionLogger()
