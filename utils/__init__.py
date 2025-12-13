"""
Utilities Module
================
Common utilities for OTREP-X PRIME.

Components:
- logger: Compliance logging and audit trail
"""

from .logger import setup_logging, get_audit_logger, AuditEvent

__all__ = ['setup_logging', 'get_audit_logger', 'AuditEvent']
