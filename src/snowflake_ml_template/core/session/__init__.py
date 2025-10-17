"""Session management for Snowflake connections.

This module provides centralized session management with connection pooling,
warehouse switching, and context management capabilities.

Classes:
    SessionManager: Singleton session manager
    SessionContext: Context manager for temporary session changes
"""

from snowflake_ml_template.core.session.context import SessionContext
from snowflake_ml_template.core.session.manager import SessionManager

__all__ = [
    "SessionManager",
    "SessionContext",
]
