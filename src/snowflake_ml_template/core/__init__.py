"""Core framework foundation for Snowflake MLOps.

This module provides the foundational abstractions, base classes, and interfaces
that all other components depend on. It implements the Dependency Inversion
Principle by defining high-level abstractions that concrete implementations
must adhere to.

Submodules:
    base: Abstract base classes and interfaces
    session: Snowflake session management
    config: Configuration loading and validation
    exceptions: Custom exception hierarchy
"""

__all__ = [
    "base",
    "session",
    "config",
    "exceptions",
]
