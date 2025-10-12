"""Helpers to render Dynamic Table and Stream templates for handling late-arriving data."""

from .templates import render_dynamic_table, render_stream_on_table

__all__ = ["render_stream_on_table", "render_dynamic_table"]
