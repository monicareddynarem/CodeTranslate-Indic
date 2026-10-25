"""
src/ontology/__init__.py
------------------------
Package init – exposes the public loader so callers can do:

    from src.ontology import load_ontology
"""

from .loader import load_ontology

__all__ = ["load_ontology"]
