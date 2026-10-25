"""
src/ontology/loader.py
----------------------
Loads all JSON ontology files from dictionaries/ontology/ and converts them
into spaCy EntityRuler-compatible pattern dicts plus a set of ambiguous term IDs.

Schema expected per entry in each JSON file:
{
  "id":        str   – unique canonical identifier (e.g. "hash_table")
  "term":      str   – primary surface form (e.g. "Hash Table")
  "aliases":   list  – alternative surface forms
  "label":     str   – spaCy entity label (e.g. "DATA_STRUCTURE")
  "category":  str   – hierarchical category path (e.g. "Data Structures > Tree")
  "ambiguous": bool  – True if the term has a common non-CS meaning
  "definition":str   – one-sentence definition (used for prompt enrichment)
  "related_ids":list – IDs of related ontology entries
  "uri":       str   – canonical Wikipedia or reference URI
}
"""

from __future__ import annotations

import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _patterns_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return spaCy EntityRuler pattern dicts for a single ontology entry."""
    label   = entry["label"]
    term_id = entry["id"]
    patterns: list[dict[str, Any]] = []

    surface_forms = [entry["term"]] + entry.get("aliases", [])
    for surface in surface_forms:
        surface = surface.strip()
        if not surface:
            continue
        patterns.append({
            "label":   label,
            "pattern": surface.lower(),   # LOWER matcher is set in pipeline.py
            "id":      term_id,
        })

    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ontology(ontology_dir: str) -> tuple[list[dict], set[str], dict[str, dict]]:
    """Parse all JSON files in *ontology_dir* and return:

    patterns       – list of spaCy EntityRuler pattern dicts
    ambiguous_ids  – set of entry IDs whose ``ambiguous`` flag is True
    entry_map      – dict mapping entry ID → full entry dict (for prompt enrichment)
    """
    patterns:      list[dict[str, Any]] = []
    ambiguous_ids: set[str]             = set()
    entry_map:     dict[str, dict]      = {}

    if not os.path.isdir(ontology_dir):
        return patterns, ambiguous_ids, entry_map

    for fname in sorted(os.listdir(ontology_dir)):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(ontology_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ontology] WARNING – could not load {fpath}: {exc}")
            continue

        for entry in data.get("entries", []):
            entry_id = entry.get("id", "").strip()
            if not entry_id:
                continue

            # Register the full entry for lookup
            entry_map[entry_id] = entry

            # Collect ambiguous IDs
            if entry.get("ambiguous", False):
                ambiguous_ids.add(entry_id)

            # Build spaCy patterns
            patterns.extend(_patterns_from_entry(entry))

    return patterns, ambiguous_ids, entry_map
