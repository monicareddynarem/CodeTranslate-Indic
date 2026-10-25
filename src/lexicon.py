"""
src/lexicon.py
--------------
Public interface for the CS ontology.

Loads structured JSON ontology files from dictionaries/ontology/ and
returns:
  - spaCy EntityRuler patterns (for NER)
  - a set of ambiguous entry IDs (for classifier.py)
  - the full entry_map (for prompt enrichment in translator.py)

Legacy plain-text dictionaries are no longer used; all term data lives
in the JSON ontology under dictionaries/ontology/.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.ontology import load_ontology


# Default ontology directory (relative to project root)
_DEFAULT_ONTOLOGY_DIR = str(
    Path(__file__).resolve().parents[1] / "dictionaries" / "ontology"
)


def generate_lexicon(
    dict_folder: str,
) -> tuple[list[dict], set[str], dict[str, dict]]:
    """Load the CS ontology and return NER patterns, ambiguous IDs, and entry map.

    Args:
        dict_folder: Path to the *dictionaries/* directory (the parent of ontology/).

    Returns:
        patterns      – list of spaCy EntityRuler pattern dicts
        ambiguous_ids – set of entry IDs marked as context-dependent
        entry_map     – dict mapping entry ID → full ontology entry (for prompts)
    """
    ontology_dir = os.path.join(dict_folder, "ontology")
    patterns, ambiguous_ids, entry_map = load_ontology(ontology_dir)
    return patterns, ambiguous_ids, entry_map
