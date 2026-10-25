"""
classifier.py
-------------
Takes a spaCy Doc with detected entities (from the structured JSON ontology)
and splits them into two categories:

1. must_preserve      – Unambiguous CS terms that ALWAYS stay in English.
2. context_dependent  – Ambiguous terms where the LLM must decide from context.

Each entry now carries extra metadata sourced from the ontology
(definition, category, URI) so the translator prompt can be richer.
"""

from __future__ import annotations


def classify_entities(
    doc,
    ambiguous_ids: set[str],
    entry_map: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Classify detected entities into must-preserve and context-dependent.

    Args:
        doc:           spaCy Doc with entities detected by the EntityRuler.
        ambiguous_ids: Set of entry IDs that are context-dependent (from ontology).
        entry_map:     Optional dict mapping entry ID → full ontology entry.
                       When provided, definitions and categories are included.

    Returns:
        must_preserve:    list of enriched entity dicts (always keep in English).
        context_dependent: list of enriched entity dicts (LLM decides per context).
    """
    entry_map = entry_map or {}

    seen_preserve: set[tuple] = set()
    seen_context:  set[tuple] = set()
    must_preserve:     list[dict] = []
    context_dependent: list[dict] = []

    for ent in doc.ents:
        # Use the canonical ontology ID if available, else fall back to surface text
        entry_id = ent.ent_id_ if ent.ent_id_ else ent.text.lower()
        label    = ent.label_

        # Pull richer metadata from the ontology entry map
        ontology_entry = entry_map.get(entry_id, {})
        definition     = ontology_entry.get("definition", "")
        category       = ontology_entry.get("category", "")
        canonical_term = ontology_entry.get("term", ent.text)

        base = {
            "id":            entry_id,
            "term":          canonical_term,
            "example":       ent.text,      # how it appeared in the transcript
            "label":         label,
            "category":      category,
            "definition":    definition,
        }

        if entry_id in ambiguous_ids:
            key = (entry_id, label)
            if key not in seen_context:
                seen_context.add(key)
                context_dependent.append(base)
        else:
            key = (entry_id, label)
            if key not in seen_preserve:
                seen_preserve.add(key)
                must_preserve.append(base)

    return must_preserve, context_dependent


def format_term_lists(
    must_preserve: list[dict],
    context_dependent: list[dict],
) -> tuple[str, str]:
    """Format classified term lists into enriched strings for the LLM prompt.

    Returns:
        preserve_str: Formatted string of must-preserve terms with category info.
        context_str:  Formatted string of context-dependent terms with definitions.
    """
    if must_preserve:
        lines = []
        for t in must_preserve:
            cat  = f" [{t['category']}]" if t["category"] else ""
            defn = f" — {t['definition']}" if t["definition"] else ""
            lines.append(f"  - {t['term']} ({t['label']}){cat}{defn}")
        preserve_str = "\n".join(lines)
    else:
        preserve_str = "  (none detected)"

    if context_dependent:
        lines = []
        for t in context_dependent:
            cat  = f" [{t['category']}]" if t["category"] else ""
            defn = f" — {t['definition']}" if t["definition"] else ""
            lines.append(
                f"  - {t['term']} ({t['label']}){cat}{defn}"
                f"\n    ↳ has everyday meaning; keep in English only if used as a CS concept"
            )
        context_str = "\n".join(lines)
    else:
        context_str = "  (none detected)"

    return preserve_str, context_str
