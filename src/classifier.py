"""
classifier.py
--------------
Takes a spaCy doc with detected entities and splits them into two categories:
1. must_preserve  - Unambiguous CS terms that should ALWAYS stay in English
2. context_dependent - Ambiguous terms where the LLM must decide based on context
"""


def classify_entities(doc, ambiguous_terms):
    """
    Classifies detected entities into must-preserve and context-dependent.
    
    Args:
        doc: spaCy Doc with entities detected by the EntityRuler
        ambiguous_terms: set of standardized term names that are ambiguous
        
    Returns:
        must_preserve: list of unique (standardized_term, label) tuples
        context_dependent: list of unique (standardized_term, label, original_text) tuples
    """
    seen_preserve = set()
    seen_context = set()
    must_preserve = []
    context_dependent = []

    for ent in doc.ents:
        # Use the standardized ID if available, otherwise use the matched text
        standardized = ent.ent_id_ if ent.ent_id_ else ent.text.lower()
        label = ent.label_

        if standardized in ambiguous_terms:
            # This term has an everyday meaning — the LLM should decide
            key = (standardized, label)
            if key not in seen_context:
                seen_context.add(key)
                context_dependent.append({
                    "term": standardized,
                    "label": label,
                    "example": ent.text  # one example of how it appeared
                })
        else:
            # This term is unambiguously CS — always keep in English
            key = (standardized, label)
            if key not in seen_preserve:
                seen_preserve.add(key)
                must_preserve.append({
                    "term": standardized,
                    "label": label,
                    "example": ent.text
                })

    return must_preserve, context_dependent


def format_term_lists(must_preserve, context_dependent):
    """
    Formats the classified term lists into readable strings for the LLM prompt.
    
    Returns:
        preserve_str: formatted string of must-preserve terms
        context_str: formatted string of context-dependent terms
    """
    if must_preserve:
        preserve_lines = [f"  - {t['example']} ({t['label']})" for t in must_preserve]
        preserve_str = "\n".join(preserve_lines)
    else:
        preserve_str = "  (none detected)"

    if context_dependent:
        context_lines = [f"  - {t['example']} ({t['label']}) — has everyday meaning, decide from context" 
                        for t in context_dependent]
        context_str = "\n".join(context_lines)
    else:
        context_str = "  (none detected)"

    return preserve_str, context_str
