import os


def load_patterns_from_file(filepath, label_name):
    """
    Reads comma-separated lines and assigns the first term as the standard ID.
    Lines ending with '| AMBIGUOUS' mark the term as context-dependent.
    
    Returns:
        patterns: list of dicts for spaCy EntityRuler
        ambiguous_terms: set of standard names that are context-dependent
    """
    patterns = []
    ambiguous_terms = set()

    # If the file doesn't exist yet, just skip it to prevent crashes
    if not os.path.exists(filepath):
        return patterns, ambiguous_terms

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Check for the AMBIGUOUS marker
            is_ambiguous = False
            if line.endswith("| AMBIGUOUS"):
                is_ambiguous = True
                line = line.rsplit("|", 1)[0].strip()

            # Split the line by commas and clean up extra spaces
            parts = [p.strip().lower() for p in line.split(',')]
            if not parts or not parts[0]:
                continue

            # The first term is our Standard Name
            standard_name = parts[0]

            if is_ambiguous:
                ambiguous_terms.add(standard_name)

            # Create a pattern for the standard name itself
            patterns.append({
                "label": label_name,
                "pattern": standard_name,
                "id": standard_name
            })

            # Create patterns for all aliases, linking them to the standard name
            for alias in parts[1:]:
                if alias:
                    patterns.append({
                        "label": label_name,
                        "pattern": alias,
                        "id": standard_name
                    })

    return patterns, ambiguous_terms


def generate_lexicon(dict_folder):
    """
    Maps specific text files to their official Entity Tags.
    
    Returns:
        cs_patterns: list of pattern dicts for spaCy EntityRuler
        ambiguous_terms: set of all terms marked as AMBIGUOUS across all files
    """
    cs_patterns = []
    all_ambiguous = set()

    # A dictionary mapping the filename to the Tag it should receive
    files_to_labels = {
        "cs_domains.txt": "CS_DOMAIN",       # CS subject/field names (never translate)
        "algorithms.txt": "ALGORITHM",
        "data_structures.txt": "DATA_STRUCTURE",
        "os_concepts.txt": "OS_CONCEPT",
        "dbms_concepts.txt": "DBMS_CONCEPT",
        "data_types.txt": "DATA_TYPE",
        "hardware.txt": "HARDWARE"
    }

    for filename, label in files_to_labels.items():
        filepath = os.path.join(dict_folder, filename)
        patterns, ambiguous = load_patterns_from_file(filepath, label)
        cs_patterns.extend(patterns)
        all_ambiguous.update(ambiguous)

    return cs_patterns, all_ambiguous