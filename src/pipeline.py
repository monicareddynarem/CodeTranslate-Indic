import spacy


def setup_ner_pipeline(patterns):
    """
    Creates a spaCy NLP pipeline with a custom EntityRuler
    using the provided patterns from the CS lexicon.
    """
    # Keep the default NER off to prevent the noisy ORG/PERSON tags
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])

    # LOWER for reliable, case-insensitive matching
    config = {
        "phrase_matcher_attr": "LOWER",
        "overwrite_ents": True
    }

    ruler = nlp.add_pipe("entity_ruler", config=config)
    ruler.add_patterns(patterns)

    return nlp