# BTP CS Translation Pipeline

This repository implements an end‑to‑end pipeline for **translating English computer‑science lecture transcripts** into a target Indic language using Google's Gemini LLM. The pipeline:

1. **Loads a structured CS ontology** (JSON files) to detect technical terms.
2. **Classifies terms** into *must‑preserve* (always English) or *context‑dependent* (potentially transliterated).
3. **Enriches the LLM prompt** with term definitions and categories for higher‑quality translations.
4. **Generates two translation variants** – one keeping technical terms in English, another fully transliterated into the target script.
5. **Optionally creates TTS audio** for the normal‑term output.

---

## Project layout
```
BTP/
├─ dictionaries/
│   └─ ontology/        # JSON ontology files (see "Ontology" section)
├─ src/
│   ├─ ontology/        # loader & validator
│   ├─ lexicon.py       # public interface to load patterns
│   ├─ classifier.py    # term classification & formatting
│   ├─ translator.py    # Gemini prompt builders & API calls
│   ├─ pipeline.py      # spaCy NER pipeline setup
│   ├─ driver.py        # high‑level orchestration (translation + TTS)
│   └─ ...
├─ input/                # sample English lecture .txt files
├─ output/               # generated translations & audio
└─ README.md            # **this file**
```

---

## Ontology
The ontology lives under `dictionaries/ontology/` as a collection of **JSON files**. Each file follows the same schema:

```json
{
  "metadata": {
    "file": "<filename>",
    "category": "<top‑level category>",
    "version": "1.0.0",
    "description": "…",
    "sources": ["…"],
    "maintainer": "BTP Project"
  },
  "entries": [
    {
      "id": "<canonical_id>",
      "term": "<primary surface form>",
      "aliases": ["<alt1>", "<alt2>"],
      "label": "<spaCy entity label>",
      "category": "<category > <sub‑category>",
      "ambiguous": false,
      "definition": "One‑sentence definition used in LLM prompts.",
      "related_ids": ["<other_id>", …],
      "uri": "https://en.wikipedia.org/..."
    }
    // … more entries …
  ]
}
```

* **`id`** – unique identifier used internally.
* **`term` / `aliases`** – surface forms that the EntityRuler will match.
* **`label`** – spaCy entity label (e.g. `ALGORITHM`, `DATA_STRUCTURE`).
* **`category`** – hierarchical path (e.g. `Algorithms > Sorting`).
* **`ambiguous`** – `true` if the word also has a common non‑CS meaning (e.g. *Array*). Those are placed in the *context‑dependent* list.
* **`definition`** – short description injected into the LLM prompt.
* **`related_ids`** – optional cross‑references to other entries.
* **`uri`** – link to an authoritative source.

### Adding or updating entries
1. **Create or edit a JSON file** in `dictionaries/ontology/` using the schema above.
2. **Run the validator** to ensure no duplicate IDs or missing fields:
   ```powershell
   python -m src.ontology.validate
   ```
3. Commit the changes – the loader automatically discovers any new files, so no code changes are required.

---

## Development workflow
1. **Install dependencies** (run in the project root):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Run the full pipeline** (example):
   ```powershell
   python src/driver.py input/lecture.txt --lang hindi
   ```
   This produces `output/translated_hindi.txt` (normal) and `output/translated_hindi_transliterated.txt` plus an audio file.
3. **Validate the ontology** after any changes:
   ```powershell
   python -m src.ontology.validate
   ```

---

## License
This project is provided for educational purposes under the MIT License.

---

*For any questions or contributions, open an issue or submit a pull request.*
