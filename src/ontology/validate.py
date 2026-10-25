"""
src/ontology/validate.py
------------------------
Standalone validation script for the CS ontology JSON files.

Usage:
    python -m src.ontology.validate
    python -m src.ontology.validate --dir path/to/ontology

Checks performed:
  1. JSON is well-formed.
  2. All required keys are present in every entry.
  3. No duplicate entry IDs within or across files.
  4. All IDs in ``related_ids`` resolve to a known entry.
  5. ``ambiguous`` field is a boolean.
  6. Prints a per-category summary table and overall stats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_KEYS = {"id", "term", "aliases", "label", "category", "ambiguous",
                 "definition", "uri"}


def _load_all(ontology_dir: str) -> dict[str, tuple[str, dict]]:
    """Returns {entry_id: (source_filename, entry_dict)}."""
    all_entries: dict[str, tuple[str, dict]] = {}
    errors: list[str] = []

    for fname in sorted(os.listdir(ontology_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(ontology_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            errors.append(f"  ✗ {fname} – invalid JSON: {exc}")
            continue

        for entry in data.get("entries", []):
            eid = entry.get("id", "").strip()
            if not eid:
                errors.append(f"  ✗ {fname} – entry without 'id': {entry}")
                continue
            if eid in all_entries:
                prev_file = all_entries[eid][0]
                errors.append(f"  ✗ Duplicate ID '{eid}' in {fname} (already in {prev_file})")
            else:
                all_entries[eid] = (fname, entry)

    return all_entries, errors


def validate(ontology_dir: str) -> bool:
    """Run all validation checks. Returns True if no errors found."""
    if not os.path.isdir(ontology_dir):
        print(f"✗ Directory not found: {ontology_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"  CS Ontology Validator")
    print(f"  Directory: {ontology_dir}")
    print(f"{'='*60}\n")

    all_entries, load_errors = _load_all(ontology_dir)
    errors: list[str] = list(load_errors)

    # ── Per-entry validation ──────────────────────────────────────────────
    category_counts: dict[str, int] = defaultdict(int)
    ambiguous_count  = 0
    alias_total      = 0

    for eid, (fname, entry) in all_entries.items():
        # Required keys
        missing = REQUIRED_KEYS - set(entry.keys())
        if missing:
            errors.append(f"  ✗ [{fname}] '{eid}' missing keys: {missing}")

        # Ambiguous must be bool
        if not isinstance(entry.get("ambiguous"), bool):
            errors.append(f"  ✗ [{fname}] '{eid}'.ambiguous must be a bool")

        # Related IDs must resolve
        for rid in entry.get("related_ids", []):
            if rid not in all_entries:
                errors.append(f"  ✗ [{fname}] '{eid}' has unresolved related_id: '{rid}'")

        # Stats
        cat = entry.get("category", "Unknown")
        top_cat = cat.split(">")[0].strip()
        category_counts[top_cat] += 1
        if entry.get("ambiguous"):
            ambiguous_count += 1
        alias_total += len(entry.get("aliases", []))

    # ── Results ──────────────────────────────────────────────────────────
    if errors:
        print("ERRORS FOUND:")
        for e in errors:
            print(e)
        print()
    else:
        print("✓ No errors found.\n")

    # ── Summary table ─────────────────────────────────────────────────────
    print("Category Summary:")
    print(f"  {'Category':<40} {'Entries':>7}")
    print(f"  {'-'*40} {'-'*7}")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:<40} {count:>7}")
    print(f"  {'─'*40} {'─'*7}")
    print(f"  {'TOTAL':<40} {len(all_entries):>7}")

    print(f"\nStats:")
    print(f"  Total entries   : {len(all_entries)}")
    print(f"  Ambiguous terms : {ambiguous_count}")
    print(f"  Total aliases   : {alias_total}")
    print(f"  Files loaded    : {len(set(fname for fname, _ in all_entries.values()))}")
    print()

    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the CS ontology JSON files.")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parents[2] / "dictionaries" / "ontology"),
        help="Path to the ontology directory (default: ../../dictionaries/ontology)",
    )
    args = parser.parse_args()
    ok = validate(args.dir)
    sys.exit(0 if ok else 1)
