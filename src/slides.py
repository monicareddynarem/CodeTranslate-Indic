"""
src/slides.py
-------------
PPTX slide processing for the CodeTranslate‑Indic pipeline.

Workflow:
  1. extract_slide_texts(pptx_bytes)
       → list of slides, each slide is a list of (shape_index, text) pairs.
  2. build_translation_input(slide_texts)
       → a single English string with ---SLIDE--- and ---BOX--- markers.
  3. split_translation_output(translated_str, slide_texts)
       → re‑aligns the translated segments back to (slide_idx, shape_idx, text).
  4. build_translated_pptx(pptx_bytes, replacements)
       → bytes of the new PPTX with the same layout, only text replaced.

Rules:
  - Any text box whose font is monospaced (Courier, Consolas, Monaco, Lucida Console)
    is treated as a code block and is NEVER sent for translation.
  - Empty shapes and shapes with only whitespace are skipped.
  - All other shapes have their text replaced run‑by‑run to preserve bold/italic/size.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from pptx import Presentation
from pptx.util import Pt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SLIDE_MARKER = "===SLIDE==="
_BOX_MARKER   = "===BOX==="

# Font families treated as code (case‑insensitive substring match)
_CODE_FONTS = {"courier", "consolas", "monaco", "lucida console", "source code",
               "fira code", "jetbrains", "inconsolata", "hack", "cascadia"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_code_shape(shape) -> bool:
    """Return True if the shape looks like a code block."""
    if not shape.has_text_frame:
        return False
    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            if run.font.name:
                name_lower = run.font.name.lower()
                if any(cf in name_lower for cf in _CODE_FONTS):
                    return True
    return False


def _shape_plain_text(shape) -> str:
    """Return the full text of a shape, preserving newlines between paragraphs."""
    if not shape.has_text_frame:
        return ""
    return "\n".join(
        para.text for para in shape.text_frame.paragraphs
    )


# ---------------------------------------------------------------------------
# Step 1 – Extract translatable text from each slide
# ---------------------------------------------------------------------------

def extract_slide_texts(
    pptx_bytes: bytes,
) -> list[list[dict[str, Any]]]:
    """Parse a PPTX file and return per‑slide, per‑shape text info.

    Returns:
        A list where each element corresponds to one slide and contains a list
        of dicts:
            {
                "shape_idx": int,      # index into slide.shapes
                "text":      str,      # plain text (paragraphs joined by \\n)
                "is_code":   bool,     # True → do NOT translate
            }
        Only shapes that have_text_frame and non-empty text are included.
    """
    prs  = Presentation(BytesIO(pptx_bytes))
    result: list[list[dict[str, Any]]] = []

    for slide in prs.slides:
        slide_boxes: list[dict[str, Any]] = []
        for idx, shape in enumerate(slide.shapes):
            if not shape.has_text_frame:
                continue
            text = _shape_plain_text(shape).strip()
            if not text:
                continue
            slide_boxes.append({
                "shape_idx": idx,
                "text":      text,
                "is_code":   _is_code_shape(shape),
            })
        result.append(slide_boxes)

    return result


# ---------------------------------------------------------------------------
# Step 2 – Build a single string to feed the translator
# ---------------------------------------------------------------------------

def build_translation_input(slide_texts: list[list[dict[str, Any]]]) -> str:
    """Serialise all translatable (non‑code) texts into one marked‑up string.

    The translator receives a single contiguous block so it can maintain
    linguistic consistency across boxes and slides.

    Format:
        ===SLIDE===
        ===BOX===
        <text of box 0>
        ===BOX===
        <text of box 1>
        ===SLIDE===
        ...

    Code boxes are represented by an empty ===BOX=== (so their slot is kept
    but no text is sent for translation).
    """
    parts: list[str] = []
    for slide_boxes in slide_texts:
        parts.append(_SLIDE_MARKER)
        for box in slide_boxes:
            parts.append(_BOX_MARKER)
            if not box["is_code"]:
                parts.append(box["text"])
            # code boxes → emit empty string so the slot is still present

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 3 – Split translation output back to per‑shape segments
# ---------------------------------------------------------------------------

def split_translation_output(
    translated_str: str,
    slide_texts: list[list[dict[str, Any]]],
) -> list[list[str]]:
    """Re‑align translated text back to (slide, box) positions.

    Returns:
        A list of slides; each slide is a list of translated strings aligned
        1‑to‑1 with slide_texts[slide_idx].  Code boxes keep their original
        text.
    """
    # Split by SLIDE marker (drop first empty segment before the first marker)
    slide_chunks = re.split(re.escape(_SLIDE_MARKER), translated_str)
    # Filter empty
    slide_chunks = [s for s in slide_chunks if s.strip()]

    result: list[list[str]] = []

    for slide_idx, slide_boxes in enumerate(slide_texts):
        translated_slide: list[str] = []

        if slide_idx < len(slide_chunks):
            # Split slide chunk by BOX marker
            box_chunks = re.split(re.escape(_BOX_MARKER), slide_chunks[slide_idx])
            box_chunks = [b.strip() for b in box_chunks if b.strip() != "" or True]
            # Remove the first empty segment before first BOX marker
            box_chunks = [b for b in box_chunks]
            # drop initial empty string
            if box_chunks and box_chunks[0] == "":
                box_chunks = box_chunks[1:]
        else:
            box_chunks = []

        for box_idx, box in enumerate(slide_boxes):
            if box["is_code"]:
                # Keep original code text unchanged
                translated_slide.append(box["text"])
            else:
                if box_idx < len(box_chunks):
                    translated_text = box_chunks[box_idx].strip()
                    translated_slide.append(translated_text if translated_text else box["text"])
                else:
                    # Fallback: keep original if translator dropped the box
                    translated_slide.append(box["text"])

        result.append(translated_slide)

    return result


# ---------------------------------------------------------------------------
# Step 4 – Write the translated PPTX
# ---------------------------------------------------------------------------

def _replace_text_in_shape(shape, new_text: str) -> None:
    """Replace shape text while preserving run‑level formatting.

    Strategy:
      - Put all new text into the first run of the first paragraph.
      - Clear all other runs and paragraphs to avoid duplication.
      - Paragraph‑level and run‑level formatting (bold, italic, font size,
        color) of the *first* run is kept.
    """
    if not shape.has_text_frame:
        return

    tf = shape.text_frame
    # Split new_text on newlines → one paragraph each
    new_lines = new_text.split("\n")

    # Preserve formatting from first run of first paragraph
    first_para = tf.paragraphs[0]
    ref_run = first_para.runs[0] if first_para.runs else None

    # Clear all existing paragraphs' text
    # We manipulate the XML directly for reliability
    from pptx.oxml.ns import qn
    from lxml import etree

    txBody = tf._txBody

    # Remove all existing <a:p> elements
    for p_elem in txBody.findall(qn("a:p")):
        txBody.remove(p_elem)

    # Re-create paragraphs
    for line_idx, line in enumerate(new_lines):
        # Clone the first paragraph element as a template for formatting
        if line_idx == 0 and ref_run is not None:
            # Re‑use original first paragraph's XML as template
            p_elem = etree.SubElement(txBody, qn("a:p"))
            # Copy paragraph properties if they existed
            if first_para._p.find(qn("a:pPr")) is not None:
                pPr = first_para._p.find(qn("a:pPr"))
                p_elem.append(etree.fromstring(etree.tostring(pPr)))
            # Create a run element
            r_elem = etree.SubElement(p_elem, qn("a:r"))
            # Copy run properties from original first run
            if ref_run._r.find(qn("a:rPr")) is not None:
                rPr = ref_run._r.find(qn("a:rPr"))
                r_elem.append(etree.fromstring(etree.tostring(rPr)))
            # Set text
            t_elem = etree.SubElement(r_elem, qn("a:t"))
            t_elem.text = line
        else:
            p_elem = etree.SubElement(txBody, qn("a:p"))
            r_elem = etree.SubElement(p_elem, qn("a:r"))
            t_elem = etree.SubElement(r_elem, qn("a:t"))
            t_elem.text = line


def build_translated_pptx(
    pptx_bytes: bytes,
    slide_texts:   list[list[dict[str, Any]]],
    replacements:  list[list[str]],
) -> bytes:
    """Return bytes of a new PPTX with translated text and the same layout.

    Args:
        pptx_bytes:   Original PPTX file bytes.
        slide_texts:  Output of extract_slide_texts (to know which shape to update).
        replacements: Output of split_translation_output (one string per box).
    """
    prs = Presentation(BytesIO(pptx_bytes))

    for slide_idx, (slide, slide_boxes) in enumerate(zip(prs.slides, slide_texts)):
        if slide_idx >= len(replacements):
            break
        slide_replacements = replacements[slide_idx]

        for box_idx, box in enumerate(slide_boxes):
            if box_idx >= len(slide_replacements):
                break
            shape      = slide.shapes[box["shape_idx"]]
            new_text   = slide_replacements[box_idx]
            _replace_text_in_shape(shape, new_text)

    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()
