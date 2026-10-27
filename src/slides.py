"""
src/slides.py
-------------
PPTX slide processing for the CodeTranslate-Indic pipeline.

Workflow:
  1. extract_slide_texts(pptx_bytes)
       -> list of slides, each slide is a list of (shape_index, text) pairs.
  2. build_translation_input(slide_texts)
       -> a single English string with ===SLIDE=== and ===BOX=== markers.
  3. split_translation_output(translated_str, slide_texts)
       -> re-aligns the translated segments back to (slide_idx, shape_idx, text).
  4. build_translated_pptx(pptx_bytes, slide_texts, replacements)
       -> bytes of the new PPTX with the same layout, only text replaced.

Rules:
  - Any text box whose font is monospaced (Courier, Consolas, Monaco, etc.)
    is treated as a code block and is NEVER sent for translation.
  - Empty shapes and shapes with only whitespace are skipped.
  - All other shapes have their text replaced while preserving bold/italic/size.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from pptx import Presentation
from pptx.util import Pt  # noqa: F401 – kept for potential future use

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SLIDE_MARKER = "===SLIDE==="
_BOX_MARKER   = "===BOX==="

# Font families treated as code (case-insensitive substring match)
_CODE_FONTS = {
    "courier", "consolas", "monaco", "lucida console",
    "source code", "fira code", "jetbrains", "inconsolata",
    "hack", "cascadia",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Strip characters that are illegal in XML 1.0 text nodes.

    XML 1.0 permits: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
    Null bytes, lone surrogates and other control characters cause lxml to
    raise a ValueError / TypeError when assigned to element.text.
    """
    # Normalise CRLF / CR -> LF so split("\n") works uniformly
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove every character that XML 1.0 does not allow
    return re.sub(r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]", "", text)


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
    return "\n".join(para.text for para in shape.text_frame.paragraphs)


# ---------------------------------------------------------------------------
# Step 1 - Extract translatable text from each slide
# ---------------------------------------------------------------------------

def extract_slide_texts(pptx_bytes: bytes) -> list[list[dict[str, Any]]]:
    """Parse a PPTX file and return per-slide, per-shape text info.

    Returns:
        A list where each element corresponds to one slide and contains a list
        of dicts:
            {
                "shape_idx": int,   # index into slide.shapes
                "text":      str,   # plain text (paragraphs joined by newline)
                "is_code":   bool,  # True -> do NOT translate
            }
        Only shapes that have_text_frame and non-empty text are included.
    """
    prs = Presentation(BytesIO(pptx_bytes))
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
# Step 2 - Build a single string to feed the translator
# ---------------------------------------------------------------------------

def build_translation_input(slide_texts: list[list[dict[str, Any]]]) -> str:
    """Serialise all translatable (non-code) texts into one marked-up string.

    Format:
        ===SLIDE===
        ===BOX===
        <text of box 0>
        ===BOX===
        <text of box 1>
        ===SLIDE===
        ...

    Code boxes emit an empty ===BOX=== slot so their position is preserved
    and the slot count stays aligned with slide_texts.
    """
    parts: list[str] = []
    for slide_boxes in slide_texts:
        parts.append(_SLIDE_MARKER)
        for box in slide_boxes:
            parts.append(_BOX_MARKER)
            if not box["is_code"]:
                parts.append(box["text"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 3 - Split translation output back to per-shape segments
# ---------------------------------------------------------------------------

def split_translation_output(
    translated_str: str,
    slide_texts: list[list[dict[str, Any]]],
) -> list[list[str]]:
    """Re-align translated text back to (slide, box) positions.

    Returns:
        A list of slides; each slide is a list of translated strings aligned
        1-to-1 with slide_texts[slide_idx]. Code boxes keep their original text.
    """
    slide_chunks = re.split(re.escape(_SLIDE_MARKER), translated_str)
    slide_chunks = [s for s in slide_chunks if s.strip()]

    result: list[list[str]] = []

    for slide_idx, slide_boxes in enumerate(slide_texts):
        translated_slide: list[str] = []

        if slide_idx < len(slide_chunks):
            box_chunks = re.split(re.escape(_BOX_MARKER), slide_chunks[slide_idx])
            box_chunks = [b.strip() for b in box_chunks]
            # Drop the leading empty segment that appears before the first BOX marker
            if box_chunks and box_chunks[0] == "":
                box_chunks = box_chunks[1:]
        else:
            box_chunks = []

        for box_idx, box in enumerate(slide_boxes):
            if box["is_code"]:
                translated_slide.append(box["text"])
            else:
                if box_idx < len(box_chunks) and box_chunks[box_idx]:
                    translated_slide.append(box_chunks[box_idx])
                else:
                    # Fallback: keep original if translator dropped the box
                    translated_slide.append(box["text"])

        result.append(translated_slide)

    return result


# ---------------------------------------------------------------------------
# Step 4 - Write the translated PPTX
# ---------------------------------------------------------------------------

def _replace_text_in_shape(shape, new_text: str) -> None:
    """Replace shape text while preserving run-level formatting.

    Strategy:
      - Sanitize the incoming text to strip XML-illegal characters (null bytes,
        lone surrogates, etc.) which cause lxml to raise an error on assignment.
      - Rebuild paragraphs one per line, copying the formatting of the original
        first run (bold, italic, font size, color) onto the first new paragraph.
    """
    if not shape.has_text_frame:
        return

    from pptx.oxml.ns import qn
    from lxml import etree

    tf = shape.text_frame

    # Sanitize FIRST - removes null bytes and other XML-illegal characters
    new_text = _sanitize(new_text)
    new_lines = new_text.split("\n") if new_text else [""]

    # Capture formatting reference before clearing anything
    first_para = tf.paragraphs[0]
    ref_run    = first_para.runs[0] if first_para.runs else None

    # Snapshot paragraph and run properties as serialised XML strings
    pPr_xml = (
        etree.tostring(first_para._p.find(qn("a:pPr")))
        if first_para._p.find(qn("a:pPr")) is not None
        else None
    )
    rPr_xml = (
        etree.tostring(ref_run._r.find(qn("a:rPr")))
        if (ref_run is not None and ref_run._r.find(qn("a:rPr")) is not None)
        else None
    )

    txBody = tf._txBody

    # Remove all existing <a:p> elements
    for p_elem in txBody.findall(qn("a:p")):
        txBody.remove(p_elem)

    # Re-create one paragraph per line
    for line_idx, line in enumerate(new_lines):
        # Extra per-line sanitize as a safety net
        line = _sanitize(line)

        p_elem = etree.SubElement(txBody, qn("a:p"))

        # First line inherits the original paragraph/run properties
        if line_idx == 0 and ref_run is not None:
            if pPr_xml is not None:
                p_elem.append(etree.fromstring(pPr_xml))

        r_elem = etree.SubElement(p_elem, qn("a:r"))

        if line_idx == 0 and rPr_xml is not None:
            r_elem.append(etree.fromstring(rPr_xml))

        t_elem          = etree.SubElement(r_elem, qn("a:t"))
        t_elem.text     = line   # safe: _sanitize already stripped illegal chars


def build_translated_pptx(
    pptx_bytes:   bytes,
    slide_texts:  list[list[dict[str, Any]]],
    replacements: list[list[str]],
) -> bytes:
    """Return bytes of a new PPTX with translated text and the same layout.

    Args:
        pptx_bytes:   Original PPTX file bytes.
        slide_texts:  Output of extract_slide_texts (which shape to update).
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
            shape    = slide.shapes[box["shape_idx"]]
            new_text = slide_replacements[box_idx]
            _replace_text_in_shape(shape, new_text)

    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()
