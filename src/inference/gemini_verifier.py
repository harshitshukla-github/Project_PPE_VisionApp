"""Gemini second opinion for compliance checks the detector cannot settle.

When YOLO fails to find a required item, that is not evidence the worker
lacks it — ``safety_vest`` is missed roughly one time in seven, and some
classes far more often. Instead of stopping at ``NEEDS_REVIEW``, the photo
itself is shown to Gemini, which answers per item: ``worn``, ``not_worn``
or ``uncertain``. :func:`src.inference.compliance.apply_verification` then
turns those findings into the final verdict.

Design choices, all in favour of accuracy over cost:

* Gemini is asked only about the items the detector missed, and is not told
  what the detector concluded, so the two opinions stay independent.
* ``uncertain`` is a first-class answer. A body part out of frame, a dark or
  blurry photo, or no visible person must never become ``worn`` or
  ``not_worn`` by guesswork.
* Any failure — no API key, SDK missing, timeout, unparseable reply — returns
  ``None`` and the verdict stays ``NEEDS_REVIEW``. An outage can neither
  clear a worker nor accuse one.

Privacy: unlike the explainer, which sends only text, this sends the worker's
photo to Google. ``PPB_GEMINI_VERIFY=false`` turns it off.

Usage::

    from src.inference.gemini_verifier import verify_items

    verification = verify_items(image_bgr, "welder", ["gloves", "goggles"])
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from src.inference.schemas import GeminiVerification, ItemStatus, ItemVerification
from src.settings import settings

if TYPE_CHECKING:
    import numpy as np

GEMINI_MODEL = "gemini-2.5-flash"

#: Longest image edge sent to Gemini. Large enough to resolve gloves and
#: goggles on a full-body shot, small enough to keep the upload quick.
MAX_EDGE_PX = 1568

#: Below this long edge the photo is a thumbnail: small items such as goggles
#: or gloves are a handful of pixels, and Gemini has been seen to answer
#: "not_worn" at confidence 1.00 on a 183x275 image. A confidence threshold
#: cannot catch a model that is confidently wrong, so such photos are not
#: verified at all and the verdict stays NEEDS_REVIEW. Phone photos are an
#: order of magnitude larger, so this only fires on images that were shrunk.
MIN_EDGE_PX = 400

#: What each class looks like, in words — the class names alone are ambiguous
#: ("mask" could be a welding mask, "boots" could be any shoe).
ITEM_DESCRIPTIONS: dict[str, str] = {
    "hardhat": "a hard hat or safety helmet, worn on the head",
    "safety_vest": (
        "high-visibility clothing covering the torso: a hi-vis vest, a reflective "
        "jacket, or a hi-vis coverall or jumpsuit all count"
    ),
    "goggles": "safety goggles or safety glasses, worn over the eyes",
    "mask": "a dust mask, respirator or face mask covering the nose and mouth",
    "gloves": "work or safety gloves, worn on the hands",
    "boots": (
        "sturdy closed work boots or safety shoes, worn on the feet. Sandals, "
        "flip-flops, slippers and bare feet are clearly not_worn. A protective "
        "toe cap cannot be seen in a photo, so do not judge it; light trainers "
        "or dress shoes that might or might not be safety footwear are uncertain"
    ),
    "ear_protection": "earmuffs or earplugs, worn on or in the ears",
    "face_shield": "a full transparent face shield, worn in front of the face",
    "harness": "a fall-arrest safety harness, strapped around the torso and legs",
}

_PROMPT_TEMPLATE = """You are checking personal protective equipment (PPE) in a \
photo taken on a work site. An automatic detector could not find the items \
listed below, and you are the second opinion.

Job category: {job}

Judge the worker in the foreground. Ignore small or distant people in the \
background. If two or more people are prominent in the foreground, judge all \
of them together: an item is "worn" only if every one of them is wearing it, \
and "not_worn" if any one of them clearly is not.

For each item choose exactly one status:
- "worn": the item is clearly visible and worn in its proper place.
- "not_worn": the part of the body where the item belongs is clearly visible, \
and the item is clearly not on it. An item held in a hand, hanging from a belt \
or lying nearby is "not_worn".
- "uncertain": that part of the body is out of frame, hidden, too small, too \
dark or too blurry to judge, or there is no person in the photo. Choose this \
whenever you are not sure. Never guess. If the photo is low resolution and the \
area you would need to inspect is only a few pixels across, that is \
"uncertain" too, however likely one answer seems.

Items to check:
{items}

Return one entry per item, using exactly the item names above. "confidence" is \
how sure you are of the status you chose, from 0 to 1. "reason" is one short \
sentence naming what you see."""


class _ReplyItem(BaseModel):
    """One entry of the JSON Gemini is asked to return."""

    item: str
    status: ItemStatus
    confidence: float
    reason: str


class _Reply(BaseModel):
    """Top-level JSON shape requested from Gemini."""

    items: list[_ReplyItem]


def _norm(name: str) -> str:
    """Normalise an item name so 'Safety Vest' matches 'safety_vest'."""
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def encode_for_gemini(image: np.ndarray) -> bytes:
    """Encode a BGR image as JPEG, downscaled so its long edge fits the cap.

    Args:
        image: BGR image array as produced by the API's upload decoder.

    Returns:
        JPEG bytes.

    Raises:
        ValueError: If the image cannot be encoded.
    """
    import cv2  # noqa: PLC0415

    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / long_edge
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode image as JPEG")
    return buffer.tobytes()


def build_prompt(job_category: str, items: list[str]) -> str:
    """Render the verifier prompt for a job and a list of items.

    Args:
        job_category: Job key from the PPE matrix.
        items: PPE class names to check.

    Returns:
        Prompt text.
    """
    lines = [
        f"- {item}: {ITEM_DESCRIPTIONS.get(item, item.replace('_', ' '))}"
        for item in items
    ]
    return _PROMPT_TEMPLATE.format(job=job_category, items="\n".join(lines))


def parse_reply(text: str, items: list[str]) -> list[ItemVerification]:
    """Turn Gemini's JSON reply into one finding per requested item.

    The reply is never trusted to be complete or well-formed. An item Gemini
    left out, a status outside the three allowed, or a confidence that is not
    a number all come back as ``uncertain`` — the reading that changes nothing.

    Args:
        text: Raw response text, expected to be JSON.
        items: The PPE class names that were asked about, in order.

    Returns:
        One :class:`ItemVerification` per requested item, in the same order.

    Raises:
        ValueError: If the text is not JSON or has no list of entries.
    """
    payload: Any = json.loads(text)
    entries = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("reply has no list of items")

    by_name: dict[str, dict] = {}
    for entry in entries:
        if isinstance(entry, dict) and "item" in entry:
            by_name.setdefault(_norm(entry["item"]), entry)

    findings: list[ItemVerification] = []
    for item in items:
        entry = by_name.get(_norm(item))
        if entry is None:
            findings.append(ItemVerification(
                item=item, status=ItemStatus.UNCERTAIN, confidence=0.0,
                reason="Gemini did not report on this item.",
            ))
            continue

        try:
            status = ItemStatus(str(entry.get("status", "")).strip().lower())
        except ValueError:
            status = ItemStatus.UNCERTAIN
        try:
            confidence = min(1.0, max(0.0, float(entry.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
            status = ItemStatus.UNCERTAIN

        findings.append(ItemVerification(
            item=item,
            status=status,
            confidence=round(confidence, 3),
            reason=str(entry.get("reason", "")).strip()[:300],
        ))
    return findings


def verify_items(
    image: np.ndarray,
    job_category: str,
    items: list[str],
) -> GeminiVerification | None:
    """Ask Gemini whether the foreground worker(s) in the photo wear each item.

    Args:
        image: BGR image array of the uploaded photo.
        job_category: Job key from the PPE matrix (context for the prompt).
        items: PPE class names the detector did not find.

    Returns:
        Per-item findings, or ``None`` when verification is switched off, has
        nothing to check, or could not be completed for any reason. ``None``
        always means "leave the verdict as it is".
    """
    if not items or not settings.ppb_gemini_verify:
        return None
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY not set — Gemini verification skipped.")
        return None

    long_edge = max(image.shape[:2])
    if long_edge < MIN_EDGE_PX:
        logger.warning(
            f"Photo is only {long_edge}px on its long edge (< {MIN_EDGE_PX}) — too "
            "small to verify PPE reliably; Gemini verification skipped."
        )
        return None

    start = time.perf_counter()
    try:
        from google import genai  # noqa: PLC0415 — optional dependency
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(
            api_key=settings.google_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.ppb_gemini_timeout_s * 1000)
            ),
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=encode_for_gemini(image), mime_type="image/jpeg"
                ),
                build_prompt(job_category, items),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_Reply,
            ),
        )
        findings = parse_reply(response.text or "", items)
    except Exception as exc:  # any failure leaves the verdict untouched
        logger.warning(f"Gemini verification failed ({exc}) — verdict left as is.")
        return None

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
    summary = ", ".join(f"{f.item}={f.status.value}@{f.confidence:.2f}" for f in findings)
    logger.info(f"Gemini verification [{job_category}] in {elapsed_ms} ms: {summary}")
    return GeminiVerification(model=GEMINI_MODEL, items=findings, elapsed_ms=elapsed_ms)
