"""Request validation and orchestration for the normalize endpoint."""
from __future__ import annotations

from .errors import ApiError
from .normalize import DiscontinuityAnchor, SegmentInput, normalize_segments
from .webvtt import parse_segment

MAX_SEGMENTS = 64
MAX_PAYLOAD_BYTES = 1 << 20  # 1 MiB of segment text per request
MAX_INT64 = (1 << 63) - 1
MAX_DISCONTINUITIES = 8


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError("INVALID_REQUEST", f"{name} must be an integer")
    if not 0 <= value <= MAX_INT64:
        raise ApiError("INVALID_REQUEST", f"{name} must be between 0 and {MAX_INT64}")
    return value


def _parse_discontinuity_anchors(body: dict) -> list[tuple[int, int]] | None:
    """Validate the shape of the optional discontinuityAnchors field.

    Returns ``(sequence, anchorTicks)`` pairs in declared order, or ``None``
    when the field is omitted.  Target existence and the first-segment rule
    are checked later, once the request's sequences are known.
    """
    raw = body.get("discontinuityAnchors")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ApiError("INVALID_REQUEST", "discontinuityAnchors must be an array")
    if len(raw) > MAX_DISCONTINUITIES:
        raise ApiError(
            "INVALID_REQUEST",
            f"discontinuityAnchors may contain at most {MAX_DISCONTINUITIES} entries, "
            f"got {len(raw)}",
        )

    anchors: list[tuple[int, int]] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ApiError(
                "INVALID_REQUEST", f"discontinuityAnchors[{position}] must be an object"
            )
        raw_sequence = item.get("sequence")
        sequence: int
        if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, int) or raw_sequence < 0:
            raise ApiError(
                "INVALID_REQUEST",
                f"discontinuityAnchors[{position}].sequence must be a non-negative integer",
            )
        sequence = raw_sequence
        try:
            anchor_ticks = _require_int(
                item.get("anchorTicks"), f"discontinuityAnchors[{position}].anchorTicks"
            )
        except ApiError as exc:
            raise exc.with_segment(sequence)
        if anchors:
            previous = anchors[-1][0]
            if sequence <= previous:
                raise ApiError(
                    "INVALID_REQUEST",
                    f"discontinuity anchor sequences must be strictly increasing: "
                    f"{sequence} does not follow {previous}",
                    segment=sequence,
                )
        anchors.append((sequence, anchor_ticks))
    return anchors


def _resolve_discontinuity_anchors(
    raw_anchors: list[tuple[int, int]] | None, segments: list[SegmentInput]
) -> list[DiscontinuityAnchor] | None:
    """Map declared ``(sequence, anchorTicks)`` pairs onto ordered segments."""
    if raw_anchors is None:
        return None

    first_sequence = segments[0].sequence
    last_sequence = segments[-1].sequence
    resolved: list[DiscontinuityAnchor] = []
    for sequence, anchor_ticks in raw_anchors:
        if sequence == first_sequence:
            raise ApiError(
                "INVALID_REQUEST",
                "a discontinuity anchor must not target the first segment; "
                "use anchorTicks for it",
                segment=sequence,
            )
        if not first_sequence < sequence <= last_sequence:
            raise ApiError(
                "ANCHOR_TARGET_NOT_FOUND",
                f"discontinuity anchor targets unknown segment sequence {sequence}",
                segment=sequence,
            )
        resolved.append(
            DiscontinuityAnchor(
                segment_index=sequence - first_sequence, anchor_ticks=anchor_ticks
            )
        )
    return resolved


def normalize_request(body: object) -> dict:
    if not isinstance(body, dict):
        raise ApiError("INVALID_REQUEST", "request body must be a JSON object")

    anchor = _require_int(body.get("anchorTicks"), "anchorTicks")
    max_interval = _require_int(body.get("maxAnchorIntervalTicks"), "maxAnchorIntervalTicks")
    raw_anchors = _parse_discontinuity_anchors(body)

    raw_segments = body.get("segments")
    if not isinstance(raw_segments, list):
        raise ApiError("INVALID_REQUEST", "segments must be an array")
    if not 1 <= len(raw_segments) <= MAX_SEGMENTS:
        raise ApiError(
            "SEGMENT_COUNT_OUT_OF_RANGE",
            f"expected 1..{MAX_SEGMENTS} segments, got {len(raw_segments)}",
        )

    items: list[tuple[int, str]] = []
    total_bytes = 0
    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ApiError("INVALID_REQUEST", f"segments[{position}] must be an object")
        sequence = _require_int(raw.get("sequence"), f"segments[{position}].sequence")
        content = raw.get("content")
        if not isinstance(content, str):
            raise ApiError(
                "INVALID_REQUEST",
                f"segments[{position}].content must be a string",
                segment=sequence,
            )
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError:
            raise ApiError(
                "INVALID_REQUEST",
                "segment content must be valid UTF-8 text",
                segment=sequence,
            )
        total_bytes += len(encoded)
        items.append((sequence, content))

    items.sort(key=lambda item: item[0])
    for (prev_seq, _), (seq, _) in zip(items, items[1:]):
        if seq != prev_seq + 1:
            raise ApiError(
                "SEGMENTS_NOT_CONSECUTIVE",
                f"segment sequence {seq} does not follow {prev_seq}",
                segment=seq,
            )

    if total_bytes > MAX_PAYLOAD_BYTES:
        raise ApiError(
            "PAYLOAD_TOO_LARGE",
            f"segment text totals {total_bytes} bytes, limit is {MAX_PAYLOAD_BYTES}",
        )

    segments: list[SegmentInput] = []
    for sequence, content in items:
        try:
            parsed = parse_segment(content)
        except ApiError as exc:
            raise exc.with_segment(sequence)
        segments.append(SegmentInput(sequence=sequence, parsed=parsed))

    discontinuities = _resolve_discontinuity_anchors(raw_anchors, segments)

    cues = normalize_segments(anchor, max_interval, segments, discontinuities)
    return {
        "cues": [
            {
                "segment": cue.segment,
                "index": cue.index,
                "startTicks": cue.start_ticks,
                "endTicks": cue.end_ticks,
                "text": cue.text,
            }
            for cue in cues
        ]
    }
