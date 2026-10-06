"""Unwrap 33-bit MPEG-TS timestamps onto an absolute 90 kHz timeline.

Each segment's ``X-TIMESTAMP-MAP`` binds its LOCAL reference time to a
33-bit MPEGTS value.  The absolute position of the first segment's map
point is given by the request anchor; every later segment's map point is
placed at the unique value ``MPEGTS + k * 2**33`` (``k >= 0``) that lies
within ``maxAnchorIntervalTicks`` of the previous segment's map point.
When no such ``k`` exists the anchors are incompatible; when more than one
exists the unwrap is not unique.

An encoder restart starts a fresh MPEGTS epoch: without help the restarted
segments would be mis-spliced onto the pre-restart wraparound chain.
``discontinuity_anchors`` marks such boundaries: the map point of the
declared segment is pinned to the given absolute tick, no adjacency
window is applied across the boundary (the new epoch may start arbitrarily
later on the unified timeline), and unique unwrapping resumes relative to
that anchor inside the new region.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import ApiError
from .webvtt import MPEGTS_MODULUS, ParsedSegment

TICKS_PER_MS = 90  # 90 000 ticks per second on the MPEG-TS clock


@dataclass
class DiscontinuityAnchor:
    """Pinned absolute map-point position for a segment after an encoder
    restart (a fresh MPEGTS epoch)."""

    sequence: int
    anchor_ticks: int


@dataclass
class SegmentInput:
    sequence: int
    parsed: ParsedSegment


@dataclass
class NormalizedCue:
    segment: int
    index: int
    start_ticks: int
    end_ticks: int
    text: str


def _unique_window_position(
    mpegts: int,
    prev_position: int,
    max_interval_ticks: int,
    sequence: int,
) -> int:
    """Find the unique ``mpegts + k*2**33`` (k >= 0) within the adjacency
    window around ``prev_position``; raise a stable error otherwise."""
    lo = prev_position - max_interval_ticks
    hi = prev_position + max_interval_ticks
    # Candidates are mpegts + k*MOD with lo <= candidate <= hi and candidate >= 0.
    k_min = -((mpegts - lo) // MPEGTS_MODULUS)  # ceil((lo - mpegts) / MOD)
    k_max = (hi - mpegts) // MPEGTS_MODULUS     # floor((hi - mpegts) / MOD)
    k_min = max(k_min, 0)                        # absolute ticks never go negative
    if k_max < k_min:
        raise ApiError(
            "ANCHOR_INCOMPATIBLE",
            f"no unwrapped position for MPEGTS {mpegts} within "
            f"{max_interval_ticks} ticks of previous map point {prev_position}",
            segment=sequence,
        )
    if k_max > k_min:
        raise ApiError(
            "UNWRAP_NOT_UNIQUE",
            f"multiple unwrapped positions for MPEGTS {mpegts} within "
            f"{max_interval_ticks} ticks of previous map point {prev_position}",
            segment=sequence,
        )
    return mpegts + k_min * MPEGTS_MODULUS


def unwrap_map_positions(
    anchor_ticks: int,
    max_interval_ticks: int,
    segments: list[SegmentInput],
    discontinuity_anchors: list[DiscontinuityAnchor] | None = None,
) -> list[int]:
    """Return the absolute 90 kHz tick of each segment's map point.

    Segments carrying a discontinuity anchor start a new continuous
    region: their position is pinned to the anchor (no adjacency window
    versus the previous segment), provided it stays strictly later than
    the last map point fixed before the boundary.  Unwrapping inside each
    region follows the unique-window rule.
    """
    pins = {anchor.sequence: anchor.anchor_ticks for anchor in discontinuity_anchors or []}

    first = segments[0]
    if anchor_ticks % MPEGTS_MODULUS != first.parsed.mpegts:
        raise ApiError(
            "ANCHOR_INCOMPATIBLE",
            f"anchor {anchor_ticks} is incompatible with the first segment's "
            f"MPEGTS {first.parsed.mpegts} (mod {MPEGTS_MODULUS})",
            segment=first.sequence,
        )

    positions = [anchor_ticks]
    for seg in segments[1:]:
        pinned = pins.get(seg.sequence)
        prev = positions[-1]
        if pinned is not None:
            if pinned % MPEGTS_MODULUS != seg.parsed.mpegts:
                raise ApiError(
                    "ANCHOR_INCOMPATIBLE",
                    f"discontinuity anchor {pinned} is incompatible with segment "
                    f"{seg.sequence} MPEGTS {seg.parsed.mpegts} (mod {MPEGTS_MODULUS})",
                    segment=seg.sequence,
                )
            if pinned <= prev:
                raise ApiError(
                    "ANCHOR_INCOMPATIBLE",
                    f"discontinuity anchor {pinned} for segment {seg.sequence} "
                    f"must be later than the previous map point {prev}",
                    segment=seg.sequence,
                )
            positions.append(pinned)
        else:
            positions.append(
                _unique_window_position(
                    seg.parsed.mpegts, prev, max_interval_ticks, seg.sequence
                )
            )
    return positions


def normalize_segments(
    anchor_ticks: int,
    max_interval_ticks: int,
    segments: list[SegmentInput],
    discontinuity_anchors: list[DiscontinuityAnchor] | None = None,
) -> list[NormalizedCue]:
    """Project every cue onto the absolute timeline and order the result by
    (absolute start, segment sequence, in-segment order)."""
    positions = unwrap_map_positions(
        anchor_ticks, max_interval_ticks, segments, discontinuity_anchors
    )
    cues: list[NormalizedCue] = []
    for seg, position in zip(segments, positions):
        base = position - seg.parsed.local_map_ms * TICKS_PER_MS
        for index, cue in enumerate(seg.parsed.cues):
            cues.append(
                NormalizedCue(
                    segment=seg.sequence,
                    index=index,
                    start_ticks=base + cue.start_ms * TICKS_PER_MS,
                    end_ticks=base + cue.end_ms * TICKS_PER_MS,
                    text=cue.text,
                )
            )
    cues.sort(key=lambda c: (c.start_ticks, c.segment, c.index))
    return cues
